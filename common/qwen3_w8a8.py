"""Qwen3 backbones (mlx_lm Qwen3Model) as W8A8 + zero-copy checkpoints, for the reranker and embedding servers (2026-11-17).

  .venv/bin/python common/qwen3_w8a8.py rerank|embed SRC_DIR DST_DIR [--float] [--dtype bfloat16|float16]

SRC_DIR is one of our MLX copies (BF16 / FP16, or exact Q8_0 = MLX 8-bit affine with fp16 scales, which is
dequantized first). DST_DIR gets one zero-copy model.safetensors (see mlx_w8a8.save), the config and tokenizer files,
and w8a8.json. Every q/k/v/o and gate/up/down projection becomes W8A8 (int8 rows of W H256, float32 row scales);
embeddings and norms keep the activation dtype (that of the source's norms: bfloat16 or float16).
Extras kept beside the backbone: rerank -> "score_rows" (the "yes"/"no" rows of the output head, all the reranker
reads of it); embed -> "linear.weight" (voyage-4-nano's output projection, float32 as the server uses it).
--float: keep every weight as it is (zero-copy only, the same numbers as the source; an MLX-quantized source such as
the Q8_0 copies stays quantized).
--dtype: cast the float weights (and so the activations) first, e.g. voyage-4-nano to bfloat16: in float16 its
activations overflow on some 300+ token inputs (NaN vectors).
Conversion runs on the CPU (no GPU needed).
"""
import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3 import ModelArgs, Qwen3Model

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlx_w8a8  # noqa: E402

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
MARKER = "w8a8.json"


def is_w8a8(path):
    return (Path(path) / MARKER).exists()


def load(path, copy=False):
    """(Qwen3Model with W8A8 projections, extras dict, activation dtype) from a converted directory."""
    path = Path(path)
    cfg = json.loads((path / "config.json").read_text())
    cfg.pop("quantization", None)
    cfg.pop("quantization_config", None)
    weights = mlx_w8a8.load(path / "model.safetensors", copy=copy)
    extras = {k: weights.pop(k) for k in ("score_rows", "linear.weight") if k in weights}
    model = Qwen3Model(ModelArgs.from_dict(cfg))
    marker = json.loads((path / MARKER).read_text())
    if marker.get("quantized", True):
        mlx_w8a8.replace(model, mlx_w8a8.linear_paths(model, PROJECTIONS))
    elif q := marker.get("mlx_quantization"):
        nn.quantize(model, group_size=q["group_size"], bits=q["bits"],
                    class_predicate=lambda p, m: f"{p}.scales" in weights)
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters(), *extras.values())
    return model, extras, model.norm.weight.dtype


def _source(kind, src):
    """The source backbone as a Qwen3Model, plus the extras to keep."""
    cfg = json.loads((src / "config.json").read_text())
    if kind == "rerank":
        from mlx_lm import load as lm_load
        full, tok = lm_load(str(src))
        model = full.model
        tied = "lm_head" not in full
        head = model.embed_tokens if tied else full.lm_head
        pair = mx.array([tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")])
        rows = head.weight[pair]
        if "scales" in head:
            rows = mx.dequantize(rows, head.scales[pair], head.biases[pair], group_size=head.group_size, bits=head.bits)
        return model, {"score_rows": rows.astype(model.norm.weight.dtype)}
    weights = {}
    for f in sorted(src.glob("*.safetensors")):
        weights.update(mx.load(str(f)))
    weights = {k.removeprefix("model."): v for k, v in weights.items() if not k.startswith("lm_head")}
    extras = {"linear.weight": weights.pop("linear.weight").astype(mx.float32)} if "linear.weight" in weights else {}
    model = Qwen3Model(ModelArgs.from_dict(cfg))
    if q := cfg.get("quantization"):
        nn.quantize(model, group_size=q["group_size"], bits=q["bits"],
                    class_predicate=lambda p, m: f"{p}.scales" in weights)
    model.load_weights(list(weights.items()))
    return model, extras


def convert(kind, src, dst, quantize=True, dtype=None):
    src, dst = Path(src), Path(dst)
    mx.set_default_device(mx.cpu)
    model, extras = _source(kind, src)
    if dtype:
        model.set_dtype(getattr(mx, dtype))
        extras = {k: v if v.dtype == mx.float32 else v.astype(getattr(mx, dtype)) for k, v in extras.items()}
    dtype = model.norm.weight.dtype
    modules = dict(model.named_modules())
    tensors = {}
    for p in mlx_w8a8.linear_paths(model, PROJECTIONS) if quantize else []:
        q, s = mlx_w8a8.quantize_weight(mlx_w8a8.float_weight(modules[p]))
        mx.eval(q, s)
        tensors[f"{p}.weight"], tensors[f"{p}.scale"] = q, s
    if quantize:
        emb = model.embed_tokens
        table = (mx.dequantize(emb.weight, emb.scales, emb.biases, group_size=emb.group_size, bits=emb.bits)
                 if isinstance(emb, nn.QuantizedEmbedding) else emb.weight)
        tensors["embed_tokens.weight"] = table.astype(dtype)
    for name, value in tree_flatten(model.parameters()):
        if quantize and (name.startswith("embed_tokens.") or name.rsplit(".", 2)[-2] in PROJECTIONS):
            continue
        tensors[name] = value
    tensors.update(extras)
    mx.eval(list(tensors.values()))
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name != "model.safetensors.index.json":
            shutil.copy2(f, dst / f.name)
    mlx_w8a8.save(dst / "model.safetensors", tensors, {"format": "mlx", "w8a8": "convrot-256" if quantize else "none"})
    mlxq = None if quantize else json.loads((src / "config.json").read_text()).get("quantization")
    (dst / MARKER).write_text(json.dumps({"kind": kind, "source": src.name, "quantized": quantize, "mlx_quantization": mlxq,
                                          "group": mlx_w8a8.GROUP,
                                          "activations": str(dtype).split(".")[-1]}, indent=1) + "\n")
    total = sum(v.nbytes for v in tensors.values())
    print(f"{dst}: {len(tensors)} tensors, {total / 1e9:.3f} GB, activations {dtype}")


if __name__ == "__main__":
    convert(*sys.argv[1:4], quantize="--float" not in sys.argv,
            dtype=sys.argv[sys.argv.index("--dtype") + 1] if "--dtype" in sys.argv else None)
