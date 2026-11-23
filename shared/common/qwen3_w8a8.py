"""Qwen3 backbones (mlx_lm Qwen3Model) as W8A8 + zero-copy checkpoints, for the reranker and embedding servers (2026-11-17).

  .venv/bin/python common/qwen3_w8a8.py rerank|embed SRC_DIR DST_DIR [--float] [--dtype bfloat16|float16]
  .venv/bin/python common/qwen3_w8a8.py mix rerank|embed FLOAT_DIR DST_DIR qkv=w8a8,o=w8a8,gateup=w8a8,down=w8a16
      (2026-11-18: one copy with per-class precision, mix.json; the servers load it zero-copy like the others)

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
import mlx_q8  # noqa: E402
import mlx_w8a8  # noqa: E402

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
MARKER = "w8a8.json"
MIX = "mix.json"     # per-class precision copy (save_mix), 2026-11-18


def is_w8a8(path):
    return (Path(path) / MARKER).exists() or (Path(path) / MIX).exists()


def apply_plan(model, plan, dtype):
    """Per-class precision in place (2026-11-18): plan maps each projection name to "w8a8" (int8 convrot weights and
    activations), "w8a16" (symmetric 8-bit weights, float activations: mlx_q8.py) or "float". A projection that is
    already MLX 8-bit (the exact Q8_0 copies) counts as w8a16 and stays as it is; "float" keeps it 8-bit too."""
    from mlx.utils import tree_unflatten
    modules = dict(model.named_modules())
    swaps, params = [], []
    for p in mlx_w8a8.linear_paths(model, PROJECTIONS):
        m, kind = modules[p], plan[p.rsplit(".", 1)[-1]]
        if kind == "w8a8":
            q, s = mlx_w8a8.quantize_weight(mlx_w8a8.float_weight(m))
            swaps.append((p, mlx_w8a8.W8A8Linear(q.shape[1], q.shape[0])))
            params += [(f"{p}.weight", q), (f"{p}.scale", s)]
        elif kind == "w8a16" and not isinstance(m, nn.QuantizedLinear):
            packed, scales, biases = mlx_q8.quantize_weight(m.weight, dtype)
            swaps.append((p, mlx_q8.layer(m.weight.shape[1], m.weight.shape[0])))
            params += [(f"{p}.weight", packed), (f"{p}.scales", scales), (f"{p}.biases", biases)]
        elif kind not in ("w8a16", "float"):
            raise ValueError(f"{p}: unknown precision {kind}")
    if swaps:
        model.update_modules(tree_unflatten(swaps))
        model.load_weights(params, strict=False)
    mx.eval(model.parameters())
    return model


def save_mix(model, extras, src, dst, plan, kind):
    """Write the model as it is in memory (after apply_plan) to DST: one page-aligned model.safetensors mapped
    zero-copy by load(), the non-weight files of SRC (config, tokenizer), and mix.json (the plan)."""
    src, dst = Path(src), Path(dst)
    tensors = dict(tree_flatten(model.parameters()))
    tensors.update(extras)
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name not in (MARKER, MIX,
                                                                                  "model.safetensors.index.json"):
            shutil.copy2(f, dst / f.name)
    mlx_w8a8.save(dst / "model.safetensors", tensors, {"format": "mlx", "w8a8": "convrot-256 mix"})
    emb = model.embed_tokens
    (dst / MIX).write_text(json.dumps({
        "kind": kind, "source": src.name, "plan": plan, "w8a8_group": mlx_w8a8.GROUP,
        "w8a16": {"group_size": 32, "bits": 8},
        "embed_quantization": {"group_size": emb.group_size, "bits": emb.bits}
        if isinstance(emb, nn.QuantizedEmbedding) else None,
        "activations": str(model.norm.weight.dtype).split(".")[-1]}, indent=1) + "\n")
    return sum(v.nbytes for v in tensors.values())


def _load_mix(path, cfg, weights):
    mix = json.loads((path / MIX).read_text())
    model = Qwen3Model(ModelArgs.from_dict(cfg))
    from mlx.utils import tree_unflatten
    swaps = []
    for p, m in model.named_modules():
        if isinstance(m, nn.Linear) and p.rsplit(".", 1)[-1] in PROJECTIONS:
            out, k = m.weight.shape
            if f"{p}.scale" in weights:
                swaps.append((p, mlx_w8a8.W8A8Linear(k, out)))
            elif f"{p}.scales" in weights:
                swaps.append((p, nn.QuantizedLinear(k, out, bias=False, **mix["w8a16"])))
    model.update_modules(tree_unflatten(swaps))
    if q := mix.get("embed_quantization"):
        nn.quantize(model, class_predicate=lambda p, m: isinstance(m, nn.Embedding) and f"{p}.scales" in weights, **q)
    return model


def load(path, copy=False):
    """(Qwen3Model with W8A8 projections, extras dict, activation dtype) from a converted directory."""
    path = Path(path)
    cfg = json.loads((path / "config.json").read_text())
    cfg.pop("quantization", None)
    cfg.pop("quantization_config", None)
    weights = mlx_w8a8.load(path / "model.safetensors", copy=copy)
    extras = {k: weights.pop(k) for k in ("score_rows", "linear.weight") if k in weights}
    if (path / MIX).exists():
        model = _load_mix(path, cfg, weights)
        model.load_weights(list(weights.items()), strict=True)
        mx.eval(model.parameters(), *extras.values())
        return model, extras, model.norm.weight.dtype
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


def load_mixed(w8a8_path, float_path, classes, copy=False, q8_classes=()):
    """Per-class precision (2026-11-17, W8A8 only where it pays): the W8A8 copy, with the projections whose last name
    is NOT in `classes` swapped back to the float copy's weights (`--float` copy of the same model: plain float, or the
    exact Q8_0 MLX 8-bit for 8B). Both files are mapped; only the swapped-back float tensors are read from the second.
    q8_classes: swapped-back plain-float projections to quantize to 8-bit weights + float activations (W8A16,
    mlx_q8.py) while loading, saving half their memory where W8A8 brings no speed."""
    model, extras, dtype = load(w8a8_path, copy=copy)
    back = [p for p in mlx_w8a8.linear_paths_w8a8(model) if p.rsplit(".", 1)[-1] not in set(classes)]
    if back:
        marker = json.loads((Path(float_path) / MARKER).read_text())
        q = marker.get("mlx_quantization")
        f = mlx_w8a8.load(Path(float_path) / "model.safetensors", copy=copy)
        modules = dict(model.named_modules())
        swaps, params = [], []
        for p in back:
            out, k = modules[p].weight.shape
            if q:
                layer = nn.QuantizedLinear(k, out, bias=False, group_size=q["group_size"], bits=q["bits"])
                names = ("weight", "scales", "biases")
            elif p.rsplit(".", 1)[-1] in set(q8_classes):
                w = f[f"{p}.weight"]
                packed, scales, biases = mlx_q8.quantize_weight(w, w.dtype)
                swaps.append((p, mlx_q8.layer(k, out)))
                params += [(f"{p}.weight", packed), (f"{p}.scales", scales), (f"{p}.biases", biases)]
                continue
            else:
                layer, names = nn.Linear(k, out, bias=False), ("weight",)
            swaps.append((p, layer))
            params += [(f"{p}.{n}", f[f"{p}.{n}"]) for n in names]
        from mlx.utils import tree_unflatten
        model.update_modules(tree_unflatten(swaps))
        model.load_weights(params, strict=False)
        mx.eval(model.parameters())
    return model, extras, dtype


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


def mix(kind, src, dst, plan, dtype=None):
    """Per-class precision copy (2026-11-18): the zero-copy float copy SRC (--float output: BF16 / FP16, or the exact
    Q8_0 copy) with plan applied as the servers would, written by save_mix. plan: {projection: w8a8|w8a16|float}."""
    model, extras, act = load(src, copy=True)
    if dtype:
        model.set_dtype(getattr(mx, dtype))
        act = model.norm.weight.dtype
        extras = {k: v if v.dtype == mx.float32 else v.astype(act) for k, v in extras.items()}
    apply_plan(model, plan, act)
    total = save_mix(model, extras, src, dst, plan, kind)
    print(f"{dst}: {total / 1e9:.3f} GB, activations {act}, plan {plan}")


GROUPS = {"qkv": ("q_proj", "k_proj", "v_proj"), "o": ("o_proj",), "gateup": ("gate_proj", "up_proj"),
          "down": ("down_proj",)}

if __name__ == "__main__":
    if sys.argv[1] == "mix":    # mix rerank|embed SRC DST qkv=w8a8,o=w8a8,gateup=w8a8,down=float [--dtype float16]
        groups = dict(x.split("=") for x in sys.argv[5].split(","))
        mix(sys.argv[2], sys.argv[3], sys.argv[4], {c: groups[g] for g, cs in GROUPS.items() for c in cs},
            dtype=sys.argv[sys.argv.index("--dtype") + 1] if "--dtype" in sys.argv else None)
    else:
        convert(*sys.argv[1:4], quantize="--float" not in sys.argv,
                dtype=sys.argv[sys.argv.index("--dtype") + 1] if "--dtype" in sys.argv else None)
