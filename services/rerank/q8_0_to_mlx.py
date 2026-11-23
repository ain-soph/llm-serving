"""Write an MLX 8-bit model whose linear weights are exactly llama.cpp Q8_0 (2026-11-16).

  .venv/bin/python services/rerank/q8_0_to_mlx.py HF_DIR OUT_DIR [--act-dtype bfloat16|float16]

Q8_0: blocks of 32 along the input dimension, d = max|w| / 127 stored as float16, q = round(w / d) in [-127, 127].
MLX 8-bit affine (group 32) dequantizes w = scale * u + bias with u in [0, 255]; u = q + 128, scale = d and
bias = -128 * d reproduce every Q8_0 weight exactly as long as scale and bias stay float16 (bias is d times a power
of two, so it is exact too). MLX's default 8-bit is affine min/max with bf16 scale and bias: its per-group offset error
does not cancel and gave 5-9x the Q8_0 score error on Qwen3-Reranker-8B.
Embeddings and lm_head are Q8_0 too (as llama.cpp's Q8_0 GGUF); norms stay in the activation dtype.
Cost vs a GGUF: the bias tensor (redundant, -128 * scale) adds 0.5 bit/weight, which MLX's kernels need.

"""
import json, shutil, sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.utils import save_model

def q8_0(w):
    """(packed uint32 weight, float16 scales, float16 biases) of the Q8_0 rounding of w [out, in]."""
    w = w.astype(mx.float32)
    out_dim, in_dim = w.shape
    blocks = w.reshape(out_dim, in_dim // 32, 32)
    d = (mx.max(mx.abs(blocks), axis=-1) / 127).astype(mx.float16)            # Q8_0 stores d as fp16
    inv = mx.where(d == 0, 0, 1 / d.astype(mx.float32))
    q = mx.clip(mx.round(blocks * inv[..., None]), -127, 127)
    u = (q + 128).astype(mx.uint32).reshape(out_dim, in_dim // 4, 4)
    packed = u[..., 0] | (u[..., 1] << 8) | (u[..., 2] << 16) | (u[..., 3] << 24)
    return packed, d, (-128 * d.astype(mx.float32)).astype(mx.float16)


def keep(path, module):
    return isinstance(module, (nn.Linear, nn.Embedding)) and module.weight.shape[-1] % 32 == 0


def apply_q8_0(model):
    """Quantize every Linear / Embedding (input dim divisible by 32) of model in place to exact Q8_0; returns the count."""
    originals = {p: m.weight for p, m in model.named_modules() if keep(p, m)}
    nn.quantize(model, group_size=32, bits=8, class_predicate=keep)
    for path, module in model.named_modules():
        if path in originals:
            packed, scales, biases = q8_0(originals[path])
            module.weight, module.scales, module.biases = packed, scales, biases
            mx.eval(module.weight, module.scales, module.biases)
    return len(originals)


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    act = getattr(mx, sys.argv[sys.argv.index("--act-dtype") + 1]) if "--act-dtype" in sys.argv else mx.bfloat16
    model, tok = load(str(src))
    model.set_dtype(act)
    count = apply_q8_0(model)
    out.mkdir(parents=True, exist_ok=True)
    save_model(out, model)
    config = json.loads((src / "config.json").read_text())
    config["quantization"] = {"group_size": 32, "bits": 8, "mode": "affine"}
    config["torch_dtype"] = "bfloat16" if act == mx.bfloat16 else "float16"
    (out / "config.json").write_text(json.dumps(config, indent=2))
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json",
                 "generation_config.json", "chat_template.jinja"):
        if (src / name).exists():
            shutil.copy(src / name, out / name)
    print(f"wrote {out}: {count} Q8_0 linears / embeddings, activations {act}")


if __name__ == "__main__":
    main()
