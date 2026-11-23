"""Per-class W8A8 / W8A16 conversion of the MLX ASR models (Whisper large-v3, Qwen3-ASR 1.7B), 2026-11-17.

A plan maps module-path regexes to a mode:
  w8a8: int8 convrot weights + per-row int8 activations on the M5 TensorOps GEMM (common/mlx_w8a8.py);
        a bias, if the layer has one, is added in the activation dtype afterwards.
  q8:   8-bit weights (Q8_0-style symmetric blocks of 32) with float activations through MLX's quantized_matmul
        (common/mlx_q8.py), i.e. W8A16. Meant for the token-by-token decoders, which read every
        weight once per token and are bandwidth bound.
Conversion runs in-process from the float weights (no files); quantize on the GPU, the CPU kernels round differently.
"""
import re
import sys
from pathlib import Path

import mlx.core as mx
from mlx import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import mlx_q8  # noqa: E402
import mlx_w8a8  # noqa: E402


class W8A8Biased(mlx_w8a8.W8A8Linear):
    """W8A8Linear plus an optional float bias (Whisper's query/value/out/mlp, Qwen3-ASR's audio encoder)."""

    def __init__(self, input_dims, output_dims, bias):
        super().__init__(input_dims, output_dims)
        if bias is not None:
            self.bias = bias

    def __call__(self, x):
        y = super().__call__(x)
        return y + self.bias if "bias" in self else y


def _w8a8(lin):
    out, k = lin.weight.shape
    m = W8A8Biased(k, out, lin.get("bias"))
    m.weight, m.scale = mlx_w8a8.quantize_weight(lin.weight)
    return m


def _q8(lin, dtype):
    out, k = lin.weight.shape
    bias = lin.get("bias")
    m = nn.QuantizedLinear(k, out, bias=bias is not None, group_size=mlx_q8.GROUP, bits=8)
    m.weight, m.scales, m.biases = mlx_q8.quantize_weight(lin.weight, dtype)
    if bias is not None:
        m.bias = bias
    return m


# Whisper: encoder blocks (1500 frames per 30 s window: compute bound) and decoder blocks (one token per step).
# Cross-attention key/value see the 1500 encoder frames once per window, so they are compute bound too.
WHISPER = {
    "enc_q": [(r"^encoder\.blocks\.\d+\.attn\.query$", "w8a8")],
    "enc_kv": [(r"^encoder\.blocks\.\d+\.attn\.(key|value)$", "w8a8")],
    "enc_out": [(r"^encoder\.blocks\.\d+\.attn\.out$", "w8a8")],
    "enc_mlp1": [(r"^encoder\.blocks\.\d+\.mlp1$", "w8a8")],
    "enc_mlp2": [(r"^encoder\.blocks\.\d+\.mlp2$", "w8a8")],
    "dec_xkv": [(r"^decoder\.blocks\.\d+\.cross_attn\.(key|value)$", "w8a8")],
    "dec_w8a8": [(r"^decoder\.blocks\.\d+\.(attn|cross_attn)\.(query|out)$", "w8a8"),
                 (r"^decoder\.blocks\.\d+\.attn\.(key|value)$", "w8a8"),
                 (r"^decoder\.blocks\.\d+\.mlp[12]$", "w8a8")],
    "dec_q8": [(r"^decoder\.blocks\.\d+\.(attn|cross_attn)\.(query|out)$", "q8"),
               (r"^decoder\.blocks\.\d+\.attn\.(key|value)$", "q8"),
               (r"^decoder\.blocks\.\d+\.mlp[12]$", "q8")],
}
WHISPER["enc_all"] = sum((WHISPER[k] for k in ("enc_q", "enc_kv", "enc_out", "enc_mlp1", "enc_mlp2")), [])

# Qwen3-ASR: audio encoder (13 tokens per second of audio), text decoder (prompt prefill with the audio tokens, then
# one token per step; the same layers serve both), lm_head tied to the embedding.
QWEN = {
    "enc_qkv": [(r"^audio_tower\.layers\.\d+\.self_attn\.(q|k|v)_proj$", "w8a8")],
    "enc_out": [(r"^audio_tower\.layers\.\d+\.self_attn\.out_proj$", "w8a8")],
    "enc_fc1": [(r"^audio_tower\.layers\.\d+\.fc1$", "w8a8")],
    "enc_fc2": [(r"^audio_tower\.layers\.\d+\.fc2$", "w8a8")],
    "dec_w8a8": [(r"^model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj$", "w8a8"),
                 (r"^model\.layers\.\d+\.mlp\.(gate|up|down)_proj$", "w8a8")],
    "dec_q8_attn": [(r"^model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj$", "q8")],
    "dec_q8_gateup": [(r"^model\.layers\.\d+\.mlp\.(gate|up)_proj$", "q8")],
    "dec_q8_down": [(r"^model\.layers\.\d+\.mlp\.down_proj$", "q8")],
    # the tied head (151936 x 2048, 0.6 GB in fp16) is read on every decode step; a q8 copy unties it (+0.3 GB)
    "dec_q8_head": [(r"^lm_head$", "q8")],
}
QWEN["enc_all"] = sum((QWEN[k] for k in ("enc_qkv", "enc_out", "enc_fc1", "enc_fc2")), [])
QWEN["dec_q8"] = sum((QWEN[k] for k in ("dec_q8_attn", "dec_q8_gateup", "dec_q8_down")), [])


def plan(table, names):
    """'enc_all+dec_q8' -> list of (regex, mode); 'none' -> []."""
    out = []
    for n in names.split("+"):
        if n and n != "none":
            out += table[n]
    return out


def convert(model, rules, dtype):
    """Swap every nn.Linear whose path matches a rule; returns {mode: count}. The first matching rule wins."""
    from mlx.utils import tree_unflatten
    compiled = [(re.compile(r), m) for r, m in rules]
    swaps, counts = [], {}
    for path, mod in model.named_modules():
        if type(mod) is not nn.Linear:
            continue
        mode = next((m for r, m in compiled if r.search(path)), None)
        if mode is None:
            continue
        if mode == "w8a8" and mod.weight.shape[1] % mlx_w8a8.GROUP:
            continue  # input width not a multiple of 256
        new = _w8a8(mod) if mode == "w8a8" else _q8(mod, dtype)
        mx.eval(new.parameters())
        swaps.append((path, new))
        counts[mode] = counts.get(mode, 0) + 1
    model.update_modules(tree_unflatten(swaps))
    return counts
