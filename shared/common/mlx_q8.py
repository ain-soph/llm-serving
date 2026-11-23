"""8-bit weights + float activations (W8A16) for MLX linear layers (2026-11-17).

Used where W8A8 (mlx_w8a8.py) gives no measurable speed-up but the float weights cost memory: the weights become
symmetric 8-bit like llama.cpp Q8_0 (blocks of 32 along the input dimension, d = max|w| / 127, q = round(w / d) in
[-127, 127]) stored in MLX's affine 8-bit format (u = q + 128, scale = d, bias = -128 * d), so MLX's quantized_matmul
runs them with the activations left in float. The scales and biases take the activation dtype (bfloat16 scales with
bfloat16 activations, float16 with float16): a mixed pair makes MLX promote to float32 and take its slow path.
d is rounded *up* to that dtype, so no weight is clipped; bias = -128 * d is exact. On the GPU the dequantized
weights match q * d to the activation dtype's rounding (MLX's CPU kernel rounds u * scale in bfloat16, much worse:
measure on the GPU). MLX's own default 8-bit (min/max affine) is not used: on Qwen3-Reranker-8B its per-group offset
gave 5-9x the Q8_0 error.
"""
import mlx.core as mx
from mlx import nn

GROUP = 32
# relative step that, applied before rounding to the dtype, lands on a representable value >= d
_UP = {mx.bfloat16: 2.0 ** -8, mx.float16: 2.0 ** -11, mx.float32: 0.0}


def quantize_weight(w: mx.array, dtype: mx.Dtype) -> tuple[mx.array, mx.array, mx.array]:
    """[out, in] float -> (uint32 [out, in/4], scales, biases [out, in/32]) in `dtype`."""
    w = w.astype(mx.float32)
    out_dim, in_dim = w.shape
    blocks = w.reshape(out_dim, in_dim // GROUP, GROUP)
    d = mx.max(mx.abs(blocks), axis=-1) / 127
    d = (d * (1 + _UP[dtype])).astype(dtype)
    df = d.astype(mx.float32)
    q = mx.clip(mx.round(blocks / mx.where(df == 0, 1, df)[..., None]), -127, 127)
    u = (q + 128).astype(mx.uint32).reshape(out_dim, in_dim // 4, 4)
    packed = u[..., 0] | (u[..., 1] << 8) | (u[..., 2] << 16) | (u[..., 3] << 24)
    return packed, d, (-128 * df).astype(dtype)


def layer(in_dims: int, out_dims: int) -> nn.QuantizedLinear:
    return nn.QuantizedLinear(in_dims, out_dims, bias=False, group_size=GROUP, bits=8)
