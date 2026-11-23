"""W8A8 linear layers and zero-copy weight files for our MLX servers (rerank, embedding, decider), 2026-11-17.

W8A8 (as ComfyUI "int8_convrot", used by h3-image-mlx and our mflux fork for Qwen-Image): each projection is rotated
by a block-diagonal Hadamard matrix (256-wide groups along the input dimension) and quantized to int8 with one float32
scale per output row. At run time the activations get the same rotation and a per-row int8 quantization, so the
rotations cancel in the product: y = (q(x H) q(W H)^T) * x_scale * w_scale ~ x W^T. The GEMM runs on the M5
TensorOps INT8 path (mlx-serve kernel, now in MLX: mx.fast.convrot_*). Projections that read the same input (q/k/v, gate/up) share
one activation quantization.

Zero-copy: `save` writes a standard safetensors file whose large tensors start on a page; `load` maps it read-only
and wraps those tensors as MLX arrays without a copy (the GPU reads the page cache; an idle process holds no
anonymous memory for them and the kernel may drop the pages under pressure).
"""
import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np

GROUP = 256

_H4 = np.ones((4, 4), dtype=np.float32)
np.fill_diagonal(_H4[:, ::-1], -1)
H256 = mx.array(np.kron(np.kron(_H4, _H4), np.kron(_H4, _H4)) / 16)

# The kernels live in MLX (ain-soph/mlx `serving`, mx.fast.convrot_*): the fused rotate + quantize kernel and the
# TensorOps INT8 GEMM (mlx-serve's, MIT) with edge tiles kept inside their tensors; float product elsewhere.


def rotate_quantize(x):
    """Per-row int8 quantization of x H256 (x: rows x K, K a multiple of 256) with float32 scales. The int8 rows are
    allocated up to a multiple of 128 (the largest GEMM tile height); the extra rows only feed rows never stored."""
    return mx.fast.convrot_quantize(x)


def gemm(xq, xs, weight, scale, dtype):
    return mx.fast.convrot_matmul_quantized(xq, xs, weight, scale, dtype)


def quantize_weight(w):
    """(out, in) float weight -> int8 rows of W H256 and float32 per-row scales (float32 math)."""
    return mx.fast.convrot_quantize_weight(w)


# The last input quantized: q/k/v (and gate/up) receive the same array, so it is rotated and quantized once.
_last = [None, None, None]


def release():
    """Drop the cached activation quantization (call after evaluating a forward pass)."""
    _last[:] = [None, None, None]


class W8A8Linear(nn.Module):
    """Drop-in for a bias-free nn.Linear holding an int8 convrot weight and its per-row float32 scales."""

    def __init__(self, input_dims, output_dims):
        super().__init__()
        if input_dims % GROUP:
            raise ValueError(f"W8A8 needs the input width to be a multiple of {GROUP}, got {input_dims}")
        self.weight = mx.zeros((output_dims, input_dims), dtype=mx.int8)
        self.scale = mx.zeros((output_dims,), dtype=mx.float32)

    def __call__(self, x):
        if _last[0] is not x:
            _last[:] = [x, *rotate_quantize(x.reshape(-1, x.shape[-1]))]
        y = gemm(_last[1], _last[2], self.weight, self.scale, x.dtype)
        return y.reshape(*x.shape[:-1], self.weight.shape[0])


def linear_paths(model, names):
    """Paths of the nn.Linear / nn.QuantizedLinear modules whose last name is in `names`."""
    return [p for p, m in model.named_modules()
            if isinstance(m, (nn.Linear, nn.QuantizedLinear)) and p.rsplit(".", 1)[-1] in names]


def linear_paths_w8a8(model):
    """Paths of the W8A8Linear modules."""
    return [p for p, m in model.named_modules() if isinstance(m, W8A8Linear)]


def replace(model, paths):
    """Swap the modules at `paths` for empty W8A8Linear layers (their weights come from load_weights)."""
    from mlx.utils import tree_unflatten
    modules = dict(model.named_modules())
    swaps = []
    for p in paths:
        m = modules[p]
        out, k = m.weight.shape[0], (m.weight.shape[1] * 32 // m.bits if isinstance(m, nn.QuantizedLinear)
                                      else m.weight.shape[1])
        swaps.append((p, W8A8Linear(k, out)))
    model.update_modules(tree_unflatten(swaps))


def float_weight(m):
    if isinstance(m, nn.QuantizedLinear):
        return mx.dequantize(m.weight, m.scales, m.biases, group_size=m.group_size, bits=m.bits)
    return m.weight


# ---- zero-copy safetensors -------------------------------------------------------------------------------------

def save(path, tensors, metadata=None):
    """Standard safetensors written page-aligned by MLX (ain-soph/mlx `serving`: save_safetensors(page_align=True)):
    header padded to a page, the tensors whose size is a whole number of pages first, then the largest other tensor
    (e.g. an embedding table), so each of them starts on a page and `load` maps it; the small rest follows."""
    tmp = f"{path}.partial.safetensors"
    mx.save_safetensors(tmp, tensors, metadata, page_align=True)
    os.replace(tmp, path)


def load(path, copy=False):
    """Tensors of a safetensors file; each one that starts on a page is a zero-copy view of a read-only mapping
    (mx.load(zero_copy=True) of the ain-soph/mlx fork), the rest are read. copy=True: plain mx.load."""
    return mx.load(str(path), zero_copy=not copy)
