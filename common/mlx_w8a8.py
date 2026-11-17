"""W8A8 linear layers and zero-copy weight files for our MLX servers (rerank, embedding, decider), 2026-11-17.

W8A8 (as ComfyUI "int8_convrot", used by h3-image-mlx and our mflux fork for Qwen-Image): each projection is rotated
by a block-diagonal Hadamard matrix (256-wide groups along the input dimension) and quantized to int8 with one float32
scale per output row. At run time the activations get the same rotation and a per-row int8 quantization, so the
rotations cancel in the product: y = (q(x H) q(W H)^T) * x_scale * w_scale ~ x W^T. The GEMM runs on the M5
TensorOps INT8 path (mlx-serve kernel, w8a8_kernels/). Projections that read the same input (q/k/v, gate/up) share
one activation quantization.

Zero-copy: `save` writes a standard safetensors file whose large tensors start on a page; `load` maps it read-only
and wraps those tensors as MLX arrays without a copy (the GPU reads the page cache; an idle process holds no
anonymous memory for them and the kernel may drop the pages under pressure).
"""
import json
import mmap
import os
import struct
from functools import cache
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

KERNELS = Path(__file__).resolve().parent / "w8a8_kernels"
GROUP = 256
PAGE = os.sysconf("SC_PAGE_SIZE")

_H4 = np.ones((4, 4), dtype=np.float32)
np.fill_diagonal(_H4[:, ::-1], -1)
H256 = mx.array(np.kron(np.kron(_H4, _H4), np.kron(_H4, _H4)) / 16)

# One threadgroup per row, one thread per position in a 256-wide group; H256 = H4 (x) H4 (x) H4 (x) H4 with
# H4 = ones - 2 * antidiagonal, so each base-4 digit is a 4-point butterfly. Same kernel as the mflux fork.
_ROTATE_QUANTIZE = """
    const uint t = thread_position_in_threadgroup.x;
    const ushort lane = thread_index_in_simdgroup;
    const uint row = threadgroup_position_in_grid.y;
    const uint K = x_shape[1];
    const uint groups = K / 256;
    threadgroup float buf[256];
    threadgroup float red[8];
    float v[MAX_GROUPS];
    float amax = 0.0f;
    device const T* xr = x + (size_t)row * K;
    for (uint g = 0; g < groups; ++g) {
        float a = float(xr[g * 256 + t]);
        for (uint s = 1; s < 256; s *= 4) {
            buf[t] = a;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint k = (t / s) % 4, base = t - k * s;
            a = buf[base] + buf[base + s] + buf[base + 2 * s] + buf[base + 3 * s] - 2.0f * buf[base + (3 - k) * s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        a *= 0.0625f;
        v[g] = a;
        amax = max(amax, fabs(a));
    }
    amax = simd_max(amax);
    if (lane == 0) red[simdgroup_index_in_threadgroup] = amax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    amax = red[0];
    for (uint i = 1; i < 8; ++i) amax = max(amax, red[i]);
    const float scale = max(amax / 127.0f, 1e-20f);
    device int8_t* qr = xq + (size_t)row * K;
    for (uint g = 0; g < groups; ++g) qr[g * 256 + t] = int8_t(clamp(rint(v[g] / scale), -127.0f, 127.0f));
    if (t == 0) xs[row] = scale;
"""


@cache
def _gemm_kernel():
    return mx.fast.metal_kernel(
        name="msv_w8a8_gemm",
        input_names=["xq", "wq", "xs", "ws"],
        output_names=["y"],
        header=(KERNELS / "mpp_tensor_ops.metal").read_text(),
        source=(KERNELS / "w8a8_gemm_body.metal").read_text() + "\n" + (KERNELS / "w8a8_epi_plain.metal").read_text(),
    )


@cache
def _rotate_quantize_kernel():
    return mx.fast.metal_kernel(name="convrot_rotate_quantize", input_names=["x"], output_names=["xq", "xs"],
                                source=_ROTATE_QUANTIZE)


def rotate_quantize(x):
    """Per-row int8 quantization of x H256 (x: rows x K, K a multiple of 256) with float32 scales."""
    rows, k = x.shape
    groups = k // GROUP
    return _rotate_quantize_kernel()(
        inputs=[x], template=[("T", x.dtype), ("MAX_GROUPS", 1 << (groups - 1).bit_length())],
        grid=(GROUP, rows, 1), threadgroup=(GROUP, 1, 1),
        output_shapes=[(rows, k), (rows,)], output_dtypes=[mx.int8, mx.float32])


def gemm(xq, xs, weight, scale, dtype):
    rows, n = xq.shape[0], weight.shape[0]
    bm = 128 if rows < 8192 else 64
    bn, bk = 8192 // bm, 64
    (y,) = _gemm_kernel()(
        inputs=[xq, weight, xs, scale], template=[("BM", bm), ("BN", bn), ("BK", bk), ("T", dtype)],
        grid=((n + bn - 1) // bn * 128, (rows + bm - 1) // bm, 1), threadgroup=(128, 1, 1),
        output_shapes=[(rows, n)], output_dtypes=[dtype])
    return y


def quantize_weight(w):
    """(out, in) float weight -> int8 rows of W H256 and float32 per-row scales (float32 math)."""
    out, k = w.shape
    rotated = (w.astype(mx.float32).reshape(-1, GROUP) @ H256).reshape(out, k)
    scale = mx.maximum(mx.max(mx.abs(rotated), axis=-1) / 127, 1e-20)
    return mx.clip(mx.round(rotated / scale[:, None]), -127, 127).astype(mx.int8), scale


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

_DTYPES = {"BF16": mx.bfloat16, "F16": mx.float16, "F32": mx.float32, "I8": mx.int8, "U8": mx.uint8,
           "I32": mx.int32, "U32": mx.uint32, "I64": mx.int64, "BOOL": mx.bool_}
_NAMES = {v: k for k, v in _DTYPES.items()}


def _bytes(a):
    a = mx.contiguous(a)
    raw = a.view(mx.uint8) if a.dtype != mx.bool_ else a.astype(mx.uint8)
    return memoryview(np.asarray(raw.reshape(-1)))


def save(path, tensors, metadata=None):
    """Standard safetensors (contiguous, header padded to a page) with the tensors whose size is a whole number of
    pages first, then the largest other tensor (e.g. an embedding table), all starting on a page; the small rest follows
(and also covers the last page of that tensor, so it can be mapped whole)."""
    sizes = {k: v.nbytes for k, v in tensors.items()}
    paged = [k for k in tensors if sizes[k] % PAGE == 0]
    rest = sorted((k for k in tensors if sizes[k] % PAGE), key=lambda k: -sizes[k])
    order = paged + rest
    header, offset = {}, 0
    for k in order:
        v = tensors[k]
        header[k] = {"dtype": _NAMES[v.dtype], "shape": list(v.shape), "data_offsets": [offset, offset + sizes[k]]}
        offset += sizes[k]
    if metadata:
        header = {"__metadata__": metadata, **header}
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * ((-(8 + len(blob))) % PAGE)
    tmp = Path(str(path) + ".partial")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(blob)) + blob)
        for k in order:
            f.write(_bytes(tensors[k]))
    os.replace(tmp, path)


def load(path, copy=False):
    """Tensors of a safetensors file; each one that starts on a page (and spans at least one) is a zero-copy view of a
    read-only mapping, the rest are copied. copy=True: plain mx.load."""
    if copy:
        return mx.load(str(path))
    with open(path, "rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(length))
        size = os.fstat(f.fileno()).st_size
        mapping = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
    start = 8 + length
    out = {}
    for key, spec in header.items():
        if key == "__metadata__":
            continue
        a, b = spec["data_offsets"]
        dtype, offset, nbytes = _DTYPES[spec["dtype"]], start + a, b - a
        padded = -(-nbytes // PAGE) * PAGE
        if offset % PAGE or nbytes < PAGE or offset + padded > size:
            raw = mx.array(np.frombuffer(mapping, np.uint8, nbytes, offset))
        else:   # Metal wraps host memory without a copy only for whole pages: map them and slice the tail off
            raw = mx.from_dlpack(np.frombuffer(mapping, np.uint8, padded, offset))[:nbytes]
        out[key] = (raw.view(dtype) if dtype != mx.uint8 else raw).reshape(spec["shape"])
    return out
