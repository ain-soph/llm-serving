"""EmbeddingGemma 2 text encoder on MLX, read straight from the ggml-org BF16 GGUF (no second copy of the weights).

Same graph as llama.cpp's gemma-embedding2 (src/models/gemma-embedding2.cpp, which matched google/embeddinggemma-2 to
cosine 1.00000): bidirectional attention, symmetric sliding window (|i - j| <= 512) on 20 layers and full attention on
4 (head_dim 512, one KV head), q/k RMS norm with weights and v without, NEOX RoPE (1e4 sliding / 1e6 full), GeGLU (tanh)
feed-forward, per-layer inputs projected from the scaled token embedding, a per-layer output scale, mean pooling over
all tokens (BOS and EOS included), the 512 -> 768 projection and L2 normalization.

Weights stay BF16; activations are float32 by default (on Apple Silicon Metal GPU, custom M5 TensorOps float32 x bfloat16
-> float32 kernels keep activations FP32 without per-use BF16->FP32 weight casts). dtype=mx.bfloat16 keeps the
activations in BF16 (weights used as stored; norms, RoPE and the residual scale in BF16, pooling and the output
projection in float32); w8a8=True additionally runs every projection on int8 convrot weights with per-row int8 activations
(common/mlx_w8a8.py), or only the classes named in w8a8 (W8A8_CLASSES). Both are measured in reports/embedding.md (2026-11-17).
Precision per op group (2026-11-18): dtype = the residual stream and elementwise ops (norms, RoPE, GeGLU, residual
scale); mm_dtype = the projections' inputs and weights (float32 accumulation in every case); attn_dtype = q/k/v into
attention; pooling and the output projection always float32. mixed=True is dtype float32 with BF16 products and
attention (what llama.cpp's BF16 GGUF does for the weights, with float32 activations). q8=True stores the projections
of W8A8_CLASSES (or the classes named in q8) as symmetric 8-bit weights (common/mlx_q8.py) for quantized_matmul with
mm_dtype activations (W8A16 for 16-bit, W8 x FP32 for float32); w8a8 as above (with dtype float32 under mixed).
Several inputs are packed into one token stream with positions restarting per input and attention kept inside each
input, so a batch gives the same vectors as one input at a time up to rounding. For max length <= 2048 and padding
overhead <= 1.25x, inputs run in batched SDPA (zero-copy when equal length, gathered/masked when ragged); longer
batches run in query chunks with shared precomputed masks.
"""
import json
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np

CHUNK = 1024
_DEFAULT_KERNEL_OPTS = ("gqa", "padded_batch", "native_mm", "native_geglu", "native_qkv_small")
_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_DTYPES = {0: (np.float32, 4), 1: (np.float16, 2), 30: (np.uint16, 2)}     # F32, F16, BF16 (raw bits)

_METAL_HEADER = """
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

METAL_FUNC float tanh_gelu(float x) {
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + metal::precise::tanh(0.7978845608028654f * (x + 0.044715f * x3)));
}
"""

_BODY_MM = """
const int M = x_shape[0], K = x_shape[1], N = w_shape[0];
const int m_lo = int(threadgroup_position_in_grid.y) * BM;
const int n_lo = int(threadgroup_position_in_grid.x) * BN;
const int m0 = min(m_lo, M - BM);
const int n0 = min(n_lo, N - BN);
constexpr auto desc = matmul2d_descriptor(BM, BN, BK, false, true, false,
                                          matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<desc, execution_simdgroups<4>> op;
tensor<device float, dextents<int, 2>, tensor_inline> a((device float*)x, dextents<int, 2>(K, M));
tensor<device bfloat, dextents<int, 2>, tensor_inline> b((device bfloat*)w, dextents<int, 2>(K, N));
auto a0 = a.template slice<BK, BM>(0, m0);
auto b0 = b.template slice<BK, BN>(0, n0);
auto acc = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
for (unsigned short i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
for (int k = 0; k < K; k += BK) {
    auto lhs = a.template slice<BK, BM>(k, m0);
    auto rhs = b.template slice<BK, BN>(k, n0);
    op.run(lhs, rhs, acc);
}
if (SCALE_OUT != 0) {
    const float sc = scale_arr[0];
    for (unsigned short i = 0; i < acc.get_capacity(); ++i) acc[i] *= sc;
}
if (m0 == m_lo && n0 == n_lo) {
    tensor<device float, dextents<int, 2>, tensor_inline> c((device float*)y, dextents<int, 2>(N, M));
    acc.store(c.template slice<BN, BM>(n0, m0));
} else {
    for (unsigned short i = 0; i < acc.get_capacity(); ++i) {
        auto ids = acc.get_multidimensional_index(i);
        const int n = n0 + ids[0], m = m0 + ids[1];
        if (m >= m_lo && m < M && n >= n_lo && n < N) {
            y[(size_t)m * N + n] = acc[i];
        }
    }
}
"""

_BODY_GEGLU = """
const int M = x_shape[0], K = x_shape[1], N = wg_shape[0];
const int m_lo = int(threadgroup_position_in_grid.y) * BM;
const int n_lo = int(threadgroup_position_in_grid.x) * BN;
const int m0 = min(m_lo, M - BM);
const int n0 = min(n_lo, N - BN);
constexpr auto desc = matmul2d_descriptor(BM, BN, BK, false, true, false,
                                          matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<desc, execution_simdgroups<4>> op;
tensor<device bfloat, dextents<int, 2>, tensor_inline> bg((device bfloat*)wg, dextents<int, 2>(K, N));
tensor<device bfloat, dextents<int, 2>, tensor_inline> bu((device bfloat*)wu, dextents<int, 2>(K, N));
tensor<device float, dextents<int, 2>, tensor_inline> a((device float*)x, dextents<int, 2>(K, M));
auto a0 = a.template slice<BK, BM>(0, m0);
auto b0 = bg.template slice<BK, BN>(0, n0);
auto acc_g = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
auto acc_u = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
for (unsigned short i = 0; i < acc_g.get_capacity(); ++i) { acc_g[i] = 0.0f; acc_u[i] = 0.0f; }
for (int k = 0; k < K; k += BK) {
    auto lhs = a.template slice<BK, BM>(k, m0);
    auto rhsg = bg.template slice<BK, BN>(k, n0);
    auto rhsu = bu.template slice<BK, BN>(k, n0);
    op.run(lhs, rhsg, acc_g);
    op.run(lhs, rhsu, acc_u);
}
for (unsigned short i = 0; i < acc_g.get_capacity(); ++i) {
    acc_g[i] = tanh_gelu(acc_g[i]) * acc_u[i];
}
if (m0 == m_lo && n0 == n_lo) {
    tensor<device float, dextents<int, 2>, tensor_inline> c((device float*)y, dextents<int, 2>(N, M));
    acc_g.store(c.template slice<BN, BM>(n0, m0));
} else {
    for (unsigned short i = 0; i < acc_g.get_capacity(); ++i) {
        auto ids = acc_g.get_multidimensional_index(i);
        const int n = n0 + ids[0], m = m0 + ids[1];
        if (m >= m_lo && m < M && n >= n_lo && n < N) {
            y[(size_t)m * N + n] = acc_g[i];
        }
    }
}
"""

_BODY_PLE_GEGLU = """
const int M = x_shape[0], K = x_shape[1], N = w_shape[0];
const int m_lo = int(threadgroup_position_in_grid.y) * BM;
const int n_lo = int(threadgroup_position_in_grid.x) * BN;
const int m0 = min(m_lo, M - BM);
const int n0 = min(n_lo, N - BN);
constexpr auto desc = matmul2d_descriptor(BM, BN, BK, false, true, false,
                                          matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<desc, execution_simdgroups<4>> op;
tensor<device float, dextents<int, 2>, tensor_inline> a((device float*)x, dextents<int, 2>(K, M));
tensor<device bfloat, dextents<int, 2>, tensor_inline> b((device bfloat*)w, dextents<int, 2>(K, N));
auto a0 = a.template slice<BK, BM>(0, m0);
auto b0 = b.template slice<BK, BN>(0, n0);
auto acc = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
for (unsigned short i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
for (int k = 0; k < K; k += BK) {
    auto lhs = a.template slice<BK, BM>(k, m0);
    auto rhs = b.template slice<BK, BN>(k, n0);
    op.run(lhs, rhs, acc);
}
for (unsigned short i = 0; i < acc.get_capacity(); ++i) {
    auto ids = acc.get_multidimensional_index(i);
    const int n = n0 + ids[0], m = m0 + ids[1];
    if (m >= m_lo && m < M && n >= n_lo && n < N) {
        y[(size_t)m * N + n] = tanh_gelu(acc[i]) * ple[(size_t)m * N + n];
    }
}
"""

_BODY_QKV = """
const int M = x_shape[0], K = x_shape[1];
const int Nq = wq_shape[0], Nk = wk_shape[0], Nv = wv_shape[0];
const int tq = Nq / BN, tk = Nk / BN;
const int tx = int(threadgroup_position_in_grid.x);
const int m_lo = int(threadgroup_position_in_grid.y) * BM;
const int m0 = min(m_lo, M - BM);

device bfloat* wp = (device bfloat*)wq;
device float* yp = yq;
int N = Nq;
int n0 = tx * BN;
if (tx >= tq + tk) {
    wp = (device bfloat*)wv; yp = yv; N = Nv; n0 = (tx - tq - tk) * BN;
} else if (tx >= tq) {
    wp = (device bfloat*)wk; yp = yk; N = Nk; n0 = (tx - tq) * BN;
}

constexpr auto desc = matmul2d_descriptor(BM, BN, BK, false, true, false,
                                          matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<desc, execution_simdgroups<4>> op;
tensor<device bfloat, dextents<int, 2>, tensor_inline> b(wp, dextents<int, 2>(K, N));
tensor<device float, dextents<int, 2>, tensor_inline> a((device float*)x, dextents<int, 2>(K, M));
auto a0 = a.template slice<BK, BM>(0, m0);
auto b0 = b.template slice<BK, BN>(0, n0);
auto acc = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
for (unsigned short i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
for (int k = 0; k < K; k += BK) {
    auto lhs = a.template slice<BK, BM>(k, m0);
    auto rhs = b.template slice<BK, BN>(k, n0);
    op.run(lhs, rhs, acc);
}
if (m0 == m_lo) {
    tensor<device float, dextents<int, 2>, tensor_inline> c(yp, dextents<int, 2>(N, M));
    acc.store(c.template slice<BN, BM>(n0, m0));
} else {
    for (unsigned short i = 0; i < acc.get_capacity(); ++i) {
        auto ids = acc.get_multidimensional_index(i);
        const int n = n0 + ids[0], m = m0 + ids[1];
        if (m >= m_lo && m < M && n < N) yp[(size_t)m * N + n] = acc[i];
    }
}
"""

_HAS_METAL = mx.metal.is_available()
_KERNELS = {}
_DUMMY_SCALE = None


def _on_gpu():
    """True when MLX default execution target is Metal GPU."""
    return _HAS_METAL and mx.default_device().type == mx.DeviceType.gpu


def _get_kernels():
    """Lazily compile Metal TensorOps kernel objects on first GPU call."""
    if not _KERNELS:
        specs = (
            ("mm", "gemma_f32_bf16_mm", ["x", "w", "scale_arr"], ["y"], _BODY_MM),
            ("geglu", "gemma_f32_bf16_geglu", ["x", "wg", "wu"], ["y"], _BODY_GEGLU),
            ("ple_geglu", "gemma_f32_bf16_ple_geglu", ["x", "w", "ple"], ["y"], _BODY_PLE_GEGLU),
            ("qkv", "gemma_f32_bf16_qkv", ["x", "wq", "wk", "wv"], ["yq", "yk", "yv"], _BODY_QKV),
        )
        for key, name, ins, outs, src in specs:
            _KERNELS[key] = mx.fast.metal_kernel(
                name=name, input_names=ins, output_names=outs, header=_METAL_HEADER, source=src
            )
    return _KERNELS


def _dummy_scale():
    """Cached unit scale array for unscaled gemma_f32_bf16_mm calls."""
    global _DUMMY_SCALE
    if _DUMMY_SCALE is None:
        _DUMMY_SCALE = mx.array([1.0], dtype=mx.float32)
    return _DUMMY_SCALE


def _pad_rows(x, bm):
    """Pad x along row axis to bm so Metal TensorOps slice<BK, BM> stays in bounds."""
    return mx.pad(x, [(0, bm - x.shape[0]), (0, 0)]) if x.shape[0] < bm else x


def _prepare_mma_grid(x, n, extra=None):
    """Select (bm, bn, bk), pad M < bm if needed, and compute Metal grid dimensions."""
    m, k = x.shape
    bm = 16 if m < 64 else 64
    bn, bk = 64, 32
    if n < bn or k % bk != 0:
        return None
    pad = m < bm
    x_in = _pad_rows(x, bm) if pad else x
    extra_in = (_pad_rows(extra, bm) if pad else extra) if extra is not None else None
    m_eff = x_in.shape[0]
    grid_x = ((n + bn - 1) // bn) * 128
    grid_y = (m_eff + bm - 1) // bm
    return m, m_eff, pad, bm, bn, bk, grid_x, grid_y, x_in, extra_in


def _native_project(x, w, scale_arr=None):
    """FP32 x [M, K] @ BF16 w [N, K].T -> FP32 [M, N] via TensorOps with optional scalar multiplier."""
    prep = _prepare_mma_grid(x, w.shape[0])
    if prep is None or x.dtype != mx.float32 or w.dtype != mx.bfloat16:
        y = x @ w.astype(x.dtype).T
        return y * scale_arr[0] if scale_arr is not None else y
    m, m_eff, pad, bm, bn, bk, grid_x, grid_y, x_in, _ = prep
    scale_flag = 1 if scale_arr is not None else 0
    s_in = scale_arr if scale_arr is not None else _dummy_scale()
    y = _get_kernels()["mm"](
        inputs=[x_in, w, s_in],
        template=[("BM", bm), ("BN", bn), ("BK", bk), ("SCALE_OUT", scale_flag)],
        grid=(grid_x, grid_y, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(m_eff, w.shape[0])],
        output_dtypes=[mx.float32],
    )[0]
    return y[:m] if pad else y


def _native_geglu(x, wg, wu):
    """Fused tanh_gelu(x @ wg.T) * (x @ wu.T) in registers for FP32 x and BF16 weights."""
    prep = _prepare_mma_grid(x, wg.shape[0])
    if prep is None or x.dtype != mx.float32 or wg.dtype != mx.bfloat16 or wu.dtype != mx.bfloat16:
        return _gelu_any(x @ wg.astype(x.dtype).T) * (x @ wu.astype(x.dtype).T)
    m, m_eff, pad, bm, bn, bk, grid_x, grid_y, x_in, _ = prep
    y = _get_kernels()["geglu"](
        inputs=[x_in, wg, wu],
        template=[("BM", bm), ("BN", bn), ("BK", bk)],
        grid=(grid_x, grid_y, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(m_eff, wg.shape[0])],
        output_dtypes=[mx.float32],
    )[0]
    return y[:m] if pad else y


def _native_ple_geglu(x, w, ple):
    """Fused tanh_gelu(x @ w.T) * ple for FP32 x/ple and BF16 w."""
    prep = _prepare_mma_grid(x, w.shape[0], extra=ple)
    if prep is None or x.dtype != mx.float32 or w.dtype != mx.bfloat16 or ple.dtype != mx.float32:
        return _gelu_any(x @ w.astype(x.dtype).T) * ple
    m, m_eff, pad, bm, bn, bk, grid_x, grid_y, x_in, ple_in = prep
    y = _get_kernels()["ple_geglu"](
        inputs=[x_in, w, ple_in],
        template=[("BM", bm), ("BN", bn), ("BK", bk)],
        grid=(grid_x, grid_y, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(m_eff, w.shape[0])],
        output_dtypes=[mx.float32],
    )[0]
    return y[:m] if pad else y


def _native_qkv(x, wq, wk, wv):
    """Fused QKV projection writing 3 contiguous FP32 buffers in one launch (optimal for small M)."""
    m, k = x.shape
    nq, nk, nv = wq.shape[0], wk.shape[0], wv.shape[0]
    bm = 16 if m < 64 else 64
    bn, bk = 64, 32
    if (x.dtype != mx.float32 or wq.dtype != mx.bfloat16 or wk.dtype != mx.bfloat16
            or wv.dtype != mx.bfloat16 or k % bk != 0 or nq % bn != 0 or nk % bn != 0 or nv % bn != 0):
        return x @ wq.astype(x.dtype).T, x @ wk.astype(x.dtype).T, x @ wv.astype(x.dtype).T
    pad = m < bm
    x_in = _pad_rows(x, bm) if pad else x
    m_eff = x_in.shape[0]
    tx = (nq + nk + nv) // bn
    grid_y = (m_eff + bm - 1) // bm
    yq, yk, yv = _get_kernels()["qkv"](
        inputs=[x_in, wq, wk, wv],
        template=[("BM", bm), ("BN", bn), ("BK", bk)],
        grid=(tx * 128, grid_y, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(m_eff, nq), (m_eff, nk), (m_eff, nv)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    return (yq[:m], yk[:m], yv[:m]) if pad else (yq, yk, yv)


def read_gguf(path):
    """GGUF v3 -> (metadata, {name: mx.array}); BF16 tensors keep their exact bits."""
    data = np.memmap(path, dtype=np.uint8, mode="r")
    buf = memoryview(data)
    pos = 0
    if True:

        def take(fmt):
            nonlocal pos
            v = struct.unpack_from(fmt, buf, pos)
            pos += struct.calcsize(fmt)
            return v[0]

        def string():
            nonlocal pos
            n = take("<Q")
            s = bytes(buf[pos:pos + n]).decode("utf-8", "replace")
            pos += n
            return s

        def value(t):
            if t == 8:
                return string()
            if t == 9:
                et, n = take("<I"), take("<Q")
                return [value(et) for _ in range(n)]
            return take(_SCALARS[t])

        if bytes(buf[:4]) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        pos = 4
        take("<I")
        n_tensors, n_kv = take("<Q"), take("<Q")
        meta = {}
        for _ in range(n_kv):
            key = string()
            t = take("<I")
            v = value(t)
            if not key.startswith("tokenizer.ggml.") or key in ("tokenizer.ggml.model",):
                meta[key] = v
        infos = []
        for _ in range(n_tensors):
            name = string()
            dims = [take("<Q") for _ in range(take("<I"))]
            infos.append((name, dims, take("<I"), take("<Q")))
        align = meta.get("general.alignment", 32)
        start = (pos + align - 1) // align * align
    tensors = {}
    for name, dims, t, off in infos:
        dt, size = _DTYPES[t]
        n = int(np.prod(dims))
        a = np.frombuffer(data, dtype=dt, count=n, offset=start + off).reshape(dims[::-1])
        a = mx.array(a)
        tensors[name] = a.view(mx.bfloat16) if t == 30 else a
    return meta, tensors


# projection classes for W8A8 (one class can be turned on at a time to measure its own speedup and error)
W8A8_CLASSES = {"qkv": ("attn_q.weight", "attn_k.weight", "attn_v.weight"), "attn_out": ("attn_output.weight",),
                "gate_up": ("ffn_gate.weight", "ffn_up.weight"), "down": ("ffn_down.weight",),
                "ple": ("inp_gate.weight", "proj.weight")}


class Gemma2Embedder:
    def __init__(self, gguf_path, tokenizer_path, dtype=mx.float32, w8a8=False, mixed=False, q8=False,
                 mm_dtype=None, attn_dtype=None, embed8=False, kernel_opts=None):
        from tokenizers import Tokenizer
        meta, w = read_gguf(gguf_path)
        a = meta["general.architecture"]
        if a != "gemma-embedding2":
            raise ValueError(f"{gguf_path}: architecture {a}, expected gemma-embedding2")
        if mixed:
            dtype, mm_dtype = mx.float32, mm_dtype or mx.bfloat16
        elif w8a8 and mm_dtype is None:
            dtype = mx.bfloat16
        self.dtype = dtype
        self.mm_dtype = mm_dtype or dtype
        self.attn_dtype = attn_dtype or self.mm_dtype
        self.mixed = self.mm_dtype != dtype or self.attn_dtype != dtype
        if self.dtype != mx.float32:
            w = {k: v.astype(self.dtype) if v.dtype == mx.float32 else v for k, v in w.items()}
        if self.mixed:        # products in mm_dtype: keep the projection weights in it (cast once, not per use)
            w = {k: v.astype(self.mm_dtype) if v.ndim == 2 and (k.startswith("blk.") or k == "per_layer_model_proj.weight")
                 else v for k, v in w.items()}
        self.w16 = {}
        if q8:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
            import mlx_q8
            classes = W8A8_CLASSES if q8 is True else {c: W8A8_CLASSES[c] for c in q8}
            suffixes = tuple(x for v in classes.values() for x in v)
            for k in list(w):
                if k.startswith("blk.") and k.endswith(suffixes):
                    self.w16[k] = mlx_q8.quantize_weight(w.pop(k), self.mm_dtype)
        self.embed8 = None
        if embed8:            # token table as symmetric 8-bit rows (gathered, then dequantized): half of the file
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
            import mlx_q8
            self.embed8 = mlx_q8.quantize_weight(w.pop("token_embd.weight"), mx.float32 if self.dtype == mx.float32
                                                 else self.dtype)
        self.q8 = {}
        if w8a8:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
            import mlx_w8a8
            self._w8 = mlx_w8a8
            classes = W8A8_CLASSES if w8a8 is True else {c: W8A8_CLASSES[c] for c in w8a8}
            suffixes = tuple(x for v in classes.values() for x in v)
            for k in list(w):
                if k.startswith("blk.") and k.endswith(suffixes):
                    self.q8[k] = mlx_w8a8.quantize_weight(w.pop(k))
        self._xq = (None, None, None)

        opts = set(_DEFAULT_KERNEL_OPTS if kernel_opts is None else kernel_opts)
        allowed = {"short_batch", "padded_batch", "gqa", "native_mm", "native_geglu", "native_qkv", "native_qkv_small"}
        unknown = opts - allowed
        if unknown:
            raise ValueError(f"unknown kernel options: {sorted(unknown)}")
        self.native_gqa = "gqa" in opts
        self.padded_batch = "padded_batch" in opts
        self.short_batch = "short_batch" in opts or self.padded_batch
        self.native_mm = "native_mm" in opts
        self.native_geglu = "native_geglu" in opts
        self.native_qkv = "native_qkv" in opts
        self.native_qkv_small = "native_qkv_small" in opts

        self.n_layer = meta[f"{a}.block_count"]
        self.n_embd = meta[f"{a}.embedding_length"]
        self.n_head = meta[f"{a}.attention.head_count"]
        self.n_head_kv = meta[f"{a}.attention.head_count_kv"]
        self.swa = meta[f"{a}.attention.sliding_window_pattern"]
        self.half_window = meta[f"{a}.attention.sliding_window"] // 2
        self.eps = meta[f"{a}.attention.layer_norm_rms_epsilon"]
        self.base = (meta[f"{a}.rope.freq_base_swa"], meta[f"{a}.rope.freq_base"])
        self.n_per_layer = meta[f"{a}.embedding_length_per_layer_input"]
        self.dim = meta[f"{a}.embedding_length_out"]
        self.max_tokens = meta[f"{a}.context_length"]

        self._ple_scale = mx.array([1.0 / np.sqrt(self.n_embd)], dtype=mx.float32)
        ple_w = w["per_layer_model_proj.weight"]
        if self.native_mm and not self.mixed and self.dtype == mx.float32 and ple_w.dtype == mx.bfloat16:
            n = self.n_per_layer
            self._ple_weights = [mx.contiguous(ple_w[il * n:(il + 1) * n]) for il in range(self.n_layer)]
        else:
            self._ple_weights = None

        self.w = w
        mx.eval(list(w.values()), list(self.q8.values()), list(self.w16.values()), self.embed8,
                self._ple_scale, *(self._ple_weights or []))
        self.tok = Tokenizer.from_file(str(tokenizer_path))
        self.tok.no_padding()
        self.tok.no_truncation()
        self._tables = {}
        # Elementwise chains compiled into single kernels (2026-11-18, MLX fork fusion survey: FP32 query 33.8 -> 7.1 ms,
        # 64 x 400 tokens 7.9K -> 26.5K tok/s, same vectors): GeGLU, RMS norm + residual, the residual scale, RoPE.
        self._geglu = mx.compile(lambda a, b: _gelu_any(a) * b)
        self._norm_add = mx.compile(lambda x, w, r, eps: mx.fast.rms_norm(x, w, eps) + r)
        self._add_scale = mx.compile(lambda c, n, s: (c + n) * s)
        self._rope_c = mx.compile(_rope)
        self._single = False
        self._fwd_rope = {}
        self._attn_plans = {}

    def tokenize(self, texts):
        return [e.ids for e in self.tok.encode_batch(texts, add_special_tokens=True)]

    def _mm(self, x, name):
        if name in self.q8:
            wq, scale = self.q8[name]
            if self._xq[0] is not x:      # q/k/v and gate/up read the same input: quantize it once
                self._xq = (x, *self._w8.rotate_quantize(x.reshape(-1, x.shape[-1])))
            _, xq, xs = self._xq
            return self._w8.gemm(xq, xs, wq, scale, x.dtype).reshape(*x.shape[:-1], wq.shape[0])
        wt = self.w[name] if name in self.w else None
        if (self.native_mm and wt is not None and _on_gpu()
                and x.dtype == mx.float32 and self.mm_dtype == mx.float32 and wt.dtype == mx.bfloat16):
            return _native_project(x, wt)
        xm = x.astype(self.mm_dtype) if x.dtype != self.mm_dtype else x
        if name in self.w16:
            packed, scales, biases = self.w16[name]
            y = mx.quantized_matmul(xm, packed, scales, biases, transpose=True, group_size=32, bits=8)
        else:
            y = xm @ wt.astype(self.mm_dtype).T
        return y.astype(self.dtype) if y.dtype != self.dtype else y

    def _qkv_proj(self, h, il, t_len):
        """Project Q, K, V using fused Metal QKV kernel when eligible, else three separate projections."""
        p = f"blk.{il}."
        use_qkv = self.native_qkv or (self.native_qkv_small and t_len < 64)
        if (use_qkv and _on_gpu() and h.dtype == mx.float32 and self.mm_dtype == mx.float32
                and (p + "attn_q.weight") in self.w):
            return _native_qkv(h, self.w[p + "attn_q.weight"], self.w[p + "attn_k.weight"], self.w[p + "attn_v.weight"])
        return self._mm(h, p + "attn_q.weight"), self._mm(h, p + "attn_k.weight"), self._mm(h, p + "attn_v.weight")

    def _ffn_geglu(self, f, il):
        """Fused dual-GEMM GeGLU when FP32 native Metal kernels are active, else two projections + compiled GeGLU."""
        p = f"blk.{il}."
        wg_name, wu_name = p + "ffn_gate.weight", p + "ffn_up.weight"
        if (self.native_geglu and _on_gpu() and f.dtype == mx.float32 and self.mm_dtype == mx.float32
                and wg_name in self.w and wu_name in self.w
                and self.w[wg_name].dtype == mx.bfloat16 and self.w[wu_name].dtype == mx.bfloat16):
            return _native_geglu(f, self.w[wg_name], self.w[wu_name])
        return self._geglu(self._mm(f, wg_name), self._mm(f, wu_name))

    def _ple_geglu(self, c, ple, il):
        """Fused PLE GeGLU when FP32 native Metal kernels are active, else projection + compiled GeGLU."""
        w_name = f"blk.{il}.inp_gate.weight"
        if (self.native_geglu and _on_gpu() and c.dtype == mx.float32 and self.mm_dtype == mx.float32
                and w_name in self.w and self.w[w_name].dtype == mx.bfloat16 and ple.dtype == mx.float32):
            return _native_ple_geglu(c, self.w[w_name], ple)
        return self._geglu(self._mm(c, w_name), ple)

    def _ple_mm(self, x0, il):
        """Layer il's per-layer input projection with fused 1/sqrt(n_embd) epilogue when using native Metal."""
        if (self._ple_weights is not None and _on_gpu()
                and x0.dtype == mx.float32 and self.mm_dtype == mx.float32):
            return _native_project(x0, self._ple_weights[il], self._ple_scale)
        n = self.n_per_layer
        w = self.w["per_layer_model_proj.weight"][il * n:(il + 1) * n]
        y = (x0.astype(self.mm_dtype) @ w.astype(self.mm_dtype).T).astype(self.dtype)
        return y * self._c(1 / np.sqrt(self.n_embd))

    def _norm(self, x, name=None):
        return mx.fast.rms_norm(x, self.w[name] if name else None, self.eps)

    def _build_chunk_plan(self, T, pos, seg, window):
        """Precompute chunk boundaries and masks once per forward pass for sliding vs full layers."""
        plan = []
        chunk = CHUNK if window else max(128, min(CHUNK, (1 << 23) // T // 128 * 128))
        for s in range(0, T, chunk):
            e = min(T, s + chunk)
            ks, ke = (max(0, s - window), min(T, e + window)) if window else (0, T)
            if self._single and not window and self.padded_batch:
                mask = None
            else:
                m = seg[s:e, None] == seg[None, ks:ke]
                if window:
                    d = pos[s:e, None] - pos[None, ks:ke]
                    m = m & (d <= window) & (d >= -window)
                mask = m[None, None]
            plan.append((s, e, ks, ke, mask))
        return plan

    def _attention(self, q, k, v, pos, seg, window, segment_length=0, padding=None):
        """q (H, T, D), k/v (Hkv, T, D); returns (T, H*D) directly to avoid redundant transpose copies."""
        T, H, D = q.shape[1], q.shape[0], q.shape[2]
        rep = H // k.shape[0]
        if rep > 1 and not self.native_gqa:
            k, v = mx.repeat(k, rep, axis=0), mx.repeat(v, rep, axis=0)
        if segment_length:
            if padding is not None:
                gather, restore, valid_mask, sliding_mask = padding
                qb = q[:, gather, :].transpose(1, 0, 2, 3)
                kb = k[:, gather, :].transpose(1, 0, 2, 3)
                vb = v[:, gather, :].transpose(1, 0, 2, 3)
                mask = sliding_mask if (window and segment_length > window + 1) else valid_mask
                values = mx.fast.scaled_dot_product_attention(qb, kb, vb, scale=1.0, mask=mask)
                return values.transpose(0, 2, 1, 3).reshape(-1, H, D)[restore].reshape(T, H * D)
            qb = q.reshape(H, -1, segment_length, D).transpose(1, 0, 2, 3)
            kb = k.reshape(k.shape[0], -1, segment_length, D).transpose(1, 0, 2, 3)
            vb = v.reshape(v.shape[0], -1, segment_length, D).transpose(1, 0, 2, 3)
            mask = self._attn_plans.get("eq_sliding_mask") if (window and segment_length > window + 1) else None
            values = mx.fast.scaled_dot_product_attention(qb, kb, vb, scale=1.0, mask=mask)
            return values.transpose(0, 2, 1, 3).reshape(T, H * D)
        key = "sliding" if window else "full"
        plan = self._attn_plans.get(key) if self.padded_batch else self._build_chunk_plan(T, pos, seg, window)
        if plan is None:
            plan = self._build_chunk_plan(T, pos, seg, window)
            self._attn_plans[key] = plan
        out = []
        for s, e, ks, ke, mask in plan:
            v_chunk = mx.fast.scaled_dot_product_attention(
                q[None, :, s:e], k[None, :, ks:ke], v[None, :, ks:ke], scale=1.0, mask=mask)[0]
            if self.padded_batch:
                out.append(v_chunk.transpose(1, 0, 2))
            else:
                out.append(v_chunk)
        if self.padded_batch:
            merged = mx.concatenate(out, axis=0) if len(out) > 1 else out[0]
            return merged.reshape(T, H * D)
        merged = mx.concatenate(out, axis=1) if len(out) > 1 else out[0]
        return merged.transpose(1, 0, 2).reshape(T, H * D)

    def _layer(self, il, x, ple, pos, seg, segment_length=0, padding=None):
        p = f"blk.{il}."
        T = x.shape[0]
        H, Hkv = self.n_head, self.n_head_kv[il]
        sliding = self.swa[il]
        h = self._norm(x, p + "attn_norm.weight")
        q, k, v = self._qkv_proj(h, il, T)
        q, k, v = q.reshape(T, H, -1), k.reshape(T, Hkv, -1), v.reshape(T, Hkv, -1)
        D = q.shape[-1]
        q = self._norm(q, p + "attn_q_norm.weight").transpose(1, 0, 2)
        k = self._norm(k, p + "attn_k_norm.weight").transpose(1, 0, 2)
        v = self._norm(v).transpose(1, 0, 2)
        base = self.base[0] if sliding else self.base[1]
        if self._single:   # one input: positions 0..T-1
            q = mx.fast.rope(q, D, traditional=False, base=base, scale=1.0, offset=0)
            k = mx.fast.rope(k, D, traditional=False, base=base, scale=1.0, offset=0)
        else:
            if self.native_gqa:
                rkey = (D, base)
                cs = self._fwd_rope.get(rkey)
                if cs is None:
                    cs = self._rope_tables(pos, D, base)
                    self._fwd_rope[rkey] = cs
                cos, sin = cs
            else:
                cos, sin = self._rope_tables(pos, D, base)
            q, k = self._rope_c(q, cos, sin), self._rope_c(k, cos, sin)
        if self.attn_dtype != self.dtype:
            q, k, v = (t.astype(self.attn_dtype) for t in (q, k, v))
        o = self._attention(q, k, v, pos, seg, self.half_window if sliding else 0, segment_length, padding)
        if o.dtype != self.dtype:
            o = o.astype(self.dtype)
        o = self._mm(o, p + "attn_output.weight")
        a = self._norm_add(o, self.w[p + "post_attention_norm.weight"], x, self.eps)
        f = self._norm(a, p + "ffn_norm.weight")
        f = self._mm(self._ffn_geglu(f, il), p + "ffn_down.weight")
        c = self._norm_add(f, self.w[p + "post_ffw_norm.weight"], a, self.eps)
        g = self._ple_geglu(c, ple, il)
        return self._add_scale(c, self._norm(self._mm(g, p + "proj.weight"), p + "post_norm.weight"),
                               self.w[p + "layer_output_scale.weight"])

    def _gelu(self, x):
        if self.dtype == mx.float32:
            return _gelu(x)
        c = self._c
        return c(0.5) * x * (c(1.0) + mx.tanh(c(np.sqrt(2 / np.pi)) * (x + c(0.044715) * x * x * x)))

    def _per_layer_input(self, x0, il):
        """Layer il's slice of the per-layer inputs: RMS norm of (x0 W_il^T) / sqrt(n_embd), as llama.cpp computes
        for all layers at once (the norm is per 512-vector, so one slice at a time is the same and needs 1/24 the memory)."""
        return self._norm(self._ple_mm(x0, il), "per_layer_proj_norm.weight")

    def _c(self, v):
        """A constant in the activation dtype (an np.float32 scalar would promote BF16 activations to float32)."""
        return np.float32(v) if self.dtype == mx.float32 else mx.array(v, dtype=self.dtype)

    def _rope_tables(self, pos, D, base):
        key = (D, base)
        if key not in self._tables:
            inv = base ** (-np.arange(0, D, 2, dtype=np.float64) / D)
            self._tables[key] = mx.array(inv.astype(np.float32))
        ang = pos.astype(mx.float32)[:, None] * self._tables[key][None]
        return mx.cos(ang).astype(self.dtype), mx.sin(ang).astype(self.dtype)

    def forward(self, ids_list):
        """Token id lists -> (len, dim) float32 L2-normalized embeddings, all inputs in one packed pass."""
        lens = [len(ids) for ids in ids_list]
        ids = mx.array(np.concatenate([np.asarray(i, dtype=np.int32) for i in ids_list]))
        pos = mx.array(np.concatenate([np.arange(n, dtype=np.int32) for n in lens]))
        seg = mx.array(np.repeat(np.arange(len(lens), dtype=np.int32), lens))
        if self.embed8 is not None:
            packed, scales, biases = self.embed8
            rows = mx.dequantize(packed[ids], scales[ids], biases[ids], group_size=32, bits=8)
        else:
            rows = self.w["token_embd.weight"][ids]
        x = rows.astype(self.dtype) * self._c(np.sqrt(self.n_embd))
        x0 = x
        self._single = len(lens) == 1
        self._fwd_rope = {}
        self._attn_plans = {}
        segment_length = 0
        padding = None
        max_len = max(lens)
        max_batch_len = 2048 if self.padded_batch else (self.half_window + 1)
        if self.short_batch and max_len <= max_batch_len:
            if min(lens) == max_len:
                segment_length = max_len
                if max_len > self.half_window + 1:
                    idx = mx.arange(max_len, dtype=mx.int32)
                    d = idx[:, None] - idx[None, :]
                    self._attn_plans["eq_sliding_mask"] = ((d <= self.half_window) & (d >= -self.half_window))[None, None]
            elif self.padded_batch and max_len * len(lens) <= 1.25 * sum(lens):
                lengths = np.asarray(lens, dtype=np.int32)[:, None]
                offsets = np.cumsum(lengths, dtype=np.int32)[:, None] - lengths
                positions = np.arange(max_len, dtype=np.int32)[None, :]
                valid = positions < lengths
                gather = offsets + np.minimum(positions, lengths - 1)
                restore = np.flatnonzero(valid).astype(np.int32)
                valid_mask = mx.array(valid)[:, None, None, :]
                sliding_mask = None
                if max_len > self.half_window + 1:
                    idx = mx.arange(max_len, dtype=mx.int32)
                    d = idx[:, None] - idx[None, :]
                    sw_ok = (d <= self.half_window) & (d >= -self.half_window)
                    sliding_mask = valid_mask & sw_ok[None, None, :, :]
                padding = mx.array(gather), mx.array(restore), valid_mask, sliding_mask
                segment_length = max_len
        for il in range(self.n_layer):   # one graph per forward: an eval per layer cost a submit / sync each
            x = self._layer(il, x, self._per_layer_input(x0, il), pos, seg, segment_length, padding)
        x = self._norm(x, "output_norm.weight").astype(mx.float32)
        if self._single and self.native_gqa:
            pooled = mx.mean(x, axis=0, keepdims=True)
        else:
            onehot = (seg[None, :] == mx.arange(len(lens))[:, None]).astype(mx.float32)
            pooled = (onehot @ x) / mx.array(lens, dtype=mx.float32)[:, None]
        if (self.native_mm and _on_gpu() and not self.mixed
                and pooled.dtype == mx.float32 and self.w["output.weight"].dtype == mx.bfloat16):
            e = _native_project(pooled, self.w["output.weight"])
        else:
            e = pooled @ self.w["output.weight"].astype(mx.float32).T
        e = e / mx.linalg.norm(e, axis=-1, keepdims=True)
        mx.eval(e)
        self._xq = (None, None, None)
        self._fwd_rope = {}
        self._attn_plans = {}
        return e


def _gelu_any(x):
    """tanh GELU in x's dtype (Python-float constants: same rounding as _gelu in FP32 and as Gemma2Embedder._gelu
    in BF16 / FP16, and no NumPy scalars inside mx.compile)."""
    return 0.5 * x * (1.0 + mx.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


def _gelu(x):
    return 0.5 * x * (1 + mx.tanh(np.float32(np.sqrt(2 / np.pi)) * (x + np.float32(0.044715) * x * x * x)))


def _rope(x, cos, sin):
    """NEOX rotation of the two halves of the last axis; x (H, T, D), cos/sin (T, D/2)."""
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
