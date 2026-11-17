"""EmbeddingGemma 2 text encoder on MLX, read straight from the ggml-org BF16 GGUF (no second copy of the weights).

Same graph as llama.cpp's gemma-embedding2 (src/models/gemma-embedding2.cpp, which matched google/embeddinggemma-2 to
cosine 1.00000): bidirectional attention, symmetric sliding window (|i - j| <= 512) on 20 layers and full attention on
4 (head_dim 512, one KV head), q/k RMS norm with weights and v without, NEOX RoPE (1e4 sliding / 1e6 full), GeGLU (tanh)
feed-forward, per-layer inputs projected from the scaled token embedding, a per-layer output scale, mean pooling over
all tokens (BOS and EOS included), the 512 -> 768 projection and L2 normalization.

Weights stay BF16; activations are float32 (each weight is cast when used: the model is small, the cast is cheap).
Several inputs are packed into one token stream with positions restarting per input and attention kept inside each
input, so a batch gives the same vectors as one input at a time up to rounding. Attention runs in query chunks, so the
largest score block is 1024 x (1024 + window) for sliding layers and ~8M scores per head for full ones; the per-layer
inputs are computed one layer at a time.
"""
import json
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np

CHUNK = 1024
_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_DTYPES = {0: (np.float32, 4), 1: (np.float16, 2), 30: (np.uint16, 2)}     # F32, F16, BF16 (raw bits)


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


class Gemma2Embedder:
    def __init__(self, gguf_path, tokenizer_path):
        from tokenizers import Tokenizer
        meta, w = read_gguf(gguf_path)
        a = meta["general.architecture"]
        if a != "gemma-embedding2":
            raise ValueError(f"{gguf_path}: architecture {a}, expected gemma-embedding2")
        self.w = w
        mx.eval(list(w.values()))
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
        self.tok = Tokenizer.from_file(str(tokenizer_path))
        self.tok.no_padding()
        self.tok.no_truncation()
        self._tables = {}

    def tokenize(self, texts):
        return [e.ids for e in self.tok.encode_batch(texts, add_special_tokens=True)]

    def _mm(self, x, name):
        return x @ self.w[name].astype(mx.float32).T

    def _norm(self, x, name=None):
        return mx.fast.rms_norm(x, self.w[name] if name else None, self.eps)

    def _attention(self, q, k, v, pos, seg, window):
        """q (H, T, D), k/v (Hkv, T, D); attention inside each segment, within +-window when window is set."""
        T = q.shape[1]
        rep = q.shape[0] // k.shape[0]
        if rep > 1:
            k, v = mx.repeat(k, rep, axis=0), mx.repeat(v, rep, axis=0)
        out = []
        chunk = CHUNK if window else max(128, min(CHUNK, (1 << 23) // T // 128 * 128))
        for s in range(0, T, chunk):
            e = min(T, s + chunk)
            ks, ke = (max(0, s - window), min(T, e + window)) if window else (0, T)
            mask = seg[s:e, None] == seg[None, ks:ke]
            if window:
                d = pos[s:e, None] - pos[None, ks:ke]
                mask = mask & (d <= window) & (d >= -window)
            out.append(mx.fast.scaled_dot_product_attention(
                q[None, :, s:e], k[None, :, ks:ke], v[None, :, ks:ke], scale=1.0, mask=mask[None, None])[0])
        return mx.concatenate(out, axis=1) if len(out) > 1 else out[0]

    def _layer(self, il, x, ple, pos, seg):
        p = f"blk.{il}."
        T = x.shape[0]
        H, Hkv = self.n_head, self.n_head_kv[il]
        sliding = self.swa[il]
        h = self._norm(x, p + "attn_norm.weight")
        q = self._mm(h, p + "attn_q.weight").reshape(T, H, -1)
        k = self._mm(h, p + "attn_k.weight").reshape(T, Hkv, -1)
        v = self._mm(h, p + "attn_v.weight").reshape(T, Hkv, -1)
        D = q.shape[-1]
        q = self._norm(q, p + "attn_q_norm.weight").transpose(1, 0, 2)
        k = self._norm(k, p + "attn_k_norm.weight").transpose(1, 0, 2)
        v = self._norm(v).transpose(1, 0, 2)
        base = self.base[0] if sliding else self.base[1]
        cos, sin = self._rope_tables(pos, D, base)
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)
        o = self._attention(q, k, v, pos, seg, self.half_window if sliding else 0)
        o = self._mm(o.transpose(1, 0, 2).reshape(T, -1), p + "attn_output.weight")
        a = self._norm(o, p + "post_attention_norm.weight") + x
        f = self._norm(a, p + "ffn_norm.weight")
        f = self._mm(_gelu(self._mm(f, p + "ffn_gate.weight")) * self._mm(f, p + "ffn_up.weight"), p + "ffn_down.weight")
        c = self._norm(f, p + "post_ffw_norm.weight") + a
        g = _gelu(self._mm(c, p + "inp_gate.weight")) * ple
        c = c + self._norm(self._mm(g, p + "proj.weight"), p + "post_norm.weight")
        return c * self.w[p + "layer_output_scale.weight"]

    def _per_layer_input(self, x0, il):
        """Layer il's slice of the per-layer inputs: RMS norm of (x0 W_il^T) / sqrt(n_embd), as llama.cpp computes
        for all layers at once (the norm is per 512-vector, so one slice at a time is the same and needs 1/24 the memory)."""
        n = self.n_per_layer
        w = self.w["per_layer_model_proj.weight"][il * n:(il + 1) * n]
        return self._norm((x0 @ w.astype(mx.float32).T) * np.float32(1 / np.sqrt(self.n_embd)), "per_layer_proj_norm.weight")

    def _rope_tables(self, pos, D, base):
        key = (D, base)
        if key not in self._tables:
            inv = base ** (-np.arange(0, D, 2, dtype=np.float64) / D)
            self._tables[key] = mx.array(inv.astype(np.float32))
        ang = pos.astype(mx.float32)[:, None] * self._tables[key][None]
        return mx.cos(ang), mx.sin(ang)

    def forward(self, ids_list):
        """Token id lists -> (len, dim) float32 L2-normalized embeddings, all inputs in one packed pass."""
        lens = [len(ids) for ids in ids_list]
        ids = mx.array(np.concatenate([np.asarray(i, dtype=np.int32) for i in ids_list]))
        pos = mx.array(np.concatenate([np.arange(n, dtype=np.int32) for n in lens]))
        seg = mx.array(np.repeat(np.arange(len(lens), dtype=np.int32), lens))
        x = self.w["token_embd.weight"][ids].astype(mx.float32) * np.float32(np.sqrt(self.n_embd))
        x0 = x
        for il in range(self.n_layer):
            x = self._layer(il, x, self._per_layer_input(x0, il), pos, seg)
            mx.eval(x)
        x = self._norm(x, "output_norm.weight")
        onehot = (seg[None, :] == mx.arange(len(lens))[:, None]).astype(mx.float32)
        pooled = (onehot @ x) / mx.array(lens, dtype=mx.float32)[:, None]
        e = self._mm(pooled, "output.weight")
        e = e / mx.linalg.norm(e, axis=-1, keepdims=True)
        mx.eval(e)
        return e


def _gelu(x):
    return 0.5 * x * (1 + mx.tanh(np.float32(np.sqrt(2 / np.pi)) * (x + np.float32(0.044715) * x * x * x)))


def _rope(x, cos, sin):
    """NEOX rotation of the two halves of the last axis; x (H, T, D), cos/sin (T, D/2)."""
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
