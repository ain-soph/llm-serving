"""Base Labs decision models (joey-1 / elephant-1 / mammoth-1, Qwen3.5) on MLX (2026-11-17).

The official runtime is OneJev's qev @81ce62f + the checkpoint's serve.patch (CUDA image, flash-linear-attention).
Here only the model forward is replaced; prompt rendering, the shared prefix / question suffix split, answer slots,
calibration and answer assembly are qev's own code (`qev.engine.DecisionEngine`, CPU only), so requests and answers
follow the official engine.

Model (readout.json + config.json `modeling_overrides`): attention `block_bidirectional` in the full-attention
layers, `causal` in the linear-attention (gated delta) layers; readout `slot_lm` = the LM-head logits of the slot
letters at the answer position (head computed in float32, as qev's `_upcast_head`).
A request is one prefill of the shared prefix (system turn + state block) and one batched forward of the question
suffixes on top of it. Under block-bidirectional attention the prefix tokens attend to the whole prefix and every
suffix token to the whole prefix and its whole suffix, so the full-attention layers need no causal mask at all once
the cache is cut at the block boundary (qev checks that boundary in `_split`); the gated-delta layers carry their
conv / recurrent state from the prefix into each branch. Right-padded suffix keys are masked.
"""
import json
import os
import sys
import threading
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "OneJev"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared/common"))
from qev import readout as ro                                      # noqa: E402
from qev.calibrate import Calibration                              # noqa: E402
from qev.engine import BranchResult, DecisionEngine, EngineInfo    # noqa: E402
from qev.prompt import DEFAULT_STYLE                               # noqa: E402

def tree_map_cast(params, dtype):
    """Floating parameters to `dtype`, except the float32 per-row scales of W8A8 layers."""
    from mlx.utils import tree_flatten, tree_unflatten
    return tree_unflatten([(k, v.astype(dtype) if mx.issubdtype(v.dtype, mx.floating) and v.dtype != dtype
                            and not (k == "scale" or k.endswith(".scale")) else v)
                           for k, v in tree_flatten(params)])


EMBED = "model.language_model.embed_tokens.weight"
# pre-quantized checkpoint (save_mix.py): MIX describes the scheme, LAYERS holds the decoder layers in MLX form,
# EMBEDS the unchanged bfloat16 embedding (and LM head when untied)
MIX, LAYERS, EMBEDS = "quant.json", "layers.safetensors", "embeds.safetensors"


def mapped_tensor(checkpoint, key):
    """A bfloat16 tensor of a safetensors checkpoint as a read-only numpy uint16 memmap (raw bfloat16 bits)."""
    import json
    import struct
    for f in sorted(Path(checkpoint).glob("*.safetensors")):
        with open(f, "rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(n))
        if key in header:
            spec = header[key]
            if spec["dtype"] != "BF16":
                raise ValueError(f"{key} is {spec['dtype']}, expected BF16")
            start = 8 + n + spec["data_offsets"][0]
            return np.memmap(f, dtype=np.uint16, mode="r", offset=start, shape=tuple(spec["shape"]))
    raise KeyError(f"{key} not found in {checkpoint}")


DTYPES = {"bfloat16": mx.bfloat16, "float16": mx.float16, "float32": mx.float32}
ATTN_BLOCK = 1024      # prefix queries per attention block at <= 8K tokens (fewer above: constant score memory)
TOKEN_BLOCK = 2048     # prefix tokens per gated-delta / MLP block
_FUSED = None


def fused_attention():
    """True when MLX has a fused (flash) SDPA kernel for head dim 256 on this device: one call over the whole sequence
    then holds no P x P scores, and blocking the queries only leaves the GPU with fewer threadgroups (M5 with the
    ain-soph/mlx serving wheel, 2026-11-23, reports/mlx-fork.md stage 3). float16 / bfloat16 only: float32 keeps the
    exact unfused blocking."""
    global _FUSED
    if _FUSED is None:
        try:
            x = mx.zeros((1, 1, 16, 256), dtype=mx.float16)
            mx.eval(mx.fast.scaled_dot_product_attention(x, x, x, scale=1.0, force_fused=True))
            _FUSED = True
        except Exception:     # no fused kernel (CPU, older GPUs): keep the exact query blocking
            _FUSED = False
    return _FUSED


class MLXJev(DecisionEngine):
    def __init__(self, checkpoint, dtype="bfloat16", max_branch_tokens=32768, max_request_tokens=65536,
                 batch_tokens=65536, quant=None, w8a8_skip=None):
        import transformers
        from mlx_vlm.models import qwen3_5
        from mlx_vlm.utils import load_config, sanitize_weights, update_module_configs

        ck = Path(checkpoint)
        self.model_path = str(ck)
        self.style = DEFAULT_STYLE
        self.max_branch_tokens, self.max_request_tokens = max_branch_tokens, max_request_tokens
        self.batch_tokens = batch_tokens
        self.lock = threading.Lock()
        self.dtype, self.dtype_name = DTYPES[dtype], dtype
        self.w8a8_skip = set(w8a8_skip or [])
        self.release = lambda: None        # w8a8: drops the cached activation quantization after each forward
        t0 = time.perf_counter()

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(str(ck), local_files_only=True)
        rcfg = ro.ReadoutConfig.load(ck)
        if not rcfg.noncausal or rcfg.head:
            raise ValueError(f"expected block-bidirectional attention with slot_lm readout, got {rcfg}")
        self.readout = ro.Readout(rcfg)
        self.template_kwargs = {"enable_thinking": False}
        self.slot_ids = self._verify_slots()
        cal = ck / "calibration.json"
        self.calibration = Calibration.load(cal) if cal.exists() else Calibration()
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        cfg = load_config(ck)
        mcfg = update_module_configs(qwen3_5.ModelConfig.from_dict(cfg), qwen3_5, cfg, ["text", "vision"])
        self.model = qwen3_5.Model(mcfg)
        del self.model.vision_tower
        # Neither the token embedding nor the LM head (1.0B parameters each in elephant-1; tied in joey-1) is loaded
        # into MLX: input rows and the float32 slot rows of the head (the only head rows slot_lm reads) come from the
        # mapped bfloat16 file (page cache, exact values).
        tied = mcfg.text_config.tie_word_embeddings
        self.embedding = mapped_tensor(ck, EMBED)
        self.head = head = self.embedding if tied else mapped_tensor(ck, "lm_head.weight")
        del self.model.language_model.model.embed_tokens
        if hasattr(self.model.language_model, "lm_head"):
            del self.model.language_model.lm_head
        mix = json.loads((ck / MIX).read_text()) if (ck / MIX).exists() else None
        if mix:
            # pre-quantized checkpoint (save_mix.py): decoder layers exactly as _quantize left them, mapped zero-copy
            if quant is not None or dtype != mix["dtype"]:
                raise ValueError(f"{ck} is pre-quantized ({mix['quant']}, {mix['dtype']}): no --quant, --dtype {mix['dtype']}")
            weights = self._mix_modules(ck)
        else:
            files = sorted(ck.glob("model-*.safetensors")) or [ck / "model.safetensors"]
            weights = {}
            for f in files:
                weights.update({k: v for k, v in mx.load(str(f)).items()
                                if not k.startswith("model.visual.") and k not in (EMBED, "lm_head.weight")})
            weights = sanitize_weights(self.model, weights)
            weights = sanitize_weights(qwen3_5.LanguageModel, weights, mcfg.text_config)
        if quant is None and not mix:
            for k, v in weights.items():          # one tensor at a time: never two copies of every weight
                if mx.issubdtype(v.dtype, mx.floating) and v.dtype != self.dtype:
                    weights[k] = v.astype(self.dtype)
                    mx.eval(weights[k])
        self.model.load_weights(list(weights.items()), strict=True)
        self.model.eval()      # training mode makes the gated-delta layers use the slow per-token path
        self.text = self.model.language_model.model
        self.slot_head = mx.array(np.ascontiguousarray(head[self.slot_ids])).view(mx.bfloat16).astype(mx.float32)
        del weights, head
        if quant is not None:
            self._quantize(quant, w8a8_skip)
        mx.eval(self.model.parameters())
        mx.clear_cache()
        self.quant = mix["quant"] if mix else quant
        self.info = EngineInfo(model_path=str(ck), model_type="qwen3_5", dtype=dtype, device="mlx", fork_mode="batched",
                               fork_max_abs_diff=None, transformers_version=transformers.__version__,
                               torch_version="", extra={"attention": rcfg.attention, "readout": rcfg.readout,
                                                        "calibration": self.calibration.source})
        self.load_seconds = time.perf_counter() - t0

    def _mix_modules(self, ck):
        """Tensors of a pre-quantized checkpoint (zero-copy where page-aligned), with each linear layer swapped for
        the type its stored tensors belong to: `.scale` = W8A8Linear, `.scales` = MLX 8-bit (group 32)."""
        import mlx.nn as nn
        import mlx_w8a8
        weights = mlx_w8a8.load(ck / LAYERS)
        w8a8 = [k[:-len(".scale")] for k in weights if k.endswith(".scale")]
        q8 = {k[:-len(".scales")] for k in weights if k.endswith(".scales")}
        mlx_w8a8.replace(self.model, w8a8)
        nn.quantize(self.model, group_size=32, bits=8, class_predicate=lambda p, m: p in q8)
        if w8a8:
            self.release = mlx_w8a8.release
        return weights

    def save_mix(self, out, source):
        """Write this engine (loaded with --quant) as a pre-quantized checkpoint: the decoder layers as they are now
        (LAYERS, page-aligned for zero-copy loading), the bfloat16 embedding and LM head copied unchanged (EMBEDS,
        read through mapped_tensor as before), the small files of `source`, and MIX describing the scheme."""
        import shutil
        from mlx.utils import tree_flatten
        import mlx_w8a8
        out, source = Path(out), Path(source)
        out.mkdir(parents=True, exist_ok=True)
        for f in source.iterdir():
            if f.name.startswith(".") or f.suffix == ".safetensors" or f.name == "model.safetensors.index.json":
                continue
            (shutil.copytree if f.is_dir() else shutil.copy2)(f, out / f.name)
        mlx_w8a8.save(out / LAYERS, dict(tree_flatten(self.model.parameters())), {"format": "mlx"})
        tables = {EMBED: self.embedding}
        if self.head is not self.embedding:
            tables["lm_head.weight"] = self.head
        mlx_w8a8.save(out / EMBEDS, {k: mx.array(np.ascontiguousarray(v)).view(mx.bfloat16) for k, v in tables.items()},
                      {"format": "pt"})
        (out / MIX).write_text(json.dumps({"quant": self.quant, "w8a8_skip": sorted(self.w8a8_skip), "dtype": self.dtype_name,
                                           "source": str(source), "written_by": "decisions/jev/save_mix.py",
                                           "date": time.strftime("%Y-%m-%d")}, indent=1) + "\n")
        return out

    def _quantize(self, quant, w8a8_skip=None):
        """8-bit weights at load, one decoder layer at a time from the lazily mapped bfloat16 file, so the bfloat16
        copy never exists in full: "q8" = MLX affine 8-bit, group 32, scales in the activation dtype (weight-only,
        like Q8_0); "w8a8" = per-row int8 weights of the Hadamard-rotated projections with int8 activations on the
        M5 INT8 TensorOps GEMM (shared/common/mlx_w8a8.py). The token embedding is not in MLX (`_rows`)."""
        import mlx.nn as nn
        if quant == "q8":
            def pred(path, m):
                return isinstance(m, nn.Linear) and m.weight.shape[-1] % 32 == 0
            for layer in self.text.layers:
                nn.quantize(layer, group_size=32, bits=8, class_predicate=pred)
                layer.update(tree_map_cast(layer.parameters(), self.dtype))
                mx.eval(layer.parameters())
            self.text.norm.update(tree_map_cast(self.text.norm.parameters(), self.dtype))
        elif quant == "w8a8":
            import mlx_w8a8
            names = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
                     "gate_proj", "up_proj", "down_proj")
            # skip: projections that stay 8-bit weight-only (q8) instead of W8A8. A projection whose input is not
            # shared with another (o_proj, out_proj) pays the activation rotate-quantize alone, which loses to q8 on
            # ~200-token prefills (decisions/jev/results/p1b-*). Default: none (env JEV_W8A8_SKIP=a,b overrides).
            skip = set(w8a8_skip if w8a8_skip is not None else
                       [n for n in os.environ.get("JEV_W8A8_SKIP", "").split(",") if n])
            names = tuple(n for n in names if n not in skip)
            self.w8a8_skip = skip

            def q8_pred(path, m):
                return (isinstance(m, nn.Linear) and path.rpartition(".")[2] in skip
                        and m.weight.shape[-1] % 32 == 0)
            for layer in self.text.layers:
                modules = dict(layer.named_modules())
                for path in mlx_w8a8.linear_paths(layer, names):
                    old = modules[path]
                    if old.weight.shape[1] % 256:
                        continue
                    new = mlx_w8a8.W8A8Linear(old.weight.shape[1], old.weight.shape[0])
                    new.weight, new.scale = mlx_w8a8.quantize_weight(old.weight)
                    parent, _, name = path.rpartition(".")
                    setattr(modules[parent], name, new)
                if skip:
                    nn.quantize(layer, group_size=32, bits=8, class_predicate=q8_pred)
                layer.update(tree_map_cast(layer.parameters(), self.dtype))
                mx.eval(layer.parameters())
            self.text.norm.update(tree_map_cast(self.text.norm.parameters(), self.dtype))
            self.release = mlx_w8a8.release
        else:
            raise ValueError(f"unknown quantization {quant!r}")

    # ---- forward --------------------------------------------------------------------------------------------------

    def _rows(self, ids):
        """Embedding rows of token ids (any nesting) from the mapped bfloat16 table, in the activation dtype."""
        rows = np.ascontiguousarray(self.embedding[np.asarray(ids, dtype=np.int64)])
        return mx.array(rows).view(mx.bfloat16).astype(self.dtype)

    @staticmethod
    def _positions(start, length, rows=1):
        return mx.broadcast_to(mx.arange(start, start + length)[None, None], (3, rows, length))

    def _prefill(self, prefix_ids):
        """Prefix forward (block 0, bidirectional in the full-attention layers): per layer the rotated keys / values
        (full attention) or the conv and recurrent state (gated delta) at the end of the prefix."""
        from mlx_vlm.models.cache import ArraysCache
        P = len(prefix_ids)
        h = self._rows([prefix_ids])
        positions = self._positions(0, P)
        states = []
        for layer in self.text.layers:
            out = []
            if layer.is_linear:
                cache = ArraysCache(size=2)
                for i in range(0, P, TOKEN_BLOCK):
                    x = h[:, i:i + TOKEN_BLOCK]
                    x = x + layer.linear_attn(layer.input_layernorm(x), None, cache)
                    out.append(x + layer.mlp(layer.post_attention_layernorm(x)))
                    mx.eval(out[-1], cache.cache[0], cache.cache[1])
                states.append((cache.cache[0], cache.cache[1]))
            else:
                attn = layer.self_attn
                x = layer.input_layernorm(h)
                q, k, v, gate, _ = attn._prepare_projected_qkv(attn.q_proj(x), attn.k_proj(x), attn.v_proj(x), None,
                                                               positions, None, None)
                mx.eval(q, k, v, gate)
                if P > ATTN_BLOCK and q.dtype != mx.float32 and fused_attention():   # one fused call
                    o = mx.fast.scaled_dot_product_attention(q, k, v, scale=attn.scale, force_fused=True)
                else:
                    blocks = []
                    block = max(256, ATTN_BLOCK * 8192 // max(P, 8192))   # constant score-block size
                    for i in range(0, P, block):
                        blocks.append(mx.fast.scaled_dot_product_attention(q[:, :, i:i + block], k, v,
                                                                           scale=attn.scale))
                        mx.eval(blocks[-1])
                    o = mx.concatenate(blocks, axis=2)
                o = o.transpose(0, 2, 1, 3).reshape(1, P, -1)
                h = h + attn.o_proj(o * mx.sigmoid(gate))
                mx.eval(h)
                for i in range(0, P, TOKEN_BLOCK):
                    x = h[:, i:i + TOKEN_BLOCK]
                    out.append(x + layer.mlp(layer.post_attention_layernorm(x)))
                    mx.eval(out[-1])
                states.append((k, v))
            h = mx.concatenate(out, axis=1)
            mx.eval(h)
        return states

    def _suffixes(self, states, P, sufs):
        """Slot logits (float32) of right-padded question suffixes on top of the prefix states."""
        from mlx_vlm.models.cache import ArraysCache
        n, width = len(sufs), max(len(s) for s in sufs)
        lengths = mx.array([len(s) for s in sufs])
        ids = [s + [self.pad_id] * (width - len(s)) for s in sufs]
        keep = mx.arange(width)[None] < lengths[:, None]
        key_mask = mx.concatenate([mx.ones((n, P), dtype=mx.bool_), keep], axis=1)[:, None, None, :]
        positions = self._positions(P, width, n)
        h = self._rows(ids)
        for layer, state in zip(self.text.layers, states):
            if layer.is_linear:
                cache = ArraysCache(size=2)
                cache.cache[0] = mx.repeat(state[0], n, axis=0)
                cache.cache[1] = mx.repeat(state[1], n, axis=0)
                h = layer(h, mask=keep, cache=cache)
            else:
                attn = layer.self_attn
                x = layer.input_layernorm(h)
                q, k, v, gate, _ = attn._prepare_projected_qkv(attn.q_proj(x), attn.k_proj(x), attn.v_proj(x), None,
                                                               positions, None, None)
                pk, pv = state
                k = mx.concatenate([mx.broadcast_to(pk, (n,) + pk.shape[1:]), k], axis=2)
                v = mx.concatenate([mx.broadcast_to(pv, (n,) + pv.shape[1:]), v], axis=2)
                o = mx.fast.scaled_dot_product_attention(q, k, v, scale=attn.scale, mask=key_mask)
                o = o.transpose(0, 2, 1, 3).reshape(n, width, -1)
                h = h + attn.o_proj(o * mx.sigmoid(gate))
                h = h + layer.mlp(layer.post_attention_layernorm(h))
            mx.eval(h)
        last = self.text.norm(h[mx.arange(n), lengths - 1])
        return last.astype(mx.float32) @ self.slot_head.T

    def _branches(self, state, rendered):
        prefix_ids, suffixes, _ = self._split(state, rendered)
        P = len(prefix_ids)
        states = self._prefill(prefix_ids)
        self.release()
        results = [None] * len(rendered)
        order = sorted(range(len(rendered)), key=lambda i: -len(suffixes[i]))
        start = 0
        while start < len(order):
            width = len(suffixes[order[start]])
            rows = max(1, self.batch_tokens // (P + width))
            idx = order[start:start + rows]
            logits = self._suffixes(states, P, [suffixes[i] for i in idx])
            mx.eval(logits)
            self.release()
            for row, i in enumerate(idx):
                r = rendered[i]
                results[i] = BranchResult(r.qid, r.kind, r.labels, r.descriptions,
                                          logits[row, : r.n_slots].tolist(), float("nan"), len(suffixes[i]))
            start += len(idx)
        del states
        return results

    def _run_batched(self, state, rendered):
        return self._branches(state, rendered)

    def _run_sequential(self, state, rendered):
        return self._branches(state, rendered)

    def decide(self, request, debias=1, debug=False):
        with self.lock:
            try:
                return super().decide(request, debias=debias, debug=debug)
            finally:
                mx.clear_cache()
