"""Perplexity Decider (protocol v4 checkpoints, e.g. pplx-decider-flash-v1.3) on MLX (2026-11-16).

The official package (pplx_decider, installed from the checkpoint's source/) runs float32 on anything but CUDA and
its gated-delta layers fall back to a slow PyTorch loop there. Here only the model forward is replaced:
prompt layout, tokenization, image preprocessing, calibration temperature and answer formatting are the official
code (protocol.py, CPU only), so requests and answers match the official Decider.decide() in mode="separate" (the
evaluated computation: one sequence per question, the state and images + that question).

Model: mlx-vlm's qwen3_5 (vision tower, M-RoPE positions from image grids) with the checkpoint's attention:
full-attention layers noncausal over the sequence (padding keys masked), linear-attention (gated delta) layers
causal. vision=False drops the vision tower (text-only entry, less memory); images then raise RequestError.
Text sequences are right-padded into batches and start at position 0, which equals running each one alone (the
official "one request per call" path); sequences with images run one at a time.
Readout: hidden state at each question's marker x readout.safetensors [255, hidden], first `count` codes,
/ temperature, softmax.
"""
import math, string, threading
from pathlib import Path

import mlx.core as mx
from transformers import Qwen3VLProcessor

from pplx_decider.protocol import MAX_OPTIONS, MIN_PIXELS, PROTOCOL, Encoder, RequestError, answer, load_json

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import mlx_q8  # noqa: E402
import mlx_w8a8  # noqa: E402

DTYPES = {"bfloat16": mx.bfloat16, "float16": mx.float16, "float32": mx.float32}
ATTN_BLOCK = 2048      # queries per attention block on long sequences (fewer above 8K tokens, see attention())
TOKEN_BLOCK = 4096     # tokens per linear-attention / MLP block on one long sequence
# Language-model projections that can run W8A8 (MLXDecider(w8a8=True) takes all; a tuple of names takes those).
W8A8_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
              "gate_proj", "up_proj", "down_proj")


class MLXDecider:
    def __init__(self, checkpoint, dtype="float16", vision=True, max_length=32768, batch_tokens=8192, w8a8=False,
                 q8=()):
        """w8a8: False, True (all W8A8_NAMES) or a tuple of projection names.
        q8: projection names (not W8A8) to load as 8-bit weights + float activations (W8A16, common/mlx_q8.py)."""
        ck = Path(checkpoint)
        if load_json(ck / "joint_config.json") != PROTOCOL:
            raise ValueError("This checkpoint does not declare the chunked joint protocol (version 4)")
        config = load_json(ck / "decision_config.json")
        if config.get("attention_mode") != "noncausal_full_attention" or config.get("pooling", "last") != "last":
            raise ValueError("Expected a noncausal checkpoint with per-position readouts")
        self.temperature = float(config["temperature"])
        self.codes = list(config["codes"])
        self.vision = vision
        self.max_length, self.batch_tokens = max_length, batch_tokens
        self.lock = threading.Lock()
        self.release = lambda: None        # to_w8a8: drops the cached activation quantization after each group

        # The checkpoint's processor class (processor_config.json), named directly: importing mlx_vlm's qwen3_5 swaps
        # transformers' AutoProcessor for mlx-vlm's processor, and the preprocessing must stay the official one.
        processor = Qwen3VLProcessor.from_pretrained(str(ck))
        processor.tokenizer.padding_side = "left"
        processor.image_processor.size = {"shortest_edge": MIN_PIXELS,
                                          "longest_edge": int(config.get("max_pixels") or 262144)}
        from mlx_vlm.models import qwen3_5
        from mlx_vlm.utils import load_config, sanitize_weights, update_module_configs
        tok = processor.tokenizer
        if len(self.codes) != MAX_OPTIONS or any(
                tok.encode(code, add_special_tokens=False) != [tid]
                for code, tid in zip(self.codes, config["token_ids"], strict=True)) or any(
                not set(code) <= set(string.ascii_uppercase) for code in self.codes):
            raise ValueError("Checkpoint answer codes differ from its tokenizer")
        self.encoder = Encoder(processor, self.codes)

        cfg = load_config(ck)
        cfg.setdefault("vision_config", {})
        mcfg = update_module_configs(qwen3_5.ModelConfig.from_dict(cfg), qwen3_5, cfg, ["text", "vision"])
        self.model = qwen3_5.Model(mcfg)
        weights = {}
        for f in sorted(ck.glob("model-*.safetensors")) or [ck / "model.safetensors"]:
            weights.update(mx.load(str(f)))
        # Saved from Qwen3_5Model ("language_model.X", "visual.X"): the usual Hugging Face layout is "model.X".
        weights = {"model." + k: v for k, v in weights.items() if vision or not k.startswith("visual.")}
        weights = sanitize_weights(self.model, weights)
        if vision:
            weights = sanitize_weights(qwen3_5.VisionModel, weights, mcfg.vision_config)
        else:
            del self.model.vision_tower
        weights = sanitize_weights(qwen3_5.LanguageModel, weights, mcfg.text_config)
        target = DTYPES[dtype]
        quantized = self.w8a8_targets(W8A8_NAMES if w8a8 is True else tuple(w8a8)) if w8a8 else set()
        q8_paths = self.q8_targets(tuple(q8), quantized) if q8 else set()
        scales = {}
        for k, v in weights.items():          # one tensor at a time: never both copies of every weight (15.7 GiB)
            if k.endswith(".weight") and k[:-len(".weight")] in quantized:
                weights[k], scales[k[:-len(".weight")] + ".scale"] = mlx_w8a8.quantize_weight(v)
                mx.eval(weights[k], scales[k[:-len(".weight")] + ".scale"])
            elif k.endswith(".weight") and k[:-len(".weight")] in q8_paths:
                base = k[:-len(".weight")]
                weights[k], scales[base + ".scales"], scales[base + ".biases"] = mlx_q8.quantize_weight(v, target)
                mx.eval(weights[k], scales[base + ".scales"], scales[base + ".biases"])
            elif mx.issubdtype(v.dtype, mx.floating) and v.dtype != target:
                weights[k] = v.astype(target)
                mx.eval(weights[k])
        weights.update(scales)
        # Cast what load_weights does not set (computed buffers) first; the loaded weights already have their dtype
        # (the W8A8 scales stay float32).
        self.model.set_dtype(target)
        self.model.load_weights(list(weights.items()), strict=True)
        self.model.eval()      # training mode makes the gated-delta layers use the slow per-token ops path
        self.readout = mx.load(str(ck / "readout.safetensors"))["weight"].astype(mx.float32)   # [255, hidden]
        mx.eval(self.model.parameters(), self.readout)
        del weights
        mx.clear_cache()
        self.text = self.model.language_model.model

    def q8_targets(self, names, w8a8_paths):
        """Make the language-model projections named in `names` 8-bit weights + float activations (W8A16); their
        weights are quantized while loading. Returns the module paths."""
        if unknown := set(names) - set(W8A8_NAMES):
            raise ValueError(f"not language-model projections: {sorted(unknown)}")
        modules = dict(self.model.named_modules())
        paths = ["language_model." + p for p in mlx_w8a8.linear_paths(self.model.language_model, names)
                 if "bias" not in modules["language_model." + p]]
        paths = [p for p in paths if p not in w8a8_paths]
        from mlx.utils import tree_unflatten
        swaps = [(p, mlx_q8.layer(modules[p].weight.shape[1], modules[p].weight.shape[0])) for p in paths]
        del modules
        self.model.update_modules(tree_unflatten(swaps))
        return set(paths)

    def w8a8_targets(self, names):
        """Make the language-model projections named in `names` (default W8A8_NAMES: attention q/k/v/o, gated-delta
        in_proj_qkv / in_proj_z / out_proj, MLP) W8A8 (../../common/mlx_w8a8.py: Hadamard-rotated int8 weights and
        activations on the M5 INT8 TensorOps GEMM); their weights are quantized while loading, so the float model never
        exists at once. in_proj_a / in_proj_b (32 outputs) always stay float. Returns the module paths."""
        if unknown := set(names) - set(W8A8_NAMES):
            raise ValueError(f"not W8A8 projections: {sorted(unknown)}")
        modules = dict(self.model.named_modules())
        paths = [p for p in mlx_w8a8.linear_paths(self.model.language_model, names) if "bias" not in
                 modules["language_model." + p]]
        paths = ["language_model." + p for p in paths]
        del modules
        mlx_w8a8.replace(self.model, paths)
        self.release = mlx_w8a8.release
        return set(paths)

    def attention(self, attn, x, mask, positions):
        """Qwen3_5Attention without cache, the scores computed for ATTN_BLOCK queries at a time when the sequence is
        long (exact: softmax is per query). Head dim 256 has no fused SDPA kernel, so one 12.8K-token sequence
        otherwise holds 16 x L x L fp16 scores (4.9 GiB). Each block is evaluated before the next is built: lazily
        built blocks are independent and MLX runs them together, holding every block's scores at once."""
        q, k, v, gate, mask = attn._prepare_projected_qkv(attn.q_proj(x), attn.k_proj(x), attn.v_proj(x), None,
                                                          positions, None, mask)
        L = q.shape[2]
        block = max(256, ATTN_BLOCK * 8192 // max(L, 8192))       # constant score-block size: 16 x 2048 x 8192
        if L <= ATTN_BLOCK:
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=attn.scale, mask=mask)
        else:
            mx.eval(q, k, v)
            blocks = []
            for i in range(0, L, block):
                blocks.append(mx.fast.scaled_dot_product_attention(q[:, :, i:i + block], k, v, scale=attn.scale,
                                                                   mask=mask))
                mx.eval(blocks[-1])
            out = mx.concatenate(blocks, axis=2)
        out = out.transpose(0, 2, 1, 3).reshape(x.shape[0], L, -1)
        return attn.o_proj(out * mx.sigmoid(gate))

    def blocked(self, layer, h, positions):
        """One decoder layer on one long sequence, TOKEN_BLOCK tokens at a time where that is exact: the gated delta
        recurrence carries its conv and recurrent state across blocks in a cache (as in decoding), the MLP is per
        token; the noncausal full attention needs every token as keys (its queries are blocked in attention())."""
        from mlx_vlm.models.cache import ArraysCache
        L, out = h.shape[1], []
        if layer.is_linear:
            cache = ArraysCache(size=2)
            for i in range(0, L, TOKEN_BLOCK):
                x = h[:, i:i + TOKEN_BLOCK]
                x = x + layer.linear_attn(layer.input_layernorm(x), None, cache)
                out.append(x + layer.mlp(layer.post_attention_layernorm(x)))
                mx.eval(out[-1], cache.state)
        else:
            h = h + self.attention(layer.self_attn, layer.input_layernorm(h), None, positions)
            mx.eval(h)
            for i in range(0, L, TOKEN_BLOCK):
                x = h[:, i:i + TOKEN_BLOCK]
                out.append(x + layer.mlp(layer.post_attention_layernorm(x)))
                mx.eval(out[-1])
        return mx.concatenate(out, axis=1)

    def layers(self, h, positions, lengths=None):
        """Final hidden states: full attention noncausal (padding keys masked), linear attention causal."""
        if lengths is None and h.shape[1] > TOKEN_BLOCK:       # one long sequence: bounded intermediates
            for layer in self.text.layers:
                h = self.blocked(layer, h, positions)
                mx.eval(h)
            return self.text.norm(h)
        keep = None if lengths is None else mx.arange(h.shape[1])[None] < lengths[:, None]
        fa_mask = None if keep is None else keep[:, None, None, :]
        long = h.shape[0] * h.shape[1] > ATTN_BLOCK
        for layer in self.text.layers:
            if layer.is_linear:
                h = layer(h, mask=keep, cache=None, position_ids=positions)
            else:
                h = h + self.attention(layer.self_attn, layer.input_layernorm(h), fa_mask, positions)
                h = h + layer.mlp(layer.post_attention_layernorm(h))
            if long:
                mx.eval(h)     # one layer at a time keeps only one layer's intermediates alive
        return self.text.norm(h)

    def hidden_text(self, group, width):
        ids = mx.array([c.input_ids.tolist() + [self.encoder.pad_id] * (width - c.length) for c in group])
        positions = mx.broadcast_to(mx.arange(width)[None, None], (3, len(group), width))
        lengths = mx.array([c.length for c in group])
        return self.layers(self.text.embed_tokens(ids), positions, None if len(group) == 1 else lengths)

    def hidden_images(self, c):
        ids = mx.array([c.input_ids.tolist()])
        feats = self.model.get_input_embeddings(ids, pixel_values=mx.array(c.pixel_values.float().numpy()),
                                                image_grid_thw=mx.array(c.image_grid_thw.numpy()))
        return self.layers(feats.inputs_embeds, feats.position_ids)

    def logits(self, chunks):
        """Readout logits per chunk and question (valid codes only)."""
        out = [None] * len(chunks)
        order = sorted(range(len(chunks)), key=lambda i: -chunks[i].length)
        start = 0
        while start < len(order):
            first = chunks[order[start]]
            width, end = first.length, start + 1
            if first.pixel_values is None:
                while (end < len(order) and chunks[order[end]].pixel_values is None
                       and width * (end - start + 1) <= self.batch_tokens):
                    end += 1
            group = [chunks[i] for i in order[start:end]]
            h = self.hidden_text(group, width) if first.pixel_values is None else self.hidden_images(first)
            rows = [i for i, c in enumerate(group) for _ in c.keys]
            cols = [m for c in group for m in c.markers.tolist()]
            z = (h[mx.array(rows), mx.array(cols)].astype(mx.float32) @ self.readout.T).tolist()
            self.release()
            cursor = 0
            for i, c in zip(order[start:end], group):
                out[i] = [z[cursor + j][:n] for j, n in enumerate(c.counts)]
                cursor += len(c.keys)
            start = end
        return out

    def decide(self, request, temperature=None):
        """The official Decider.decide() answer for one request (decision questions)."""
        if request.get("images") and not self.vision:
            raise RequestError("this model entry is text only; use the multimodal entry for images")
        if any(isinstance(q, dict) and q.get("type") == "locate" for q in (request.get("questions") or {}).values()):
            raise RequestError("'locate' questions are not supported by this server")
        scale = self.temperature if temperature is None else float(temperature)
        if not math.isfinite(scale) or scale <= 0:
            raise RequestError("temperature must be positive and finite")
        with self.lock:
            chunks = self.encoder.plan(request, self.max_length, "separate")
            per_chunk = self.logits(chunks)
        logits = {k: v for c, rows in zip(chunks, per_chunk) for k, v in zip(c.keys, rows)}
        answers = {}
        for key, question in request["questions"].items():
            z = [v / scale for v in logits[key]]
            peak = max(z)
            e = [math.exp(v - peak) for v in z]
            answers[key] = answer(question, [x / sum(e) for x in e])
        return {"answers": answers,
                "usage": {"input_tokens": sum(c.length for c in chunks), "output_tokens": 0, "sequences": len(chunks)}}
