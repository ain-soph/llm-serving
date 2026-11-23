"""OpenAI-style /v1/embeddings for Qwen3-Embedding and voyage-4-nano on MLX, behind llama-swap (2026-11-16).

  .venv/bin/python services/embedding/qwen3/server.py --port PORT --model DIR [--model-id ID] [--max-batch-tokens 8192]

DIR is a copy written by convert_embed.py: float16 (--fp16) or exact Q8_0 with fp16 activations (--q8_0). The original
BF16 checkpoint also loads (--dtype float16 casts it, but briefly holds both copies). A W8A8 / zero-copy copy written by
../../common/qwen3_w8a8.py (w8a8.json) runs its projections on the INT8 TensorOps GEMM and maps its weights from the
page cache (--copy: read them into memory).
  Qwen3-Embedding  causal Qwen3, last-token pooling (the tokenizer appends <|endoftext|>), L2 normalize.
  voyage-4-nano    bidirectional Qwen3 + linear 1024 -> 2048, mean pooling over all tokens, L2 normalize.
                   (The projection is applied after the mean: it has no bias, so the result is the same.)
Request fields, as embedding/server.py (EmbeddingGemma) so clients can switch models:
  input            string or list of strings.
  input_type       "query"/"search_query" -> the model's query prompt (Qwen3: "Instruct: {instruction}\\nQuery:{text}",
                   instruction = request "instruction" or the web-search default; voyage: its query prompt);
                   "document"/"passage"/"search_document" -> Qwen3: as sent, voyage: its document prompt.
                   Absent: embedded as sent.
  dimensions       < native: Matryoshka truncation + L2 re-normalization (both models are trained for it);
                   > native: zero padding (cosine and dot products unchanged); absent: native.
  encoding_format  "float" (default) or "base64" (float32 little-endian).
Inputs are sorted by length and packed into right-padded batches of up to --max-batch-tokens tokens (causal: padding
never reaches real tokens; bidirectional: padding keys are masked), so every vector equals the unbatched one up to
rounding. Inputs longer than --max-tokens are truncated (the model limit is 32768). Requests run one at a time.
"""
import argparse, base64, json, logging, math, struct, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3 import ModelArgs, Qwen3Model
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared/common"))
import mlx_w8a8, qwen3_w8a8  # noqa: E402

log = logging.getLogger("embed")
lock = threading.Lock()

QWEN_TASK = "Given a web search query, retrieve relevant passages that answer the query"
VOYAGE_QUERY = "Represent the query for retrieving supporting documents: "
VOYAGE_DOC = "Represent the document for retrieval: "
QUERY_TYPES = {"query", "search_query"}
DOC_TYPES = {"document", "passage", "search_document"}


class Embedder:
    def __init__(self, path, max_batch_tokens, max_tokens, dtype=None, copy=False, w8a8_classes=None,
                 float_model=None, q8_classes=()):
        path = Path(path)
        cfg = json.loads((path / "config.json").read_text())
        self.tok = AutoTokenizer.from_pretrained(str(path))
        self.max_batch_tokens, self.max_tokens = max_batch_tokens, max_tokens
        if w8a8_classes is not None or qwen3_w8a8.is_w8a8(path):
            if w8a8_classes is not None:  # per-class precision (INT8 audit 2026-11-17)
                self.model, extras, _ = qwen3_w8a8.load_mixed(path, float_model, w8a8_classes, copy=copy,
                                                                q8_classes=q8_classes)
            else:
                self.model, extras, _ = qwen3_w8a8.load(path, copy=copy)
            self.proj = extras.get("linear.weight")
            self.voyage = self.proj is not None
            self.dim = self.proj.shape[0] if self.voyage else cfg["hidden_size"]
            return
        weights = {}
        for f in sorted(path.glob("*.safetensors")):
            weights.update(mx.load(str(f)))
        weights = {k.removeprefix("model."): v for k, v in weights.items() if not k.startswith("lm_head")}
        self.proj = weights.pop("linear.weight", None)         # voyage-4-nano's output projection
        self.voyage = self.proj is not None
        self.model = Qwen3Model(ModelArgs.from_dict(cfg))
        if q := cfg.get("quantization"):
            nn.quantize(self.model, group_size=q["group_size"], bits=q["bits"],
                        class_predicate=lambda p, m: f"{p}.scales" in weights)
        self.model.load_weights(list(weights.items()))
        if dtype:
            self.model.set_dtype(getattr(mx, dtype))         # activations follow the weights' float dtype
        mx.eval(self.model.parameters())
        if self.proj is not None:
            self.proj = self.proj.astype(mx.float32)
            mx.eval(self.proj)
        del weights
        mx.clear_cache()
        self.dim = self.proj.shape[0] if self.voyage else cfg["hidden_size"]

    def prompt(self, text, input_type, instruction):
        if input_type in QUERY_TYPES:
            return VOYAGE_QUERY + text if self.voyage else f"Instruct: {instruction or QWEN_TASK}\nQuery:{text}"
        if input_type in DOC_TYPES and self.voyage:
            return VOYAGE_DOC + text
        return text

    def layers(self, h, mask):
        """The decoder layers; above max_batch_tokens the MLP (per token, so exact) runs in token chunks, one layer
        evaluated at a time: 8B on 32K tokens peaks at +1.4 GiB instead of +3.5 GiB."""
        long = h.shape[0] * h.shape[1] > self.max_batch_tokens
        step = max(1, self.max_batch_tokens // h.shape[0])
        for layer in self.model.layers:
            h = h + layer.self_attn(layer.input_layernorm(h), mask)
            if long:
                h = mx.concatenate([x + layer.mlp(layer.post_attention_layernorm(x))
                                    for x in (h[:, i:i + step] for i in range(0, h.shape[1], step))], axis=1)
                mx.eval(h)
            else:
                h = h + layer.mlp(layer.post_attention_layernorm(h))
        return self.model.norm(h)

    def forward(self, ids, lengths):
        h = self.model.embed_tokens(ids)
        if not self.voyage:                                      # causal: right padding never reaches real tokens
            h = self.layers(h, "causal")
            return h[mx.arange(len(lengths)), lengths - 1].astype(mx.float32)
        keep = mx.arange(ids.shape[1])[None] < lengths[:, None]  # [B, L] real tokens
        h = self.layers(h, keep[:, None, None, :]).astype(mx.float32)   # bidirectional, padding keys masked
        mean = (h * keep[..., None]).sum(axis=1) / lengths[:, None].astype(mx.float32)
        return mean @ self.proj.T

    def embed(self, texts):
        ids = [self.tok(t)["input_ids"][: self.max_tokens] for t in texts]
        order = sorted(range(len(ids)), key=lambda i: -len(ids[i]))
        out, start = [None] * len(ids), 0
        while start < len(order):                    # longest first: each batch's width is its first input
            width, end = len(ids[order[start]]), start + 1
            while end < len(order) and width * (end - start + 1) <= self.max_batch_tokens:
                end += 1
            chunk = [ids[i] for i in order[start:end]]
            batch = mx.array([x + [0] * (width - len(x)) for x in chunk])
            v = self.forward(batch, mx.array([len(x) for x in chunk]))
            v = v / mx.maximum(mx.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
            mx.eval(v)
            mlx_w8a8.release()
            for i, row in zip(order[start:end], v.tolist()):
                out[i] = row
            start = end
        mx.clear_cache()        # the buffer cache (--cache-mb) only serves reuse within a request
        return out, sum(map(len, ids))


def resize(vec, dims):
    if not dims or dims == len(vec):
        return vec
    if dims > len(vec):
        return vec + [0.0] * (dims - len(vec))
    vec = vec[:dims]
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        log.info("%s " + fmt, self.client_address[0], *args)

    def reply(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            return self.reply(200, {"status": "ok"})
        if self.path in ("/v1/models", "/models"):
            return self.reply(200, {"object": "list", "data": [{"id": self.server.model_id, "object": "model"}]})
        self.reply(404, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/v1/embeddings", "/embeddings"):
            return self.reply(404, {"error": "not found"})
        emb = self.server.embedder
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            inputs = req.get("input")
            inputs = [inputs] if isinstance(inputs, str) else inputs
            if not isinstance(inputs, list) or not inputs or not all(isinstance(x, str) for x in inputs):
                return self.reply(400, {"error": {"message": "input must be a string or a non-empty list of strings",
                                                  "type": "invalid_request_error"}})
            dims = int(req["dimensions"]) if req.get("dimensions") is not None else None
            if dims is not None and dims < 1:
                raise ValueError("dimensions must be positive")
            fmt = req.get("encoding_format") or "float"
            if fmt not in ("float", "base64"):
                raise ValueError("encoding_format must be float or base64")
            texts = [emb.prompt(x, req.get("input_type"), req.get("instruction")) for x in inputs]
        except (ValueError, TypeError) as e:
            return self.reply(400, {"error": {"message": f"bad request: {e}", "type": "invalid_request_error"}})
        t = time.perf_counter()
        with lock:
            vecs, tokens = emb.embed(texts)
        data = []
        for i, v in enumerate(vecs):
            v = resize(v, dims)
            if fmt == "base64":
                v = base64.b64encode(struct.pack(f"<{len(v)}f", *v)).decode()
            data.append({"object": "embedding", "index": i, "embedding": v})
        log.info("embedded %d inputs (%d tokens) in %.2f s; MLX active %.2f GiB, peak %.2f GiB", len(texts), tokens,
                 time.perf_counter() - t, mx.get_active_memory() / 2**30, mx.get_peak_memory() / 2**30)
        self.reply(200, {"object": "list", "data": data, "model": self.server.model_id,
                         "usage": {"prompt_tokens": tokens, "total_tokens": tokens}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-id", default="qwen3-embedding")
    ap.add_argument("--max-batch-tokens", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=32768, help="longer inputs are truncated")
    ap.add_argument("--cache-mb", type=int, default=256, help="MLX buffer cache kept between requests")
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], help="cast the (unquantized) weights")
    ap.add_argument("--copy", action="store_true", help="W8A8 / zero-copy directory: copy the weights into memory")
    ap.add_argument("--w8a8-classes", help="with --float-model: comma-separated projections (q_proj, ..., down_proj) "
                    "taken W8A8 from --model; the rest stay float")
    ap.add_argument("--float-model", help="zero-copy float copy (qwen3_w8a8.py --float) for the other projections")
    ap.add_argument("--q8-classes", help="with --w8a8-classes: float projections to load as 8-bit weights + float "
                    "activations (W8A16, common/mlx_q8.py)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[embed] %(message)s")
    mx.set_cache_limit(args.cache_mb << 20)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.embedder = Embedder(args.model, args.max_batch_tokens, args.max_tokens, args.dtype, args.copy,
                               args.w8a8_classes.split(",") if args.w8a8_classes else None, args.float_model,
                               tuple(args.q8_classes.split(",")) if args.q8_classes else ())
    server.model_id = args.model_id
    log.info("listening on http://%s:%d (%s, %d dims); MLX active %.2f GiB, load peak %.2f GiB", args.host, args.port,
             args.model, server.embedder.dim, mx.get_active_memory() / 2**30, mx.get_peak_memory() / 2**30)
    server.serve_forever()


if __name__ == "__main__":
    main()
