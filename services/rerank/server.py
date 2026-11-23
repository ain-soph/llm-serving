"""Qwen3-Reranker on MLX behind a llama.cpp-compatible /v1/rerank (2026-11-16).

  .venv/bin/python services/rerank/server.py --port PORT --model MLX_DIR [--max-batch-tokens 8192] [--cache-mb 256] [--copy]

MLX_DIR is an mlx_lm checkpoint, or a W8A8 / zero-copy copy written by ../common/qwen3_w8a8.py (w8a8.json): its
projections run on the INT8 TensorOps GEMM and its weights are mapped from the page cache instead of copied
(--copy: read them into memory as before).

POST /v1/rerank (also /rerank, /v1/reranking): {"model", "query", "documents": [str], "top_n"?, "return_documents"?,
"instruction"?} -> {"model", "object": "list", "results": [{"index", "relevance_score", "document"?}], "usage"}.
relevance_score = sigmoid(logit("yes") - logit("no")) at the last token of the official Qwen3-Reranker prompt.
Documents are sorted by length (longest first) and right-padded into batches of up to --max-batch-tokens tokens (the causal mask keeps every document exact);
only the last real token of each goes through the "yes"/"no" rows of the head (the rest of an untied head is freed at
load). Requests run one at a time.
Performance and precision tradeoffs are documented in docs/rerank.md and docs/performance.md.
"""
import argparse, json, logging, math, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.tokenizer_utils import load as load_tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared/common"))
import mlx_w8a8, qwen3_w8a8  # noqa: E402

from prompts import INSTRUCTION, PREFIX, SUFFIX

log = logging.getLogger("rerank")
lock = threading.Lock()


class Reranker:
    def __init__(self, path, max_batch_tokens, copy=False, w8a8_classes=None, float_model=None, q8_classes=()):
        self.max_batch_tokens = max_batch_tokens
        if w8a8_classes is not None:     # per-class precision: W8A8 only for these projections (INT8 audit 2026-11-17)
            self.inner, extras, _ = qwen3_w8a8.load_mixed(path, float_model, w8a8_classes, copy=copy,
                                                            q8_classes=q8_classes)
            self.tok, self.rows = load_tokenizer(Path(float_model)), extras["score_rows"]
            return
        if qwen3_w8a8.is_w8a8(path):
            self.inner, extras, _ = qwen3_w8a8.load(path, copy=copy)
            self.tok, self.rows = load_tokenizer(Path(path)), extras["score_rows"]
            return
        self.model, self.tok = load(path)
        self.inner = self.model.model
        pair = mx.array([self.tok.convert_tokens_to_ids("yes"), self.tok.convert_tokens_to_ids("no")])
        # Only the "yes"/"no" rows of the head are ever used: keep those two and free an untied head (8B: -0.6 GB).
        tied = "lm_head" not in self.model
        head = self.inner.embed_tokens if tied else self.model.lm_head
        rows = head.weight[pair]
        if "scales" in head:
            rows = mx.dequantize(rows, head.scales[pair], head.biases[pair], group_size=head.group_size, bits=head.bits)
        self.rows = rows.astype(self.inner.norm.weight.dtype)
        mx.eval(self.rows)
        if not tied:
            del self.model.lm_head
        mx.clear_cache()

    def hidden(self, ids):
        """Final hidden states. A batch above max_batch_tokens (one long document) runs its MLP (per token, so exact) in
        token chunks, one layer evaluated at a time, as embedding/qwen3/server.py does (2026-11-18): a 32K-token
        document no longer needs the whole sequence's MLP intermediates at once (evals/longcache)."""
        if ids.shape[0] * ids.shape[1] <= self.max_batch_tokens:
            return self.inner(ids)
        h = self.inner.embed_tokens(ids)
        step = max(1, self.max_batch_tokens // ids.shape[0])
        for layer in self.inner.layers:
            h = h + layer.self_attn(layer.input_layernorm(h), "causal")
            h = mx.concatenate([x + layer.mlp(layer.post_attention_layernorm(x))
                                for x in (h[:, i:i + step] for i in range(0, h.shape[1], step))], axis=1)
            mx.eval(h)
            mlx_w8a8.release()
        return self.inner.norm(h)

    def scores(self, query, docs, instruction):
        ids = [self.tok.encode(PREFIX + f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {d}" + SUFFIX)
               for d in docs]
        order = sorted(range(len(ids)), key=lambda i: -len(ids[i]))
        out, start = [0.0] * len(ids), 0
        while start < len(order):                   # longest first: each batch's width is its first document
            width, end = len(ids[order[start]]), start + 1
            while end < len(order) and width * (end - start + 1) <= self.max_batch_tokens:
                end += 1
            chunk = [ids[i] for i in order[start:end]]
            hidden = self.hidden(mx.array([x + [0] * (width - len(x)) for x in chunk]))
            last = hidden[mx.arange(len(chunk)), mx.array([len(x) - 1 for x in chunk])]
            logits = (last @ self.rows.T).astype(mx.float32)
            mx.eval(logits)
            mlx_w8a8.release()
            for i, (a, b) in zip(order[start:end], logits.tolist()):
                out[i] = 1 / (1 + math.exp(-(a - b)))
            start = end
        mx.clear_cache()        # the buffer cache (--cache-mb) only serves reuse within a request
        return out, sum(map(len, ids))


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
        if self.path not in ("/v1/rerank", "/rerank", "/v1/reranking"):
            return self.reply(404, {"error": "not found"})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            query, docs = req.get("query"), req.get("documents")
            if not isinstance(query, str) or not isinstance(docs, list) or not docs:
                return self.reply(400, {"error": "query (string) and documents (non-empty list) are required"})
            docs = [d if isinstance(d, str) else (d or {}).get("text", "") for d in docs]
            top_n = int(req.get("top_n") or len(docs))
            instruction = req.get("instruction") or INSTRUCTION
        except (ValueError, TypeError, AttributeError) as e:
            return self.reply(400, {"error": f"bad request: {e}"})
        t = time.perf_counter()
        with lock:
            scores, tokens = self.server.reranker.scores(query, docs, instruction)
        order = sorted(range(len(docs)), key=lambda i: -scores[i])[:top_n]
        results = [dict({"index": i, "relevance_score": scores[i]},
                        **({"document": {"text": docs[i]}} if req.get("return_documents") else {})) for i in order]
        log.info("reranked %d docs (%d tokens) in %.2f s", len(docs), tokens, time.perf_counter() - t)
        self.reply(200, {"model": self.server.model_id, "object": "list", "results": results,
                         "usage": {"prompt_tokens": tokens, "total_tokens": tokens}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-id", default="qwen3-reranker")
    ap.add_argument("--max-batch-tokens", type=int, default=8192)
    ap.add_argument("--cache-mb", type=int, default=256, help="MLX buffer cache kept between requests (0 vs 1024: <1%% speed)")
    ap.add_argument("--copy", action="store_true", help="W8A8 / zero-copy directory: copy the weights into memory")
    ap.add_argument("--w8a8-classes", help="with --float-model: comma-separated projections (q_proj, ..., down_proj) "
                    "taken W8A8 from --model; the rest stay float")
    ap.add_argument("--float-model", help="zero-copy float copy (qwen3_w8a8.py --float) for the other projections")
    ap.add_argument("--q8-classes", help="with --w8a8-classes: float projections to load as 8-bit weights + float "
                    "activations (W8A16, common/mlx_q8.py)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[rerank] %(message)s")
    mx.set_cache_limit(args.cache_mb << 20)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.reranker, server.model_id = Reranker(
        args.model, args.max_batch_tokens, args.copy,
        args.w8a8_classes.split(",") if args.w8a8_classes else None, args.float_model,
        tuple(args.q8_classes.split(",")) if args.q8_classes else ()), args.model_id
    log.info("listening on http://%s:%d (%s)", args.host, args.port, args.model)
    server.serve_forever()


if __name__ == "__main__":
    main()
