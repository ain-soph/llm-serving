"""OpenAI-style /v1/embeddings for EmbeddingGemma 2 (GGUF on llama-server), meant to run behind llama-swap.

  python3 server.py --port PORT [--model GGUF] [--model-id ID] [--backend llama|mlx]

--backend llama (default) starts llama-server --embedding on a private port; --backend mlx (text only, run with an MLX
venv) computes the vectors in this process with gemma2_mlx.py from the same GGUF: no ubatch-sized compute buffer kept
allocated, inputs up to --max-tokens. Either way the answers are adapted to what the clients send:
  dimensions     < 768: Matryoshka truncation (the model is trained for 512/256/128) and L2 re-normalization;
                 > 768: zero-padding, which keeps cosine similarity and dot products unchanged (LobeChat always asks
                 for 1024 because its pgvector column is vector(1024)); 768 or absent: unchanged.
  input_type     "query"/"search_query" -> "task: search result | query: <text>", "document"/"passage"/
                 "search_document" -> "title: none | text: <text>" (EmbeddingGemma's prompts; OpenClaw sends these
                 with queryInputType/documentInputType). Absent: the text is embedded as sent (Open WebUI adds the
                 same prefixes itself via RAG_EMBEDDING_QUERY_PREFIX / RAG_EMBEDDING_CONTENT_PREFIX).
  encoding_format "base64" (the OpenAI SDK default) is honoured after the adaptation (float32 little-endian).
Vectors from llama-server are L2-normalized; the BF16 GGUF matched google/embeddinggemma-2 (sentence-transformers,
float32) to cosine 1.00000 in compare_gguf.py.

With --mmproj (vision + audio encoders) an input item may also be a list of content parts, as in chat messages:
  {"type": "text", "text": ...}, {"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}},
  {"type": "input_audio", "input_audio": {"data": "<base64 wav/mp3>"}}, {"type": "video_url", "video_url": {"url":
  "data:video/mp4;base64,..."}}
Each list gives ONE embedding of the interleaved parts (same vector space as text). Only inline base64 is accepted.
It is sent to llama-server's native /embeddings with its media marker (random per start, read from /props).
compare_mm.py: images 0.987-0.993, audio 0.976-0.997, video 0.953 cosine to the official model (preprocessing differs).
"""
import argparse
import base64
import json
import logging
import math
import os
import signal
import socket
import socketserver
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

NATIVE_DIM = 768
PREFIX = {"query": "task: search result | query: ", "search_query": "task: search result | query: ",
          "document": "title: none | text: ", "passage": "title: none | text: ", "search_document": "title: none | text: "}
log = logging.getLogger("embedding-server")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def media_b64(part):
    """A content part -> base64 payload, or None for text."""
    kind = part.get("type")
    if kind == "text":
        return None
    if kind == "input_audio":
        data = (part.get("input_audio") or {}).get("data")
    elif kind in ("image_url", "video_url"):
        v = part.get(kind)
        data = v.get("url") if isinstance(v, dict) else v
    else:
        raise ValueError(f"unknown content part type {kind!r}")
    if not isinstance(data, str) or not data:
        raise ValueError(f"{kind} needs inline base64 data")
    if data.startswith(("http://", "https://")):
        raise ValueError(f"{kind}: remote URLs are not fetched; send base64 or a data: URL")
    return data.split(",", 1)[1] if data.startswith("data:") else data


def normalize(vec):
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


def adapt(vec, dims):
    if not dims or dims == len(vec):
        return vec
    if dims < len(vec):
        v = vec[:dims]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]
    return vec + [0.0] * (dims - len(vec))


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def server_bind(self):  # skip the slow reverse lookup in HTTPServer.server_bind
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, message):
        self._json(status, {"error": {"message": message, "type": "invalid_request_error"}})

    def log_message(self, fmt, *args):
        log.info("%s %s", self.address_string(), fmt % args)

    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        if path == "":     # llama-swap's preload hook requests GET /
            return self._json(200, {"status": "ok", "model": self.server.model_id})
        if path == "/health":
            if self.server.mlx:
                return self._json(200, {"status": "ok"})
            try:
                urllib.request.urlopen(f"{self.server.backend}/health", timeout=5).read()
                return self._json(200, {"status": "ok"})
            except Exception as e:
                return self._json(503, {"status": "loading", "detail": str(e)})
        if path in ("/v1/models", "/models"):
            return self._json(200, {"object": "list", "data": [
                {"id": self.server.model_id, "object": "model", "owned_by": "local"}]})
        self._error(404, f"unknown path {self.path}")

    def do_POST(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        if path not in ("/v1/embeddings", "/embeddings"):
            return self._error(404, f"unknown path {self.path}")
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            inputs = req.get("input")
            single = isinstance(inputs, str)
            inputs = [inputs] if single else inputs
            if not isinstance(inputs, list) or not inputs:
                raise ValueError("input must be a non-empty string or list")
            if isinstance(inputs[0], dict):    # one content-part list given flat: [{"type": ...}, ...]
                inputs = [inputs]
            for t in inputs:
                if isinstance(t, list):
                    if not self.server.marker:
                        raise ValueError("this model is text-only; use embeddinggemma-2-multimodal for images/audio/video")
                    if not t or not all(isinstance(p, dict) for p in t):
                        raise ValueError("a content-part list must hold objects with a type")
                elif not isinstance(t, str):
                    raise ValueError("each input must be a string or a list of content parts")
            dims = req.get("dimensions")
            if dims is not None and (not isinstance(dims, int) or dims < 1 or dims > 8192):
                raise ValueError("dimensions must be an integer between 1 and 8192")
            fmt = req.get("encoding_format") or "float"
            if fmt not in ("float", "base64"):
                raise ValueError("encoding_format must be float or base64")
        except (ValueError, json.JSONDecodeError) as e:
            return self._error(400, str(e))
        prefix = PREFIX.get(str(req.get("input_type") or "").lower(), "")
        vecs, tokens = [None] * len(inputs), 0
        try:
            texts = [i for i, t in enumerate(inputs) if isinstance(t, str)]
            if texts and self.server.mlx:
                embedded, n = self.server.mlx.submit(self.server.embed_mlx, [prefix + inputs[i] for i in texts]).result()
                for i, vec in zip(texts, embedded):
                    vecs[i] = vec
                tokens += n
            elif texts:
                up = self._backend("/v1/embeddings", {"input": [prefix + inputs[i] for i in texts],
                                                      "model": self.server.model_id})
                for item in up["data"]:
                    vecs[texts[item["index"]]] = item["embedding"]
                tokens += (up.get("usage") or {}).get("prompt_tokens", 0)
            for i, parts in enumerate(inputs):
                if isinstance(parts, list):
                    vecs[i] = self._embed_parts(parts, prefix)
        except ValueError as e:
            return self._error(400, str(e))
        except urllib.error.HTTPError as e:
            return self._json(e.code, json.loads(e.read() or b"{}") or {"error": {"message": str(e)}})
        except Exception as e:
            return self._json(502, {"error": {"message": f"llama-server: {e}", "type": "server_error"}})
        data = []
        for i, vec in enumerate(vecs):
            vec = adapt(vec, dims)
            if fmt == "base64":
                vec = base64.b64encode(struct.pack(f"<{len(vec)}f", *vec)).decode()
            data.append({"object": "embedding", "index": i, "embedding": vec})
        self._json(200, {"object": "list", "data": data, "model": self.server.model_id,
                         "usage": {"prompt_tokens": tokens, "total_tokens": tokens}})

    def _backend(self, path, payload):
        with urllib.request.urlopen(urllib.request.Request(
                f"{self.server.backend}{path}", data=json.dumps(payload).encode(), method="POST",
                headers={"Content-Type": "application/json"}), timeout=600) as r:
            return json.loads(r.read())

    def _embed_parts(self, parts, prefix):
        """Interleaved text/media parts -> one normalized vector via llama-server's native /embeddings."""
        text, media = prefix, []
        for p in parts:
            b64 = media_b64(p)
            if b64 is None:
                text += str(p.get("text") or "")
            else:
                text += self.server.marker
                media.append(b64)
        d = self._backend("/embeddings", {"content": {"prompt_string": text, "multimodal_data": media}})
        e = d[0]["embedding"] if isinstance(d, list) else d["data"][0]["embedding"]
        return normalize(e[0] if e and isinstance(e[0], list) else e)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default=os.path.expanduser(
        "~/models/embeddinggemma-2/ggml-org/embeddinggemma-2-BF16.gguf"))
    ap.add_argument("--model-id", default="embeddinggemma-2")
    ap.add_argument("--parallel", type=int, default=4, help="llama-server slots, 8192 tokens each")
    ap.add_argument("--mmproj", help="vision/audio projector GGUF: enables content-part inputs")
    ap.add_argument("--llama-server", default="llama-server")
    ap.add_argument("--llama-arg", action="append", default=[], help="extra llama-server argument (repeatable)")
    ap.add_argument("--backend", choices=("llama", "mlx"), default="llama")
    ap.add_argument("--tokenizer", help="mlx: tokenizer.json (default: next to the GGUF)")
    ap.add_argument("--max-tokens", type=int, default=32768, help="mlx: longest input, BOS/EOS included")
    ap.add_argument("--max-batch-tokens", type=int, default=8192, help="mlx: tokens packed into one forward pass")
    ap.add_argument("--cache-mb", type=int, default=256, help="mlx: MLX buffer cache kept between requests")
    ap.add_argument("--mlx-dtype", choices=("float32", "bfloat16", "mixed", "mixed-f16"), default="float32",
                    help="mlx: activation dtype (float32 matches the official model to cosine 1.000000; mixed = "
                    "float32 elementwise with BF16 matrix products; mixed-f16 = with FP16 products and attention, as accurate "
                    "as float32 and llama.cpp, docs/performance.md)")
    ap.add_argument("--mlx-q8", action="store_true", help="mlx: 8-bit projection weights, BF16 products (W8A16)")
    ap.add_argument("--mlx-w8a8", nargs="?", const="all", help="mlx: int8 convrot projections, BF16 activations "
                    "(all, or comma-separated classes of gemma2_mlx.W8A8_CLASSES)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[embedding] %(message)s")
    if args.backend == "mlx":
        return serve_mlx(args)
    port = free_port()
    ctx = 8192 * args.parallel
    child = subprocess.Popen([args.llama_server, "-m", args.model, "--embedding", "--host", "127.0.0.1",
                              "--port", str(port), "-c", str(ctx), "-np", str(args.parallel),
                              "-b", "8192", "-ub", "8192", "--no-webui"]
                             + (["--mmproj", args.mmproj] if args.mmproj else []) + args.llama_arg,
                             stdin=subprocess.DEVNULL,
                             # llama-swap's LLAMA_API_KEY would also become llama-server's own key (same env name)
                             # and lock this adapter out; the child only listens on 127.0.0.1
                             env={k: v for k, v in os.environ.items() if k != "LLAMA_API_KEY"})
    # kill llama-server if this wrapper dies, even by SIGKILL
    subprocess.Popen(["/bin/sh", "-c", f"while kill -0 {os.getpid()} 2>/dev/null && kill -0 {child.pid} 2>/dev/null; "
                      f"do sleep 1; done; kill -TERM {child.pid} 2>/dev/null"],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)

    def stop(signum, _frame):
        child.terminate()
        try:
            child.wait(10)
        except subprocess.TimeoutExpired:
            child.kill()
        os._exit(0)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    server = Server((args.host, args.port), Handler)
    server.backend, server.model_id, server.marker, server.mlx = f"http://127.0.0.1:{port}", args.model_id, None, None
    for _ in range(240):
        if child.poll() is not None:
            sys.exit(f"llama-server exited with {child.returncode}")
        try:
            urllib.request.urlopen(f"{server.backend}/health", timeout=2).read()
            break
        except Exception:
            time.sleep(0.25)
    if args.mmproj:
        server.marker = json.loads(urllib.request.urlopen(f"{server.backend}/props", timeout=10).read())["media_marker"]
    log.info("listening on http://%s:%d -> llama-server :%d (%s)", args.host, args.port, port, args.model)
    server.serve_forever()


def serve_mlx(args):
    """Text-only: one worker thread owns the MLX model (MLX streams are per thread); requests run one at a time."""
    if args.mmproj:
        sys.exit("--backend mlx is text only; media inputs need --backend llama --mmproj")
    state = {}

    def load():
        import mlx.core as mx
        from gemma2_mlx import Gemma2Embedder
        mx.set_cache_limit(args.cache_mb << 20)
        tokenizer = args.tokenizer or os.path.join(os.path.dirname(args.model), "tokenizer.json")
        mixed = args.mlx_dtype.startswith("mixed")
        f16 = mx.float16 if args.mlx_dtype == "mixed-f16" else None
        state["model"], state["mx"] = Gemma2Embedder(args.model, tokenizer, mx.float32 if mixed else getattr(mx, args.mlx_dtype),
            True if args.mlx_w8a8 == "all" else tuple(args.mlx_w8a8.split(",")) if args.mlx_w8a8 else False,
            mixed=mixed, q8=args.mlx_q8, mm_dtype=f16, attn_dtype=f16), mx
        state["model"].forward([[2, 1]])

    def embed(texts):
        model = state["model"]
        ids = model.tokenize(texts)
        for n in map(len, ids):
            if n > args.max_tokens:
                raise ValueError(f"input ({n} tokens) is too large to process (maximum {args.max_tokens})")
        out = [None] * len(ids)
        batch, size = [], 0
        for i in sorted(range(len(ids)), key=lambda i: len(ids[i])):
            if batch and size + len(ids[i]) > args.max_batch_tokens:
                _run(model, ids, batch, out)
                batch, size = [], 0
            batch.append(i)
            size += len(ids[i])
        if batch:
            _run(model, ids, batch, out)
        return out, sum(map(len, ids))

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    worker = ThreadPoolExecutor(1)
    worker.submit(load).result()
    signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
    signal.signal(signal.SIGINT, lambda *_: os._exit(0))
    server = Server((args.host, args.port), Handler)
    server.backend, server.model_id, server.marker = None, args.model_id, None
    server.mlx, server.embed_mlx = worker, embed
    log.info("listening on http://%s:%d (MLX, %s)", args.host, args.port, args.model)
    server.serve_forever()


def _run(model, ids, batch, out):
    for i, vec in zip(batch, model.forward([ids[i] for i in batch]).tolist()):
        out[i] = vec


if __name__ == "__main__":
    main()
