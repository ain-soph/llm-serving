"""SystemOne server (POST /v1/systemone) for Perplexity Decider checkpoints on MLX, run by llama-swap (2026-11-16).

  .venv/bin/python decisions/decider/server.py --port PORT --model CHECKPOINT [--model-id ID] [--no-vision] [--dtype float16]

Request: {"model"?, "state", "questions": {id: {"type": "noul"|"choice"|"score", "instructions"?, "criteria"?}},
"images"?: ["data:image/...;base64,..."], "temperature"?} or {"requests": [...]} (answered in order, as "results").
Answers are the official Decider format (decider_mlx.py): noul = P(true); choice = choice, probabilities {option: p},
confidence; score = score, probabilities {"0": p, ...}, legend, confidence. usage = input tokens and sequences.
Images: base64 data URLs only. The official loader opens any other string as a local file path, which a remote
client must not reach. --no-vision: text-only entry (no vision tower, 0.6 GiB less), images are refused.
Errors: 400 {"error": {"message", "type": "invalid_request_error"}} for requests the protocol rejects.
"""
import argparse, json, logging, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mlx.core as mx

import sys
from pathlib import Path

# The official pplx_decider package ships inside the checkpoint (source/src, Apache-2.0). It pins older
# torch / transformers than this environment, so it is imported from there instead of being installed.
if "--model" in sys.argv:
    sys.path.insert(0, str(Path(sys.argv[sys.argv.index("--model") + 1]) / "source/src"))

from decider_mlx import MLXDecider
from pplx_decider.protocol import LengthError, RequestError

log = logging.getLogger("decider")


def check(req):
    if not isinstance(req, dict):
        raise RequestError("each request must be an object")
    images = req.get("images") or []
    if not isinstance(images, list) or not all(isinstance(x, str) and x.startswith("data:image/") for x in images):
        raise RequestError("images must be a list of base64 data URLs (data:image/...;base64,...)")
    return {k: v for k, v in req.items() if k in ("state", "questions", "images")}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        log.info("%s " + fmt, self.client_address[0], *args)

    def reply(self, code, body):
        data = json.dumps(body, ensure_ascii=False).encode()
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
        self.reply(404, {"error": {"message": "not found", "type": "not_found_error"}})

    def do_POST(self):
        if self.path.split("?")[0] not in ("/v1/systemone", "/systemone"):
            return self.reply(404, {"error": {"message": "not found", "type": "not_found_error"}})
        t = time.perf_counter()
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"null")
            if not isinstance(body, dict):
                raise RequestError("request body must be a JSON object")
            temperature = body.get("temperature")
            batch = body.get("requests")
            if batch is not None and ("state" in body or "questions" in body):
                raise RequestError("send either requests or state and questions")
            reqs = [check(r) for r in batch] if isinstance(batch, list) else [check(body)]
            results = [self.server.decider.decide(r, temperature) for r in reqs]
        except json.JSONDecodeError as e:
            return self.reply(400, {"error": {"message": f"invalid JSON: {e}", "type": "invalid_request_error"}})
        except (RequestError, LengthError, ValueError, TypeError, KeyError) as e:
            return self.reply(400, {"error": {"message": str(e), "type": "invalid_request_error"}})
        model = body.get("model") or self.server.model_id
        log.info("%d request(s), %d tokens in %.2f s; MLX active %.2f GiB, peak %.2f GiB", len(results),
                 sum(r["usage"]["input_tokens"] for r in results), time.perf_counter() - t,
                 mx.get_active_memory() / 2**30, mx.get_peak_memory() / 2**30)
        if isinstance(batch, list):
            return self.reply(200, {"model": model, "results": [{"model": model, **r} for r in results]})
        self.reply(200, {"model": model, **results[0]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True, help="checkpoint directory")
    ap.add_argument("--model-id", default="pplx-decider-flash")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--no-vision", action="store_true", help="text-only: do not load the vision tower")
    ap.add_argument("--max-length", type=int, default=32768, help="tokens per sequence (state + one question)")
    ap.add_argument("--cache-mb", type=int, default=256, help="MLX buffer cache kept between requests")
    ap.add_argument("--w8a8", action="store_true", help="W8A8 language-model projections (INT8 TensorOps GEMM)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[decider] %(message)s")
    mx.set_cache_limit(args.cache_mb << 20)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.decider = MLXDecider(args.model, dtype=args.dtype, vision=not args.no_vision, max_length=args.max_length,
                                 w8a8=args.w8a8)
    server.model_id = args.model_id
    log.info("listening on http://%s:%d (%s, %s, %s); MLX active %.2f GiB", args.host, args.port, args.model,
             args.dtype, "text only" if args.no_vision else "text + images", mx.get_active_memory() / 2**30)
    server.serve_forever()


if __name__ == "__main__":
    main()
