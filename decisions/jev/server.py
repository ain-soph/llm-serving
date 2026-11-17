"""SystemOne server (POST /v1/systemone) for Base Labs decision checkpoints (elephant-1 etc.) on MLX, run by llama-swap
(2026-11-17).

  .venv/bin/python decisions/jev/server.py --port PORT --model CHECKPOINT [--model-id ID] [--dtype float16] [--quant q8|w8a8]

Request: {"model"?, "state", "questions": {id: {"type": "noul"|"choice"|"score", "instructions"?, "criteria"?}}} or
{"requests": [...]} (answered in order, as "results"). Answers are qev's (OneJev) format: noul = P(yes); choice =
choice, probabilities {option: p}, confidence; score = score, legend, probabilities {"0": p, ...}, confidence.
usage = input tokens and output tokens (one per question). Text only: requests with images get 400.
Errors: 400 {"error": {"message", "type": "invalid_request_error"}} for requests qev rejects (schema, length limits).
"""
import argparse, json, logging, time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mlx.core as mx
from pydantic import ValidationError

from jev_mlx import MLXJev
from qev.schema import SystemOneRequest

log = logging.getLogger("jev")


class RequestError(ValueError):
    pass


def parse(req):
    if not isinstance(req, dict):
        raise RequestError("each request must be an object")
    if req.get("images"):
        raise RequestError("this model entry is text only; images are not supported")
    return SystemOneRequest(**{k: v for k, v in req.items() if k in ("state", "questions")})


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
        if self.path in ("/health", "/"):
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
            batch = body.get("requests")
            if batch is not None and ("state" in body or "questions" in body):
                raise RequestError("send either requests or state and questions")
            reqs = [parse(r) for r in batch] if isinstance(batch, list) else [parse(body)]
            results = []
            for r in reqs:
                resp, _ = self.server.worker.submit(self.server.engine.decide, r).result()
                results.append(resp.model_dump(exclude={"model"}))
        except json.JSONDecodeError as e:
            return self.reply(400, {"error": {"message": f"invalid JSON: {e}", "type": "invalid_request_error"}})
        except ValidationError as e:
            return self.reply(400, {"error": {"message": str(e), "type": "invalid_request_error"}})
        except (RequestError, ValueError, TypeError, KeyError) as e:
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
    ap.add_argument("--model-id", default="elephant-1")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--quant", choices=["q8", "w8a8"], help="8-bit weights at load (jev_mlx.MLXJev._quantize)")
    ap.add_argument("--max-branch-tokens", type=int, default=16384,
                    help="state + one question (16,384 = the model card's maximum sequence length)")
    ap.add_argument("--max-request-tokens", type=int, default=65536, help="state + all questions")
    ap.add_argument("--cache-mb", type=int, default=256, help="MLX buffer cache kept between requests")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[jev] %(message)s")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    # MLX streams belong to the thread that created them: load and run the model on one worker thread
    server.worker = ThreadPoolExecutor(1)
    server.worker.submit(mx.set_cache_limit, args.cache_mb << 20).result()
    server.engine = server.worker.submit(MLXJev, args.model, dtype=args.dtype, max_branch_tokens=args.max_branch_tokens,
                                         max_request_tokens=args.max_request_tokens, quant=args.quant).result()
    server.model_id = args.model_id
    log.info("listening on http://%s:%d (%s, %s, %s, loaded in %.1f s); MLX active %.2f GiB", args.host, args.port,
             args.model, args.dtype, args.quant or "no quantization", server.engine.load_seconds,
             mx.get_active_memory() / 2**30)
    server.serve_forever()


if __name__ == "__main__":
    main()
