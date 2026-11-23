"""OpenAI-compatible image API for Z-Image-Turbo (Tongyi-MAI, Apache-2.0) via mflux, meant to run behind llama-swap.

Endpoints:
  GET  /health                  readiness probe for llama-swap's checkEndpoint
  GET  /v1/models               lists the single model ID
  POST /v1/images/generations   JSON: prompt, size ("WxH" or "auto"), n, seed, steps, output_format

Requests block until the image is done, so llama-swap counts them as in flight and never swaps the model out
mid-generation. Generations run one at a time on one MLX worker thread. Images are returned as b64_json with
"seed" and "id" (kept under MEDIA_DIR like the other image models, served by the gateway at /media/<id>).
Turbo is guidance-distilled: there is no negative prompt or guidance; steps default to 9 (the model card's setting).

Usage: python server.py --port PORT --model DIR [--model-id ID] [-q 8]
"""
import argparse
import base64
import io
import json
import logging
import os
import secrets
import socketserver
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import mlx.core as mx

from mflux.models.common.config import ModelConfig
from mflux.models.z_image import ZImage

MAX_N = 4
DEFAULT_SIDE = 1024
DEFAULT_STEPS = 9
MIN_SIDE, MAX_SIDE = 256, 2048
MAX_AREA = 2048 * 2048
ALIGN = 16  # VAE 8 x patch 2
OUTPUT_FORMATS = {"png": "PNG", "jpeg": "JPEG", "webp": "WEBP"}
MEDIA_DIR = Path(os.environ.get("H3_MEDIA_IMAGES") or Path(os.environ.get("LLM_STATE_DIR", str(Path.home() / "Library/Application Support/llm-serving"))) / "media/images")
MEDIA_KEEP_DAYS, MEDIA_KEEP_BYTES = 30, 2 << 30

log = logging.getLogger("z-image-server")


class BadRequest(Exception):
    pass


def keep_media(image):
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    mid = "img_" + secrets.token_hex(8)
    image.save(MEDIA_DIR / f"{mid}.png", format="PNG")
    files = sorted(MEDIA_DIR.glob("img_*.png"), key=lambda f: f.stat().st_mtime)
    total, cutoff = sum(f.stat().st_size for f in files), time.time() - MEDIA_KEEP_DAYS * 86400
    for f in files[:-1]:
        if f.stat().st_mtime >= cutoff and total <= MEDIA_KEEP_BYTES:
            break
        total -= f.stat().st_size
        f.unlink(missing_ok=True)
    return mid


def parse_size(size):
    if size in (None, "", "auto"):
        return DEFAULT_SIDE, DEFAULT_SIDE
    try:
        w, h = (int(v) for v in str(size).lower().split("x"))
    except ValueError:
        raise BadRequest(f"size must be WIDTHxHEIGHT or auto, got {size!r}")
    if not all(MIN_SIDE <= v <= MAX_SIDE for v in (w, h)) or w * h > MAX_AREA:
        raise BadRequest(f"width and height must be from {MIN_SIDE} through {MAX_SIDE}")
    return w, h


def render_size(width, height):
    """Other sides render at the next multiple of 16 and are center-cropped back."""
    return tuple(-(-v // ALIGN) * ALIGN for v in (width, height))


def parse_int(value, name, default, lo, hi):
    if value in (None, ""):
        return default
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise BadRequest(f"{name} must be an integer")
    if not lo <= v <= hi:
        raise BadRequest(f"{name} must be between {lo} and {hi}")
    return v


class Generator:
    def __init__(self, model_dir, quantize):
        # MLX keeps freed buffers in a cache that stays wired while the GPU runs; bound it so the wired peak stays
        # inside the llama-swap capacity (see the conf.d comment for the measured peaks).
        self.cache_bytes = int(float(os.environ.get("ZIMAGE_CACHE_MB", "2048")) * 2**20)
        # MLX streams are per thread and the HTTP server answers each request on a new thread: load the model and
        # run every generation on one worker thread (this also serializes generations).
        self.worker = ThreadPoolExecutor(1, thread_name_prefix="mlx")
        self.worker.submit(self._load, model_dir, quantize).result()

    def _load(self, model_dir, quantize):
        t = time.time()
        self.model = ZImage(quantize=quantize, model_path=model_dir, model_config=ModelConfig.z_image_turbo())
        mx.eval(self.model.parameters())
        log.info("model ready in %.1fs (bits=%s)", time.time() - t, self.model.bits)

    def run(self, req):
        prompt = req.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise BadRequest("prompt is required")
        fmt_name = req.get("output_format") or "png"
        fmt = OUTPUT_FORMATS.get(fmt_name)
        if fmt is None:
            raise BadRequest(f"output_format must be one of {', '.join(OUTPUT_FORMATS)}")
        if (req.get("background") or "auto") not in ("auto", "opaque"):
            raise BadRequest("background must be auto or opaque (this model has no transparent output)")
        if (req.get("response_format") or "b64_json") != "b64_json":
            raise BadRequest("response_format must be b64_json")
        width, height = parse_size(req.get("size"))
        n = parse_int(req.get("n"), "n", 1, 1, MAX_N)
        seed = parse_int(req.get("seed"), "seed", None, 0, 2**31 - 1)
        seed = secrets.randbelow(2**31) if seed is None else seed
        steps = parse_int(req.get("steps"), "steps", DEFAULT_STEPS, 1, 50)
        rw, rh = render_size(width, height)
        return self.worker.submit(self._generate, prompt, fmt, fmt_name, width, height, rw, rh, n, seed, steps).result()

    def _generate(self, prompt, fmt, fmt_name, width, height, rw, rh, n, seed, steps):
        mx.set_cache_limit(self.cache_bytes)
        mx.clear_cache()
        data = []
        for i in range(n):
            t = time.time()
            image = self.model.generate_image(seed=seed + i, prompt=prompt, width=rw, height=rh,
                                              num_inference_steps=steps).image
            mx.clear_cache()
            if (rw, rh) != (width, height):
                left, top = (rw - width) // 2, (rh - height) // 2
                image = image.crop((left, top, left + width, top + height))
            image = image.convert("RGB")
            buf = io.BytesIO()
            image.save(buf, format=fmt)
            data.append({"b64_json": base64.b64encode(buf.getvalue()).decode(), "seed": seed + i, "id": keep_media(image)})
            log.info("generated %dx%d steps=%d seed=%d in %.1fs", width, height, steps, seed + i, time.time() - t)
        return {"created": int(time.time()), "output_format": fmt_name, "data": data}


class Server(ThreadingHTTPServer):
    def server_bind(self):
        # HTTPServer.server_bind calls socket.getfqdn(), which can block ~35 s on this Mac; the name is unused.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


class Handler(BaseHTTPRequestHandler):
    server_version = "z-image-server"

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, message, kind="invalid_request_error"):
        self._send_json(status, {"error": {"message": message, "type": kind}})

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self._send_json(200, {"object": "list", "data": [
                {"id": self.server.model_id, "object": "model", "owned_by": "local"}]})
        else:
            self._error(404, f"unknown path {self.path}")

    def do_POST(self):
        try:
            if self.path != "/v1/images/generations":
                self._error(404, f"unknown path {self.path}")
                return
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            except json.JSONDecodeError:
                raise BadRequest("body must be JSON")
            if not isinstance(req, dict):
                raise BadRequest("body must be a JSON object")
            self._send_json(200, self.server.generator.run(req))
        except (BadRequest, ValueError) as e:
            self._error(400, str(e))
        except Exception as e:
            log.exception("generation failed")
            self._error(500, f"generation failed: {e}", "server_error")

    def log_message(self, fmt, *args):
        log.info("%s %s", self.address_string(), fmt % args)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", required=True, help="Z-Image-Turbo checkpoint dir (diffusers layout or mflux-save)")
    parser.add_argument("-q", "--quantize", type=int, choices=(4, 6, 8))
    parser.add_argument("--model-id", default="z-image-turbo")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[z-image] %(message)s")
    server = Server((args.host, args.port), Handler)
    server.generator = Generator(args.model, args.quantize)
    server.model_id = args.model_id
    log.info("listening on http://%s:%d (%s)", args.host, args.port, args.model)
    server.serve_forever()


if __name__ == "__main__":
    main()
