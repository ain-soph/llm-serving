"""OpenAI-compatible image API for MiniMax-H3-Image via h3-image-mlx, meant to run behind llama-swap.

Endpoints:
  GET  /health                  readiness probe for llama-swap's checkEndpoint
  GET  /v1/models               lists the single model ID
  POST /v1/images/generations   JSON: prompt, size ("WxH" or "auto"), n, seed, output_format
  POST /v1/images/edits         multipart: prompt, image[] (up to 16, referenced as <Picture 1>...), size, n, seed

Requests block until the image is done, so llama-swap counts them as in flight and never swaps
the model out mid-generation. Generations run one at a time; concurrent requests queue on a lock.
Images are always returned as b64_json. Extension fields: "seed" in requests and in each data item.

Usage: .venv/bin/python server.py --port PORT [--models DIR] [--model-id ID]
"""
import argparse
import base64
import email.parser
import email.policy
import io
import json
import logging
import os
import secrets
import socketserver
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from h3_image_mlx.api import generate_result
from PIL import Image

MAX_REFERENCES = 16
MAX_N = 4
DEFAULT_SIDE = 1024
MAX_AREA = 4096 * 4096
OUTPUT_FORMATS = {"png": "PNG", "jpeg": "JPEG", "webp": "WEBP"}
# Every generated image is also kept as PNG under MEDIA_DIR (id returned as "id" next to b64_json) so that links
# such as http://127.0.0.1:8080/media/<id> work; the read-only video instance serves /media for both. Kept
# 30 days and at most 2 GiB (oldest first), like the videos.
MEDIA_DIR = Path(os.environ.get("H3_MEDIA_IMAGES") or Path(os.environ.get("LLM_STATE_DIR", str(Path.home() / "Library/Application Support/llm-serving"))) / "media/images")
MEDIA_KEEP_DAYS, MEDIA_KEEP_BYTES = 30, 2 << 30


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

log = logging.getLogger("h3-image-server")
generate_lock = threading.Lock()


class BadRequest(Exception):
    pass


def parse_size(size, references=()):
    """Returns (width, height). "auto"/missing keeps the first reference's size (rounded to 32, within
    256-4096 and 16 MP), else 1024x1024. Any side from 256 through 4096 is accepted (1920x1080, 3840x2160):
    see render_size."""
    if size in (None, "", "auto"):
        if references:
            with Image.open(references[0]) as im:
                w, h = im.size
            scale = min(1.0, (MAX_AREA / (w * h)) ** 0.5, 4096 / max(w, h))
            return tuple(min(4096, max(256, round(v * scale / 32) * 32)) for v in (w, h))
        return DEFAULT_SIDE, DEFAULT_SIDE
    try:
        w, h = (int(v) for v in str(size).lower().split("x"))
    except ValueError:
        raise BadRequest(f"size must be WIDTHxHEIGHT or auto, got {size!r}")
    if not all(256 <= v <= 4096 for v in (w, h)):
        raise BadRequest("width and height must be from 256 through 4096")
    return w, h


def render_size(width, height):
    """The model works in 32-pixel steps: a side that is not a multiple of 32 is rendered at the next one and the
    image is center-cropped back (3840x2160 renders at 3840x2176)."""
    return tuple(-(-v // 32) * 32 for v in (width, height))


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


def run_generation(models, prompt, width, height, n, seed, output_format, references=()):
    if not isinstance(prompt, str) or not prompt.strip():
        raise BadRequest("prompt is required")
    fmt = OUTPUT_FORMATS.get(output_format or "png")
    if fmt is None:
        raise BadRequest(f"output_format must be one of {', '.join(OUTPUT_FORMATS)}")
    if seed is None:
        seed = secrets.randbelow(2**31)
    # Always the streamed DiT (2026-11-16): with zero-copy weights (H3_ZERO_COPY=1, set by llama-swap) it runs at
    # resident speed with 2-6 GB wired. llama-swap targets differ only in the memory they declare.
    data = []
    with generate_lock:
        for i in range(n):
            t = time.time()
            rw, rh = render_size(width, height)
            result = generate_result(prompt, models=models, width=rw, height=rh, seed=seed + i,
                                     references=list(references), dit_mode="stream")
            image = result.image
            if (rw, rh) != (width, height):
                left, top = (rw - width) // 2, (rh - height) // 2
                image = image.crop((left, top, left + width, top + height))
            buf = io.BytesIO()
            image.save(buf, format=fmt)
            data.append({"b64_json": base64.b64encode(buf.getvalue()).decode(), "seed": seed + i,
                         "id": keep_media(image)})
            log.info("generated %dx%d refs=%d seed=%d in %.1fs", width, height, len(references), seed + i,
                     time.time() - t)
    return {"created": int(time.time()), "output_format": output_format or "png", "data": data}


class Server(ThreadingHTTPServer):
    def server_bind(self):
        # HTTPServer.server_bind calls socket.getfqdn(), which blocks ~35 s on this Mac with
        # Homebrew Python 3.15; the server name is unused, so skip the reverse lookup.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


class Handler(BaseHTTPRequestHandler):
    server_version = "h3-image-mlx-server"

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, message, kind="invalid_request_error"):
        self._send_json(status, {"error": {"message": message, "type": kind}})

    def _body(self):
        return self.rfile.read(int(self.headers.get("Content-Length") or 0))

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
            if self.path == "/v1/images/generations":
                self._send_json(200, self._generations())
            elif self.path == "/v1/images/edits":
                self._send_json(200, self._edits())
            else:
                self._error(404, f"unknown path {self.path}")
        except BadRequest as e:
            self._error(400, str(e))
        except Exception as e:
            log.exception("generation failed")
            self._error(500, f"generation failed: {e}", "server_error")

    def _generations(self):
        try:
            req = json.loads(self._body() or b"{}")
        except json.JSONDecodeError:
            raise BadRequest("body must be JSON")
        width, height = parse_size(req.get("size"))
        return run_generation(self.server.models, req.get("prompt"), width, height,
                              parse_int(req.get("n"), "n", 1, 1, MAX_N),
                              parse_int(req.get("seed"), "seed", None, 0, 2**31 - 1),
                              req.get("output_format"))

    def _edits(self):
        ctype = self.headers.get("Content-Type", "")
        if not ctype.startswith("multipart/form-data"):
            raise BadRequest("edits require multipart/form-data")
        msg = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
            f"Content-Type: {ctype}\r\n\r\n".encode() + self._body())
        fields, images = {}, []
        for part in msg.iter_parts():
            name = part.get_param("name", header="content-disposition")
            if name in ("image", "image[]"):
                images.append(part.get_payload(decode=True))
            elif name == "mask":
                raise BadRequest("mask is not supported by MiniMax-H3-Image; describe the edit in the prompt")
            elif name:
                fields[name] = part.get_content().strip() if part.get_content_maintype() == "text" else None
        if not images:
            raise BadRequest("at least one image is required")
        if len(images) > MAX_REFERENCES:
            raise BadRequest(f"at most {MAX_REFERENCES} reference images are supported")
        with tempfile.TemporaryDirectory(prefix="h3-image-refs-") as tmp:
            refs = []
            for i, raw in enumerate(images):
                path = Path(tmp) / f"ref{i}.png"
                try:
                    with Image.open(io.BytesIO(raw)) as im:
                        im.convert("RGB").save(path)
                except Exception:
                    raise BadRequest(f"image {i + 1} is not a readable image")
                refs.append(path)
            width, height = parse_size(fields.get("size"), refs)
            return run_generation(self.server.models, fields.get("prompt"), width, height,
                                  parse_int(fields.get("n"), "n", 1, 1, MAX_N),
                                  parse_int(fields.get("seed"), "seed", None, 0, 2**31 - 1),
                                  fields.get("output_format"), refs)

    def log_message(self, fmt, *args):
        log.info("%s %s", self.address_string(), fmt % args)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--models", default=str(Path.home() / "models/minimax-h3-image"))
    parser.add_argument("--model-id", default="minimax-h3-image")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[h3-image] %(message)s")
    server = Server((args.host, args.port), Handler)
    server.models, server.model_id = args.models, args.model_id
    log.info("listening on http://%s:%d (models: %s)", args.host, args.port, args.models)
    server.serve_forever()


if __name__ == "__main__":
    main()
