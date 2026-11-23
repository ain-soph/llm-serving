"""OpenAI-compatible image API for MiniMax-H3-Image via h3-image-mlx, meant to run behind llama-swap.

Endpoints:
  GET  /health                  readiness probe for llama-swap's checkEndpoint
  GET  /v1/models               lists the single model ID
  POST /v1/images/generations   JSON: prompt, size ("WxH" or "auto"), n, seed, output_format
  POST /v1/images/edits         multipart: prompt, image[] (up to 16, referenced as <Picture 1>...), size, n, seed

Requests block until the image is done, so llama-swap counts them as in flight and never swaps
the model out mid-generation. Generations run one at a time; concurrent requests queue on a lock.
Images are always returned as b64_json. Extension fields: "seed" in requests and in each data item.

Prompt expansion (2026-11-17), request field "prompt_enhance" (default --default-enhance):
  off       the prompt as given
  official  MiniMax-H3-Image-PE (llama-swap model H3_PE_MODEL, mlx_vlm.server over the official NVFP4 weights) with the
            released T2I / edit task prompts and request parameters
  llm       the same released prompts sent to the chat model (H3_PE_LLM_MODEL, Flash-Next)
The expansion returns an English caption and an aspect ratio; with size "auto" the ratio picks the canvas (at the
area of a default image, or of the first reference for edits). Each data item then carries "revised_prompt".
The official PE model does not fit next to Flash-Next: calling it makes llama-swap unload the chat model.

Generation area (2026-11-17): MiniMax recommends 4-20 MP (about 16 MP best) and its own service renders requests below
min_generation_area (default 2048²) on a larger canvas of the same shape, then resizes back. H3_MIN_GENERATION_AREA
(pixels, 0 = off) or the request field "min_generation_area" does the same here.

Usage: .venv/bin/python server.py --port PORT [--models DIR] [--model-id ID] [--default-enhance off|official|llm]
"""
import argparse
import base64
import email.parser
import email.policy
import io
import json
import logging
import os
import math
import re
import secrets
import socketserver
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from h3_image_mlx.api import generate_result
from PIL import Image

MAX_REFERENCES = 16
MAX_N = 4
DEFAULT_SIDE = 1024
# h3_image_mlx.pipeline.generate renders sides that are multiples of 32 from 256 through 4096 and raises ValueError
# otherwise (a 500 here before 2026-11-23); requests are checked against that limit up front and refused with 400.
# The official 16 MP sizes with a side above 4096 (4736x3552, 5024x3360, 5472x3072, 6272x2688) are not supported.
PIPELINE_MAX_SIDE = 4096
MAX_SIDE = min(int(os.environ.get("H3_MAX_SIDE") or PIPELINE_MAX_SIDE), PIPELINE_MAX_SIDE)
MAX_AREA = int(os.environ.get("H3_MAX_AREA") or 17_000_000)
MIN_GENERATION_AREA = int(os.environ.get("H3_MIN_GENERATION_AREA") or 0)
OUTPUT_FORMATS = {"png": "PNG", "jpeg": "JPEG", "webp": "WEBP"}

# Prompt expansion: MiniMax-H3-Image-PE's released prompts and request parameters (prompt/manifest.json hashes match).
PE_DIR = Path(os.environ.get("H3_PE_DIR") or Path.home() / "models/minimax-h3-image-pe")
PE_URL = os.environ.get("H3_PE_URL") or "http://127.0.0.1:8080/v1/chat/completions"
PE_MODEL = os.environ.get("H3_PE_MODEL") or "minimax-h3-image-pe-expander"
PE_EFFORT = os.environ.get("H3_PE_EFFORT") or "xhigh"     # the chat template accepts xhigh, medium, low
PE_LLM_MODEL = os.environ.get("H3_PE_LLM_MODEL") or "qwen3.8-flash-next"
API_KEY_FILE = Path(os.environ.get("H3_API_KEY_FILE") or Path(__file__).resolve().parents[3] / "platform/llama-swap/.api_key")
ENHANCE_MODES = ("off", "official", "llm")
PE_ATTEMPTS = 3
PE_LOAD_WAIT_S = 600    # mlx_vlm.server loading the PE weights (17.9 GB) after llama-swap already saw /health
T2I_RATIOS = ("1:1", "5:4", "4:3", "3:2", "16:9", "21:9", "4:5", "3:4", "2:3", "9:16")
# Every generated image is also kept as PNG under MEDIA_DIR (id returned as "id" next to b64_json) so that links
# such as http://127.0.0.1:8080/media/<id> work; the read-only video instance serves /media for both. Kept
# 30 days and at most 2 GiB (oldest first), like the videos.
MEDIA_DIR = Path(os.environ.get("H3_MEDIA_IMAGES") or Path(__file__).resolve().parents[3] / "data/media/images")
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


def fit_ratio(ratio, area):
    """(width, height) in 32-pixel steps with the W:H ratio and about the given area, within the side and area caps."""
    rw, rh = (int(v) for v in ratio.split(":"))
    scale = min(math.sqrt(min(area, MAX_AREA) / (rw * rh)), MAX_SIDE / max(rw, rh))
    return tuple(min(MAX_SIDE, max(256, round(v * scale / 32) * 32)) for v in (rw, rh))


def parse_size(size, references=()):
    """Returns (width, height), or None for "auto"/missing (decided after prompt expansion by default_size).
    Any side from 256 through MAX_SIDE (4096, the pipeline's limit) with an area up to MAX_AREA is accepted
    (1920x1080, 3840x2160, 4096x4096): see render_size."""
    if size in (None, "", "auto"):
        return None
    try:
        w, h = (int(v) for v in str(size).lower().split("x"))
    except ValueError:
        raise BadRequest(f"size must be WIDTHxHEIGHT or auto, got {size!r}")
    if not all(256 <= v <= MAX_SIDE for v in (w, h)):
        raise BadRequest(f"size {w}x{h} is not supported: width and height must each be from 256 through {MAX_SIDE} "
                         f"pixels (MiniMax-H3-Image renders at most {MAX_SIDE}x{MAX_SIDE})")
    if w * h > MAX_AREA:
        raise BadRequest(f"width x height must be at most {MAX_AREA} pixels")
    return w, h


def default_size(references=(), ratio=None):
    """size "auto": the expansion's aspect ratio if any, at the area of the first reference (edits) or of a
    DEFAULT_SIDE² image; without a ratio, the first reference's size or DEFAULT_SIDE²."""
    ref = None
    if references:
        with Image.open(references[0]) as im:
            ref = im.size
    area = ref[0] * ref[1] if ref else DEFAULT_SIDE * DEFAULT_SIDE
    if ratio:
        return fit_ratio(ratio, area)
    if ref:
        return fit_ratio(f"{ref[0]}:{ref[1]}", area)
    return DEFAULT_SIDE, DEFAULT_SIDE


def render_size(width, height, min_area=0):
    """The size the model renders at. Below min_area the canvas is scaled up with the same shape (MiniMax's
    min_generation_area) and the image is resized back afterwards; otherwise a side that is not a multiple of 32 is
    rendered at the next one and the image is center-cropped back (3840x2160 renders at 3840x2176)."""
    if min_area and width * height < min_area:
        scale = math.sqrt(min_area / (width * height))
        return tuple(-(-round(v * scale) // 32) * 32 for v in (width, height))
    return tuple(-(-v // 32) * 32 for v in (width, height))


def check_render_size(width, height, min_area):
    """BadRequest unless the canvas render_size picks stays within the pipeline's side limit (a min_generation_area
    larger than the requested image scales the canvas up and can push a side past it)."""
    rw, rh = render_size(width, height, min_area)
    if max(rw, rh) > MAX_SIDE:
        raise BadRequest(f"min_generation_area {min_area} would render {width}x{height} on a {rw}x{rh} canvas; "
                         f"MiniMax-H3-Image renders at most {MAX_SIDE} pixels per side: lower min_generation_area "
                         f"(0 = off) or the size")


def reduced_ratio(width, height):
    g = math.gcd(width, height)
    return f"{width // g}:{height // g}"


def parse_pe_json(text, keys):
    """The JSON object in a PE answer (thinking already separated by the server; a leftover <think> block is
    stripped). Returns None unless it has exactly the expected string fields."""
    text = (text or "").split("</think>")[-1].strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or set(obj) != set(keys) or not all(isinstance(obj[k], str) for k in keys):
        return None
    return obj


def pe_request(payload):
    """One chat completion from the expansion model. mlx_vlm.server answers /health (llama-swap's readiness check)
    while it is still loading the weights and refuses requests with 503 "model_loading" + Retry-After meanwhile
    (2026-11-23 go-live: the first -pe request failed so); wait for the load, up to PE_LOAD_WAIT_S."""
    data = json.dumps(payload).encode()
    deadline = time.time() + PE_LOAD_WAIT_S
    while True:
        req = urllib.request.Request(PE_URL, data=data, headers={
            "Content-Type": "application/json", "Authorization": "Bearer " + API_KEY_FILE.read_text().strip()})
        try:
            with urllib.request.urlopen(req, timeout=1200) as r:
                reply = json.load(r)
            return reply["choices"][0], reply.get("usage") or {}
        except urllib.error.HTTPError as e:
            body = e.read()
            if e.code != 503 or b"model_loading" not in body or time.time() > deadline:
                raise
            time.sleep(min(float(e.headers.get("Retry-After") or 1), 5))


def expand_prompt(prompt, mode, references=(), ratio_selection="auto", effort=None):
    """(caption, aspect ratio, info) from MiniMax-H3-Image-PE's released T2I / edit prompts, sent to the official PE
    model (mode "official") or to the chat model (mode "llm"). Retries an answer that is not the required JSON object.
    info: attempts, completion tokens and finish reason of the accepted answer."""
    params = json.loads((PE_DIR / "prompt/t2i/request_params.json").read_text())
    params["reasoning_effort"] = effort or PE_EFFORT
    if mode == "llm":
        # The chat model thinks longer and has its own sampling defaults; keep only the schema-free essentials.
        params = {"temperature": params["temperature"], "top_p": params["top_p"], "max_tokens": 8192}
    if references:
        task = (PE_DIR / "prompt/edit/task_prompt.txt").read_text(encoding="utf-8")
        content = [{"type": "text", "text": task}, {"type": "text", "text": (
            f"Reference Image Count:\n{len(references)} input image(s), labeled <Picture 1> through "
            f"<Picture {len(references)}>.")}]
        for i, path in enumerate(references, start=1):
            with Image.open(path) as im:
                ratio = reduced_ratio(*im.size)
            data = base64.b64encode(Path(path).read_bytes()).decode()
            content += [{"type": "text", "text": f"Reference Media:\n- <Picture {i}> [aspect_ratio {ratio}]"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}]
        selection = "adaptive" if ratio_selection == "auto" else ratio_selection
        content.append({"type": "text", "text": (
            f"\nUser Editing Instruction (JSON string):\n{json.dumps(prompt, ensure_ascii=False)}\n\n"
            f"User Aspect Ratio Selection:\n{selection}\n\n"
            "Apply the instructions above to the supplied inputs and reference images. "
            "Output only the required two-field JSON object.")})
        messages, keys, ratio_key = [{"role": "user", "content": content}], ("expanded_caption", "target_aspect_ratio"), \
            "target_aspect_ratio"
        if mode == "official":
            params = {k: v for k, v in params.items() if k != "response_format"}
            params["response_format"] = {"type": "json_schema", "json_schema": {"name": "edit_caption", "strict": True,
                "schema": {"type": "object", "additionalProperties": False,
                           "required": ["expanded_caption", "target_aspect_ratio"],
                           "properties": {"expanded_caption": {"type": "string"},
                                          "target_aspect_ratio": {"type": "string"}}}}}
    else:
        system = (PE_DIR / "prompt/t2i/system_prompt.txt").read_text(encoding="utf-8").strip()
        task = (PE_DIR / "prompt/t2i/task_prompt.txt").read_text(encoding="utf-8")
        messages = [{"role": "system", "content": system}, {"role": "user", "content": task.format(prompt)}]
        keys, ratio_key = ("expanded_caption", "aspect_ratio"), "aspect_ratio"
    model = PE_MODEL if mode == "official" else PE_LLM_MODEL
    for attempt in range(1, PE_ATTEMPTS + 1):
        t = time.time()
        choice, usage = pe_request({**params, "model": model, "messages": messages, "stream": False})
        obj = parse_pe_json(choice["message"].get("content"), keys)
        ratio = obj and obj[ratio_key]
        valid = obj and obj["expanded_caption"].strip() and (
            ratio in T2I_RATIOS if not references else re.fullmatch(r"[1-9]\d*:[1-9]\d*", ratio or ""))
        log.info("prompt expansion %s attempt %d: %s, %s tokens, finish %s, in %.1fs", mode, attempt,
                 "ok" if valid else "invalid", usage.get("completion_tokens"), choice.get("finish_reason"),
                 time.time() - t)
        if valid:
            return obj["expanded_caption"].strip(), ratio, {
                "attempts": attempt, "tokens": usage.get("completion_tokens"),
                "finish_reason": choice.get("finish_reason")}
    raise RuntimeError(f"prompt expansion ({mode}) gave no valid answer in {PE_ATTEMPTS} attempts")


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


def run_generation(models, prompt, size, n, seed, output_format, references=(), enhance="off", min_area=None):
    if not isinstance(prompt, str) or not prompt.strip():
        raise BadRequest("prompt is required")
    fmt = OUTPUT_FORMATS.get(output_format or "png")
    if fmt is None:
        raise BadRequest(f"output_format must be one of {', '.join(OUTPUT_FORMATS)}")
    if enhance not in ENHANCE_MODES:
        raise BadRequest(f"prompt_enhance must be one of {', '.join(ENHANCE_MODES)}")
    if seed is None:
        seed = secrets.randbelow(2**31)
    min_area = MIN_GENERATION_AREA if min_area is None else min_area
    if size:
        check_render_size(*size, min_area)      # before the prompt expansion
    revised = None
    ratio = None
    if enhance != "off":
        # Outside the lock: the expansion runs on another llama-swap model.
        selection = "auto" if size is None else reduced_ratio(*size)
        revised, ratio, _ = expand_prompt(prompt, enhance, references, selection)
    width, height = size or default_size(references, ratio)
    check_render_size(width, height, min_area)
    # Always the streamed DiT (2026-11-16): with zero-copy weights (H3_ZERO_COPY=1, set by llama-swap) it runs at
    # resident speed with 2-6 GB wired. llama-swap targets differ only in the memory they declare.
    data = []
    with generate_lock:
        for i in range(n):
            t = time.time()
            rw, rh = render_size(width, height, min_area)
            result = generate_result(revised or prompt, models=models, width=rw, height=rh, seed=seed + i,
                                     references=list(references), dit_mode="stream")
            image = result.image
            if (rw, rh) != (width, height):
                if rw * rh > width * height * 1.1:      # rendered on a larger canvas: resize back
                    image = image.resize((width, height), Image.LANCZOS)
                else:                                    # padded to a multiple of 32: crop back
                    left, top = (rw - width) // 2, (rh - height) // 2
                    image = image.crop((left, top, left + width, top + height))
            buf = io.BytesIO()
            image.save(buf, format=fmt)
            item = {"b64_json": base64.b64encode(buf.getvalue()).decode(), "seed": seed + i, "id": keep_media(image)}
            if revised:
                item["revised_prompt"] = revised
            data.append(item)
            log.info("generated %dx%d (rendered %dx%d) refs=%d enhance=%s seed=%d in %.1fs", width, height, rw, rh,
                     len(references), enhance, seed + i, time.time() - t)
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
        return run_generation(self.server.models, req.get("prompt"), parse_size(req.get("size")),
                              parse_int(req.get("n"), "n", 1, 1, MAX_N),
                              parse_int(req.get("seed"), "seed", None, 0, 2**31 - 1),
                              req.get("output_format"), enhance=req.get("prompt_enhance") or self.server.enhance,
                              min_area=parse_int(req.get("min_generation_area"), "min_generation_area", None, 0,
                                                 MAX_AREA))

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
            return run_generation(self.server.models, fields.get("prompt"), parse_size(fields.get("size")),
                                  parse_int(fields.get("n"), "n", 1, 1, MAX_N),
                                  parse_int(fields.get("seed"), "seed", None, 0, 2**31 - 1),
                                  fields.get("output_format"), refs,
                                  enhance=fields.get("prompt_enhance") or self.server.enhance,
                                  min_area=parse_int(fields.get("min_generation_area"), "min_generation_area", None,
                                                     0, MAX_AREA))

    def log_message(self, fmt, *args):
        log.info("%s %s", self.address_string(), fmt % args)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--models", default=str(Path.home() / "models/minimax-h3-image"))
    parser.add_argument("--model-id", default="minimax-h3-image")
    parser.add_argument("--default-enhance", choices=ENHANCE_MODES, default="off",
                        help="prompt expansion when a request does not set prompt_enhance")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[h3-image] %(message)s")
    server = Server((args.host, args.port), Handler)
    server.models, server.model_id, server.enhance = args.models, args.model_id, args.default_enhance
    log.info("listening on http://%s:%d (models: %s)", args.host, args.port, args.models)
    server.serve_forever()


if __name__ == "__main__":
    main()
