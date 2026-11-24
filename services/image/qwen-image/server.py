"""OpenAI-compatible image API for Qwen-Image-2.1 (base or Turbo) via mflux, meant to run behind llama-swap.

Endpoints:
  GET  /health                  readiness probe for llama-swap's checkEndpoint
  GET  /v1/models               lists the single model ID
  POST /v1/images/generations   JSON: prompt, size ("WxH" or "auto"), n, seed, output_format, background
  POST /v1/images/edits         multipart: prompt, image[] (up to 10), optional mask, size, n, seed, background

Requests block until the image is done, so llama-swap counts them as in flight and never swaps
the model out mid-generation. Generations run one at a time; concurrent requests queue on a lock.
Images are always returned as b64_json, with "id" (kept under MEDIA_DIR, like the H3 images) and "seed".
Extension fields: seed, steps, negative_prompt + guidance (base only, true CFG), reference_resolution (edits: the
side each reference is resized to; default 1024 for 1-3 references, smaller for more, see reference_side),
prompt_enhance (off | mflux: the text encoder rewrites | llm: the chat model rewrites with the official Qwen system
prompts; the result is returned as revised_prompt, and with size auto its aspect ratio picks the canvas),
auto_mask (edits: a phrase; the text encoder locates it and only that box is repainted), lora + lora_scale (an
adapter registered in QWEN_IMAGE_LORAS by name, default scale 1.0, applied to this request only), control_image +
control_strength (alibaba-pai's Fun ControlNet-Union branch, QWEN_IMAGE_CONTROLNET: a preprocessed control map such as
canny / depth / pose / HED / lineart / MLSD / scribble / grayscale steers the layout; generations take it as base64 or a
data URL, edits as a multipart file and then inpaint image[0] where the mask is transparent; strength 0-2, default 1).
background="transparent" uses the official RGBA prompt template and returns RGBA PNG/WebP.
mask follows OpenAI: fully transparent pixels of the mask are repainted, the rest of image[0] is kept.

Usage: .venv/bin/python server.py --port PORT --model DIR [--turbo] [--model-id ID]
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
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import mlx.core as mx
from PIL import Image

from mflux.models.common.config import ModelConfig
from mflux.models.qwen21.variants.edit.qwen_image_21_edit import QwenImage21Edit

MAX_REFERENCES = 10
MAX_N = 4
DEFAULT_SIDE = 1024
MIN_SIDE, MAX_SIDE = 256, 2752   # the widest recommended size is 2752x1536 (16:9)
MAX_AREA = 2400 * 1792           # native 2K: the largest recommended size (4:3); 16:9 and 3:2 are just above 2048²
OUTPUT_FORMATS = {"png": "PNG", "jpeg": "JPEG", "webp": "WEBP"}
RGBA_TEMPLATE = ("This is an RGBA image with transparency. {}. "
                 "The image has alpha channel and the background is transparent.")
MEDIA_DIR = Path(os.environ.get("H3_MEDIA_IMAGES") or Path(__file__).resolve().parents[3] / "data/media/images")
MEDIA_KEEP_DAYS, MEDIA_KEEP_BYTES = 30, 2 << 30

log = logging.getLogger("qwen-image-server")

# ---- ControlNet (request field `control_image`, 2026-11-24) -------------------------------------------------------
# QWEN_IMAGE_CONTROLNET names the control branch (the W8A8 file written by mflux's qwen21_controlnet_w8a8, zero-copy
# streamed like the DiT). It is attached to the loaded model on the first request with a control image and shares its
# transformer, text encoder and VAE (QwenImage21Controlnet.from_model); requests without control_image never touch it.
CONTROLNET = os.environ.get("QWEN_IMAGE_CONTROLNET") or None
MAX_CONTROL_BYTES = 32 << 20


class BadRequest(Exception):
    pass


# ---- LoRA adapters (request field `lora`, 2026-11-23) ------------------------------------------------------------
# The server only applies adapters registered by name in QWEN_IMAGE_LORAS ("name=/path/file.safetensors,..."); a
# request never names a file. An adapter is applied for one generation and removed afterwards, so requests without
# `lora` run the unchanged W8A8 path (fused q/k/v and gate/up, byte-identical output). While active, each targeted
# projection runs as LoRAProjection: the same int8 convrot product (W8A8Linear.__call__) plus the low-rank update
# scale * (x A^T) B^T in the activation dtype, as mflux's unbaked LoRALinear computes it. The module keeps its
# identity and parameter names (weight, scale), so zero-copy streaming (MFLUX_ZERO_COPY_STREAM) is unaffected.
LORA_TARGETS = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "img_mlp.gate_layer", "img_mlp.proj",
                "img_mlp.out")
LORA_KEY = None  # compiled lazily (re)


def lora_registry():
    out = {}
    for item in filter(None, (s.strip() for s in os.environ.get("QWEN_IMAGE_LORAS", "").split(","))):
        name, _, path = item.partition("=")
        if not name or not path:
            raise SystemExit(f"QWEN_IMAGE_LORAS: expected name=path, got {item!r}")
        out[name.strip()] = Path(path.strip())
    return out


def read_lora(path):
    """{target module path: (A (r, in), B (out, r), alpha / r)} of a diffusers / PEFT / DiffSynth Qwen-Image LoRA.
    Every tensor must map to one of LORA_TARGETS; anything else (DoRA, LoKr, other layers) is refused."""
    import re
    global LORA_KEY
    if LORA_KEY is None:
        names = "|".join(re.escape(t) for t in LORA_TARGETS)
        LORA_KEY = re.compile(rf"^(?:transformer\.|diffusion_model\.)?transformer_blocks\.(\d+)\.({names})"
                              r"\.(lora_A|lora_B|alpha)(?:\.default)?(?:\.weight)?$")
    tensors = mx.load(str(path))
    meta = {}
    try:
        from safetensors import safe_open
        with safe_open(str(path), "np") as f:
            meta = f.metadata() or {}
    except Exception:
        pass
    peft_alpha = None
    if meta.get("lora_adapter_metadata"):
        cfg = json.loads(meta["lora_adapter_metadata"])
        a, r = cfg.get("transformer.lora_alpha", cfg.get("lora_alpha")), cfg.get("transformer.r", cfg.get("r"))
        if a is not None and r:
            peft_alpha = float(a)
    parts = {}
    for key, value in tensors.items():
        m = LORA_KEY.match(key)
        if m is None:
            raise ValueError(f"{path.name}: unsupported LoRA tensor {key}")
        parts.setdefault(f"transformer_blocks.{m.group(1)}.{m.group(2)}", {})[m.group(3)] = value
    out = {}
    for target, p in parts.items():
        if "lora_A" not in p or "lora_B" not in p:
            raise ValueError(f"{path.name}: {target} lacks lora_A or lora_B")
        rank = p["lora_A"].shape[0]
        alpha = float(p["alpha"].item()) if "alpha" in p else peft_alpha if peft_alpha is not None else rank
        out[target] = (p["lora_A"].astype(mx.bfloat16), p["lora_B"].astype(mx.bfloat16), alpha / rank)
    return out


def _lora_projection_class():
    from mlx import nn
    from mflux.models.common.quantization.w8a8 import W8A8Linear

    class LoRAProjection(nn.Module):
        """A W8A8Linear (same parameters) with a low-rank update; not a W8A8Linear subclass, so the fused q/k/v and
        gate/up paths, which would skip the update, use the per-projection call instead."""

        def __call__(self, x):
            y = W8A8Linear.__call__(self, x)
            update = (x @ self.lora_A.astype(x.dtype).T) @ self.lora_B.astype(x.dtype).T
            return y + (self.lora_scale * update).astype(y.dtype)
    return LoRAProjection, W8A8Linear


class LoRAState:
    """The adapter applied to the transformer right now (at most one), with the modules it changed."""

    def __init__(self, transformer):
        self.transformer = transformer
        self.registry = lora_registry()
        self.loaded = {}
        self.active = None
        self.modules = []
        self.cls, self.base = _lora_projection_class()

    def parse(self, req):
        name, scale = req.get("lora"), req.get("lora_scale")
        if name in (None, "", "none"):
            if scale not in (None, ""):
                raise BadRequest("lora_scale needs lora")
            return None
        if not isinstance(name, str) or name not in self.registry:
            raise BadRequest(f"unknown lora {name!r}; available: {', '.join(sorted(self.registry)) or 'none'}")
        return str(name), parse_float(scale, "lora_scale", 1.0, -2.0, 2.0)

    def apply(self, choice):
        self.remove()
        if choice is None:
            return
        name, scale = choice
        if name not in self.loaded:
            t = time.time()
            self.loaded[name] = read_lora(self.registry[name])
            log.info("lora %s: %d projections read in %.2fs", name, len(self.loaded[name]), time.time() - t)
        modules = dict(self.transformer.named_modules())
        for target, (a, b, alpha_scale) in self.loaded[name].items():
            m = modules.get(target)
            if not isinstance(m, self.base):
                self.remove()
                raise BadRequest(f"lora {name}: {target} is not a W8A8 projection of this model")
            if m.weight.ndim == 2 and a.shape[1] != m.weight.shape[1]:
                self.remove()
                raise BadRequest(f"lora {name}: {target} input width {a.shape[1]} != {m.weight.shape[1]}")
            m.__class__ = self.cls
            m.lora_A, m.lora_B, m.lora_scale = a, b, alpha_scale * scale
            self.modules.append(m)
        self.active = choice

    def remove(self):
        for m in self.modules:
            for k in ("lora_A", "lora_B"):
                m.pop(k, None)
            m.__dict__.pop("lora_scale", None)
            m.__class__ = self.base
        self.modules, self.active = [], None


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


def parse_size(size, references=()):
    """Returns (width, height). "auto"/missing keeps the last reference's aspect ratio at ~1 MP (the model's own
    default for edits), else 1024x1024. Any side from 256 through 2752 is accepted, area at most 2400x1792 (the
    largest of the official recommended sizes)."""
    if size in (None, "", "auto"):
        if references:
            w, h = references[-1].size
            scale = (DEFAULT_SIDE * DEFAULT_SIDE / (w * h)) ** 0.5
            return tuple(max(MIN_SIDE, round(v * scale / 32) * 32) for v in (w, h))
        return DEFAULT_SIDE, DEFAULT_SIDE
    try:
        w, h = (int(v) for v in str(size).lower().split("x"))
    except ValueError:
        raise BadRequest(f"size must be WIDTHxHEIGHT or auto, got {size!r}")
    if not all(MIN_SIDE <= v <= MAX_SIDE for v in (w, h)) or w * h > MAX_AREA:
        raise BadRequest(f"width and height must be from {MIN_SIDE} through {MAX_SIDE}, area at most 2400x1792")
    return w, h


def render_size(width, height):
    """The model works in 32-pixel steps: other sides render at the next multiple and are center-cropped back."""
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


def parse_float(value, name, default, lo, hi):
    if value in (None, ""):
        return default
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise BadRequest(f"{name} must be a number")
    if not lo <= v <= hi:
        raise BadRequest(f"{name} must be between {lo} and {hi}")
    return v


REF_PIXELS = float(os.environ.get("QWEN_IMAGE_REF_PIXELS", 3 * 1024 * 1024))


def reference_side(count, requested=None):
    """Side of the square area each reference image is resized to (the model's output_resolution, default 1024).
    Every reference adds ~(side/16)² tokens to each denoising step, so many references share REF_PIXELS (3 MP):
    1-3 references keep 1024, 10 get ~544, never below 512. reference_resolution overrides it (256-1024)."""
    if requested not in (None, ""):
        return parse_int(requested, "reference_resolution", 1024, 256, 1024) // 32 * 32
    side = min(1024.0, (REF_PIXELS / count) ** 0.5)
    return max(512, int(side) // 32 * 32)


PROMPT_ENHANCE = ("off", "mflux", "llm")
# Default when a request does not say (llama-swap entry env): "off" for the plain IDs, "llm" for the -pe ID.
DEFAULT_ENHANCE = os.environ.get("QWEN_IMAGE_DEFAULT_ENHANCE", "off")
# llm: the chat model rewrites the prompt with the official Qwen-Image-2.1 prompt-rewrite system prompts
# (QwenLM/Qwen-Image-2.1 prompt_rewrite/prompts, copied to ./prompt_rewrite).
LLM_URL = os.environ.get("QWEN_IMAGE_LLM_URL", "http://127.0.0.1:8080/v1/chat/completions")
LLM_MODEL = os.environ.get("QWEN_IMAGE_LLM_MODEL", "qwen3.8-flash-next")
LLM_KEY_FILE = Path(os.environ.get("QWEN_IMAGE_LLM_KEY_FILE") or Path(__file__).resolve().parents[3] / "platform/llama-swap/.api_key")
# Not redistributed here (no license stated upstream): docs/qwen-image.md shows where to download them.
SYSTEM_PROMPTS = Path(os.environ.get("QWEN_IMAGE_SYSTEM_PROMPTS") or Path(__file__).resolve().parent / "prompt_rewrite/prompts")
RATIO_SIDES = {"1:1": (2048, 2048), "4:3": (2400, 1792), "3:4": (1792, 2400), "3:2": (2528, 1696),
               "2:3": (1696, 2528), "16:9": (2752, 1536), "9:16": (1536, 2752)}


def ratio_size(ratio, area=DEFAULT_SIDE * DEFAULT_SIDE):
    """(width, height) for "W:H" at about `area` pixels, multiples of 32 (the rewrite's suggested canvas)."""
    try:
        w, h = (float(v) for v in ratio.split(":"))
    except (AttributeError, ValueError):
        return None
    if not (w > 0 and h > 0) or not 1 / 4 <= w / h <= 4:
        return None
    width = (area * w / h) ** 0.5
    return max(MIN_SIDE, round(width / 32) * 32), max(MIN_SIDE, round(width * h / w / 32) * 32)


def parse_rewrite_json(text):
    """(rewritten_prompt, wh_ratio or None) from the last JSON object in a reply, or None."""
    from mflux.models.qwen21.model.qwen21_text_encoder.grounding import QwenImage21Grounding

    text = text.split("</think>")[-1]
    for candidate in reversed(QwenImage21Grounding.balanced_json_spans(text)):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        rewritten = obj.get("rewritten_prompt") or obj.get("rewrited_prompt")
        if isinstance(rewritten, str) and rewritten.strip():
            ratio = obj.get("wh_ratio")
            return rewritten.strip(), ratio if isinstance(ratio, str) and ratio.strip() else None
    return None


def llm_rewrite(prompt, references=()):
    """Rewrite with the chat model and the official system prompt (t2i, or edit with the references attached)."""
    task = "edit" if references else "t2i"
    system = (SYSTEM_PROMPTS / f"system_prompt_{task}.txt").read_text()
    content = [{"type": "text", "text": prompt}]
    for image in references:
        buf = io.BytesIO()
        small = image.convert("RGB")
        small.thumbnail((1024, 1024))
        small.save(buf, format="JPEG", quality=90)
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}})
    # 8192 (was 4096, 2026-11-18): Flash-Next thinks before the JSON; on the Qwen-Image-Bench s23 prompts the rewrites
    # took 2068-3810 tokens (median 3631), so 4096 left almost no margin and a longer one would fail with a 400
    body = json.dumps({"model": LLM_MODEL, "temperature": 0.7, "max_tokens": 8192,
                       "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}]})
    req = urllib.request.Request(LLM_URL, data=body.encode(), headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {LLM_KEY_FILE.read_text().strip()}"})
    with urllib.request.urlopen(req, timeout=600) as r:
        reply = json.load(r)["choices"][0]["message"]["content"] or ""
    parsed = parse_rewrite_json(reply)
    if parsed is None:
        raise RuntimeError(f"the chat model's rewrite could not be parsed: {reply[:160]!r}")
    return parsed


def openai_mask(raw, size):
    """OpenAI mask (transparent = edit) -> mflux mask (white = repaint), at the source image's size."""
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im = im.convert("RGBA").resize(size, Image.Resampling.BILINEAR)
    except Exception:
        raise BadRequest("mask is not a readable image")
    return im.getchannel("A").point(lambda a: 255 - a)


def decode_image(value, name):
    """A PIL image from a base64 string or a data URL (JSON requests) or raw bytes (multipart)."""
    if isinstance(value, (bytes, bytearray)) and value[:5] == b"data:":
        value = value.decode("ascii", "replace")  # a data URL sent as a multipart text field
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("data:"):
            text = text.partition(",")[2]
        try:
            value = base64.b64decode(text, validate=False)
        except Exception:
            raise BadRequest(f"{name} must be base64 image data or a data URL")
    if not isinstance(value, (bytes, bytearray)) or not value:
        raise BadRequest(f"{name} must be an image")
    if len(value) > MAX_CONTROL_BYTES:
        raise BadRequest(f"{name} is larger than {MAX_CONTROL_BYTES >> 20} MB")
    try:
        with Image.open(io.BytesIO(value)) as im:
            from PIL import ImageOps
            return ImageOps.exif_transpose(im).convert("RGB")
    except Exception:
        raise BadRequest(f"{name} is not a readable image")


class Generator:
    def __init__(self, model_dir, turbo, quantize):
        # MLX keeps freed buffers in a cache that is resident (wired) while the GPU runs. A large one makes the
        # denoiser fast (2048²: 160 -> 119 ms per block with 6 GB instead of 2 GB, docs/performance.md), but on top of
        # the longer sequences of reference images it overflows the capacity (3 refs: 11.7 GB with 6 GB), so edits
        # with references use the small one. The VAE has its own limit (MFLUX_QWEN21_VAE_CACHE_MB).
        self.cache_bytes = int(float(os.environ.get("QWEN_IMAGE_CACHE_MB", "6144")) * 2**20)
        self.ref_cache_bytes = int(float(os.environ.get("QWEN_IMAGE_REF_CACHE_MB", "2048")) * 2**20)
        self.turbo = turbo
        # Control requests run the uncached joint stream plus the control chain; QWEN_IMAGE_CONTROL_CACHE_MB sets their
        # MLX cache (default: QWEN_IMAGE_CACHE_MB).
        self.control_cache_bytes = int(float(os.environ.get("QWEN_IMAGE_CONTROL_CACHE_MB")
                                             or os.environ.get("QWEN_IMAGE_CACHE_MB", "6144")) * 2**20)
        self.control_model = None
        # MLX streams are per thread, and the HTTP server answers each request on a new thread: load the model
        # and run every generation on one worker thread (this also serializes generations).
        self.worker = ThreadPoolExecutor(1, thread_name_prefix="mlx")
        self.worker.submit(self._load, model_dir, turbo, quantize).result()

    def _load(self, model_dir, turbo, quantize):
        config = ModelConfig.qwen_image_21_turbo() if turbo else ModelConfig.qwen_image_21()
        t = time.time()
        self.model = QwenImage21Edit(quantize=quantize, model_path=model_dir, model_config=config)
        self.lora = LoRAState(self.model.transformer)
        log.info("model ready in %.1fs (%s, q=%s)", time.time() - t, "turbo" if turbo else "base", quantize)

    def _controlnet(self):
        # worker thread only (MLX streams are per thread)
        if self.control_model is None:
            from mflux.models.qwen21.variants.controlnet.qwen_image_21_controlnet import QwenImage21Controlnet
            t = time.time()
            self.control_model = QwenImage21Controlnet.from_model(self.model, CONTROLNET)
            log.info("controlnet %s attached in %.1fs", Path(CONTROLNET).name, time.time() - t)
        return self.control_model

    def parse_control(self, req, references, mask, control):
        """(control image, strength) or None. In edits the control branch inpaints image[0] (mask required)."""
        if control is None and req.get("control_image") not in (None, ""):
            control = req.get("control_image")
        strength = req.get("control_strength")
        if control is None:
            if strength not in (None, ""):
                raise BadRequest("control_strength needs control_image")
            return None
        if CONTROLNET is None:
            raise BadRequest("control_image is not available on this model")
        image = decode_image(control, "control_image")
        if references and (len(references) != 1 or mask is None):
            raise BadRequest("with control_image, edits take exactly one image and a mask (the region to repaint)")
        if req.get("auto_mask"):
            raise BadRequest("auto_mask cannot be combined with control_image")
        return image, parse_float(strength, "control_strength", 1.0, 0.0, 2.0)

    def run(self, req, references=(), mask=None, control=None):
        prompt = req.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise BadRequest("prompt is required")
        fmt_name = req.get("output_format") or "png"
        fmt = OUTPUT_FORMATS.get(fmt_name)
        if fmt is None:
            raise BadRequest(f"output_format must be one of {', '.join(OUTPUT_FORMATS)}")
        background = req.get("background") or "auto"
        if background not in ("auto", "opaque", "transparent"):
            raise BadRequest("background must be auto, opaque or transparent")
        if background == "transparent":
            if fmt == "JPEG":
                raise BadRequest("background=transparent needs output_format png or webp")
            if (req.get("prompt_enhance") or DEFAULT_ENHANCE) == "off":
                prompt = RGBA_TEMPLATE.format(prompt.strip().rstrip("."))
        lora = self.lora.parse(req)
        control = self.parse_control(req, references, mask, control)
        enhance = req.get("prompt_enhance") or DEFAULT_ENHANCE
        if enhance not in PROMPT_ENHANCE:
            raise BadRequest(f"prompt_enhance must be one of {', '.join(PROMPT_ENHANCE)}")
        auto_mask = req.get("auto_mask")
        if auto_mask and not references:
            raise BadRequest("auto_mask needs an image to edit")
        if auto_mask and mask is not None:
            raise BadRequest("give either mask or auto_mask")
        auto_size = req.get("size") in (None, "", "auto") and not references
        width, height = parse_size(req.get("size"), references)
        if control is not None:
            # the canvas follows the control map (or the image to inpaint); a rewrite's aspect ratio does not move it
            auto_size = False
            if req.get("size") in (None, "", "auto"):
                width, height = parse_size(None, references or [control[0]])
        n = parse_int(req.get("n"), "n", 1, 1, MAX_N)
        seed = parse_int(req.get("seed"), "seed", None, 0, 2**31 - 1)
        seed = secrets.randbelow(2**31) if seed is None else seed
        steps = parse_int(req.get("steps"), "steps", None, 2, 100)
        kw = {}
        if not self.turbo and req.get("negative_prompt") is not None:
            kw = {"negative_prompt": str(req["negative_prompt"]),
                  "guidance": parse_float(req.get("guidance"), "guidance", 4.0, 1.0, 20.0)}
        if mask is not None:
            kw["mask_image"] = mask
        if references and control is None:
            kw["output_resolution"] = reference_side(len(references), req.get("reference_resolution"))
        if auto_mask:
            kw["auto_mask"] = str(auto_mask)
        revised = None
        if enhance == "llm":
            t = time.time()
            try:
                revised, ratio = llm_rewrite(req["prompt"], () if control is not None else references)
            except Exception as e:
                raise BadRequest(f"prompt_enhance=llm failed: {e}")
            log.info("llm rewrite in %.1fs: %r", time.time() - t, revised[:120])
            if auto_size and ratio_size(ratio):
                width, height = ratio_size(ratio)
            prompt = RGBA_TEMPLATE.format(revised.rstrip(".")) if background == "transparent" else revised
        rw, rh = render_size(width, height)
        args = (prompt, fmt, fmt_name, background, width, height, rw, rh, n, seed, steps, references, mask, kw, enhance,
                auto_size, revised)
        if control is not None:
            args += (control,)
        return self.worker.submit(self._generate_with_lora, lora, *args).result()

    def _generate_with_lora(self, lora, *args):
        self.lora.apply(lora)
        try:
            return self._generate(*args)
        finally:
            self.lora.remove()

    def _generate(self, prompt, fmt, fmt_name, background, width, height, rw, rh, n, seed, steps, references, mask, kw,
                  enhance="off", auto_size=False, revised=None, control=None):
        if control is not None:
            mx.set_cache_limit(self.control_cache_bytes)
        else:
            mx.set_cache_limit(self.ref_cache_bytes if references else self.cache_bytes)
        mx.clear_cache()
        if enhance == "mflux":
            # The text encoder itself rewrites (its generation head): edits with mflux's condensation of the
            # official edit recipe, text-to-image with ours of the t2i recipe.
            t = time.time()
            try:
                if references and control is None:
                    revised = self.model.rewrite_prompt(prompt, list(references))
                else:
                    revised, ratio = self.model.rewrite_text_prompt(prompt)
                    if auto_size and ratio_size(ratio):
                        width, height = ratio_size(ratio)
                        rw, rh = render_size(width, height)
            except Exception as e:
                raise BadRequest(f"prompt_enhance=mflux failed: {e}")
            log.info("mflux rewrite in %.1fs: %r", time.time() - t, revised[:120])
            prompt = RGBA_TEMPLATE.format(revised.rstrip(".")) if background == "transparent" else revised
        data = []
        for i in range(n):
            t = time.time()
            if control is not None:
                source = {"image_path": references[0], "mask_image": mask} if references else {}
                extra = {k: v for k, v in kw.items() if k in ("negative_prompt", "guidance")}
                result = self._controlnet().generate_image(
                    seed=seed + i, prompt=prompt, controlnet_image_path=control[0], controlnet_strength=control[1],
                    num_inference_steps=steps, width=rw, height=rh, **source, **extra)
            else:
                result = self.model.generate_image(seed=seed + i, prompt=prompt, width=rw, height=rh,
                                                   num_inference_steps=steps, image_paths=list(references) or None,
                                                   **kw)
            image = result.image
            mx.clear_cache()
            if (rw, rh) != (width, height):
                left, top = (rw - width) // 2, (rh - height) // 2
                image = image.crop((left, top, left + width, top + height))
            if background != "transparent" or fmt == "JPEG":
                image = image.convert("RGB")
            buf = io.BytesIO()
            image.save(buf, format=fmt)
            item = {"b64_json": base64.b64encode(buf.getvalue()).decode(), "seed": seed + i, "id": keep_media(image)}
            if revised:
                item["revised_prompt"] = revised
            data.append(item)
            log.info("generated %dx%d refs=%d mask=%s seed=%d lora=%s control=%s in %.1fs", width, height,
                     len(references), mask is not None, seed + i, self.lora.active,
                     None if control is None else control[1], time.time() - t)
        return {"created": int(time.time()), "output_format": fmt_name, "background": background, "data": data}


class Server(ThreadingHTTPServer):
    def server_bind(self):
        # HTTPServer.server_bind calls socket.getfqdn(), which can block ~35 s on this Mac; the name is unused.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


class Handler(BaseHTTPRequestHandler):
    server_version = "qwen-image-server"

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
        except (BadRequest, ValueError) as e:
            self._error(400, str(e))
        except Exception as e:
            log.exception("generation failed")
            self._error(500, f"generation failed: {e}", "server_error")

    def _generations(self):
        try:
            req = json.loads(self._body() or b"{}")
        except json.JSONDecodeError:
            raise BadRequest("body must be JSON")
        if not isinstance(req, dict):
            raise BadRequest("body must be a JSON object")
        return self.server.generator.run(req)

    def _edits(self):
        ctype = self.headers.get("Content-Type", "")
        if not ctype.startswith("multipart/form-data"):
            raise BadRequest("edits require multipart/form-data")
        msg = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
            f"Content-Type: {ctype}\r\n\r\n".encode() + self._body())
        fields, images, mask_raw, control_raw = {}, [], None, None
        for part in msg.iter_parts():
            name = part.get_param("name", header="content-disposition")
            if name in ("image", "image[]"):
                images.append(part.get_payload(decode=True))
            elif name == "mask":
                mask_raw = part.get_payload(decode=True)
            elif name == "control_image":
                control_raw = part.get_payload(decode=True)
            elif name:
                fields[name] = part.get_content().strip() if part.get_content_maintype() == "text" else None
        if not images:
            raise BadRequest("at least one image is required")
        if len(images) > MAX_REFERENCES:
            raise BadRequest(f"at most {MAX_REFERENCES} reference images are supported")
        refs = []
        for i, raw in enumerate(images):
            try:
                with Image.open(io.BytesIO(raw)) as im:
                    refs.append(im.convert("RGBA"))
            except Exception:
                raise BadRequest(f"image {i + 1} is not a readable image")
        mask = openai_mask(mask_raw, refs[0].size) if mask_raw else None
        return self.server.generator.run(fields, refs, mask, control_raw)

    def log_message(self, fmt, *args):
        log.info("%s %s", self.address_string(), fmt % args)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", required=True, help="diffusers checkpoint dir (Qwen/Qwen-Image-2.1[-Turbo])")
    parser.add_argument("--turbo", action="store_true")
    parser.add_argument("-q", "--quantize", type=int, choices=(4, 6, 8))
    parser.add_argument("--model-id", default="qwen-image-2.1")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[qwen-image] %(message)s")
    server = Server((args.host, args.port), Handler)
    server.generator = Generator(args.model, args.turbo, args.quantize)
    server.model_id = args.model_id
    log.info("listening on http://%s:%d (%s)", args.host, args.port, args.model)
    server.serve_forever()


if __name__ == "__main__":
    main()
