"""Video API for MiniMax-H3 (video + synchronized stereo audio) via antirez/h3.c, meant to run behind llama-swap.

Endpoints:
  GET  /health                     readiness probe for llama-swap's checkEndpoint
  GET  /v1/models                  lists the single model ID
  POST /v1/videos                  JSON or multipart/form-data, see below; returns the queued job at once (OpenAI style)
  GET  /v1/videos                  recent jobs (newest first)
  GET  /v1/videos/{id}             job object (status, progress, timings)
  GET  /v1/videos/{id}/content     the MP4 (H.264 + 32 kHz stereo AAC); ?part=segment: an extension's new part
  GET  /v1/videos/share/{id}?exp=&sig=   the MP4 via a signed, expiring link ("share_url" in a completed job): for
                                   players that cannot send the API key (chat embeds). An optional external proxy may add authentication for this path
                                   only; the wrapper checks the HMAC (key derived from llama-swap's API key) and expiry.
  GET  /media/{id}                 watch page (player + download, or progress until done); /media/{id}.mp4 the file,
                                   /media/{id}/status JSON. Served by the read-only instance; the gateway in front of
                                   llama-swap (~/llm-serving/gateway) routes /media and /v1/videos here.
  Downloads honour Range requests (Safari's video player needs them).

POST /v1/videos fields (JSON values or multipart form fields):
  prompt            required. Describe scene, action, camera, look and the desired sound.
  seconds | frames  duration; default 5 s, at most 15 s (362 frames) at any size. H3 rounds frames up to 5+17n at
                    24 fps (5 s -> 124 frames). 1344x768: standard 5 s ~4.5 min (15 s not measured with the wide
                    sparse layout; dense was ~43 min); fast 5 s ~3 min, 15 s ~13 min; best 5 s ~12 min.
  size              "WxH", multiples of 32, W*H <= 768*1344 (default 1344x768; 768x1344 portrait).
  seed              integer (default random)
  preset            four tiers, fastest first (1344x768, 5 s): "fast" (~3 min; tight block-sparse attention + token
                    reduction, composition may drift), "turbo" (~3.3 min; the community Turbo distillation LoRA folded
                    into the DiT, 6 passes: sharp, close to best in look, but a different take of the prompt; text and
                    first/last-frame renders up to 8 s only, anything else falls back to standard and the job says
                    so in settings.fallback), "standard" (default, ~4.5 min; wide block-sparse attention: no
                    visible defect in the hard-case study, the picture differs slightly from the exact one), "best"
                    (exact: all 50 blocks, every pass, dense attention, ~12 min). Old names: "quality" = best,
                    "draft" = fast. Knobs override it: steps, reuse, layers, core_reuse, token_reduction (bool),
                    sparse (false / "dense": exact attention; "relaxed": the standard layout; "fast": the tight layout;
                    true: the preset's own layout, "relaxed" for best), render_size ("WxH", lower internal canvas,
                    same aspect).
  first_frame, last_frame, input_reference (= first_frame)   FL2VA keyframe images: multipart files, or in
                    JSON a base64 string / data: URL / https URL on an allowed host (H3_VIDEO_FETCH_HOSTS,
                    default empty: clients may explicitly allow their upload host).
  Ref2VA references (need ~/models/minimax-h3/Ref2VA; cannot be mixed with first/last frames):
  reference_images  up to 9 images -> <Picture 1>, <Picture 2>, ...
  reference_videos  up to 3 videos (2-15 s each, 15 s in total) -> <Video 1>, ...; an item is a video (its soundtrack is
                    used, if any) or {"video": ..., "audio": false | true | <replacement audio>}. Multipart:
                    reference_videos[] (with sound) / reference_silent_videos[].
  reference_audio   up to 3 audio clips. <Audio N> numbers video soundtracks first (in video order), then these.
  video             your own starting video (base64 / data: URL / allowed https URL, or a multipart file): continued
                    exactly like extend_from. It is converted to 24 fps, 32 kHz stereo (silence if it has none) and the
                    largest supported size with its aspect ratio, not above its own resolution (or "size", scaled to cover and
                    centre-cropped); the
                    whole upload stays at the start of the result. 2 s to 10 min.
  extend_from       id of a completed video: continue it. Its last extend_context_seconds (default 3, 2-15) become
                    <Video 1> + <Audio 1>. H3 re-creates the reference first and then goes on, so the render is context
                    + "seconds" long (at most 15 s: with 3 s of context a step adds up to 12 s); the earlier video minus
                    that context is followed by the whole render, so the join falls where the re-creation starts (measured: join
                    step 9.3 vs 4.3 between ordinary frames; the re-created seconds look alike but drift from the
                    original, so a longer context replaces more of it). A plain
                    prompt ("what happens next") is wrapped in the official continuation structure ([video continuation
                    + audio reference]); one with subject_definitions: is used as is. Size follows the earlier video.
                    content returns the whole video (all parts encoded once from their sources), ?part=segment the raw
                    render. Chain extensions for longer videos.
                    Media fields take base64, data: URLs or https URLs on H3_VIDEO_FETCH_HOSTS.
  wait              true: block until the video is done (see below). Default false: async, poll GET /v1/videos/{id}.
  response_format   "json" (default): the job object; "b64_json": job object plus the MP4 as base64 in
                    "b64_json"; "mp4": the MP4 bytes (also chosen by "Accept: video/mp4"). b64_json and mp4 imply wait.

Async (default, as OpenAI's /v1/videos and what LobeChat expects): the POST answers at once with the job
("queued") and the render runs in the background. So that llama-swap does not swap this model out mid-render
(it only waits for requests in flight), the wrapper opens a "hold" request to itself THROUGH llama-swap
(GET /upstream/<model-id>/v1/videos/<id>/hold) that stays open until the render ends; the POST answers only once
the hold has arrived. The job file is rewritten as progress changes, so the read-only instance reports it.
Waiting mode (wait / b64_json / mp4) keeps the old behaviour (only for short renders: the public proxy cuts a request
after 1200 s, so long videos must use the async default): the POST itself stays open; for JSON responses the
status line goes out at once and then a space every 15 s (proxy read timeouts), and a client hang-up cancels the
render. Renders run one at a time; a POST while one runs gets 429 (Retry-After: 60) instead of queueing.
Each render runs the h3 CLI as a child process, so the wrapper holds no model memory while idle.

The body's optional "model" picks the preset per request: "minimax-h3-video[-fast|-standard|-best]" (old
-quality/-draft and -stream still work; -stream is now an alias: one zero-copy mode since 2026-11-17; the suffix sets
the preset unless "preset" is given); any other
value (or none) uses --mode and the default preset. This lets one public URL (an optional reverse proxy maps /v1/videos onto this
wrapper) serve every variant, and lets clients that only send a model name (LobeChat) pick the preset.
--files-only runs a no-render instance for GET (status, MP4) that refuses POST, so fetching a finished video
through llama-swap does not reserve render memory or evict other models.

Usage: python3 server.py --port PORT [--models DIR] [--model-id ID] [--mode resident|stream] [--files-only]
"""
import argparse
import base64
import binascii
import email.parser
import email.policy
import json
import logging
import math
import os
import re
import secrets
import shutil
import signal
import socketserver
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
H3_DIR = Path(os.environ.get("H3_VIDEO_SOURCE", str(HERE / "h3.c")))
H3_BIN = Path(os.environ.get("H3_VIDEO_BIN") or H3_DIR / "h3")  # override only for plumbing tests
MAX_AREA = 768 * 1344
FPS = 24
MAX_FRAMES = 362          # 15.08 s = the model maximum; r11 768p 362f fast: 2727 s, footprint 36.0 GB (2026-11-15)
MAX_STEPS = 20            # = the quality preset; bounds the worst render (~16 min, 768p stream) for proxy timeouts
DEFAULT_SIZE = (1344, 768)
DEFAULT_SECONDS = 5.0
KEEPALIVE_S = 15
KEEP_DAYS = 30            # outputs older than this are deleted ...
KEEP_BYTES = 2 << 30      # ... and the oldest go first while all outputs exceed 2 GiB (user choice, 2026-11-16)
# body "model" -> DiT mode. Since 2026-11-17 the packed INT8 DiT is mapped zero-copy (h3.c fork 183be4a): resident
# loads in 0.33 s and renders as fast as --ssd-streaming did (outputs byte-identical), so there is one llama-swap
# entry and the old -stream name is an alias of it (old clients keep working). --mode stream still exists for tests.
MODE_BY_MODEL = {"minimax-h3-video": "resident", "minimax-h3-video-stream": "resident"}
MODEL_BY_MODE = {"resident": "minimax-h3-video", "stream": "minimax-h3-video"}
MODEL_RE = re.compile(r"(minimax-h3-video(?:-stream)?)(?:-(fast|turbo|standard|best|quality|draft))?")
LSW_URL = os.environ.get("H3_VIDEO_LSW_URL", "http://127.0.0.1:8080")       # where the hold request goes
LSW_KEY_FILE = Path(os.environ.get("H3_VIDEO_LSW_KEY_FILE") or Path(os.environ.get("LLM_STATE_DIR", str(Path.home() / "Library/Application Support/llm-serving"))) / "api_key")
FETCH_HOSTS = {h for h in os.environ.get("H3_VIDEO_FETCH_HOSTS", "").split(",") if h}
FETCH_MAX = 100 << 20     # reference videos can be large
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("FFPROBE", "ffprobe")
MEDIA_IMAGES = Path(os.environ.get("H3_MEDIA_IMAGES") or Path(os.environ.get("LLM_STATE_DIR", str(Path.home() / "Library/Application Support/llm-serving"))) / "media/images")
SHARE_BASE = os.environ.get("H3_VIDEO_SHARE_BASE", "http://127.0.0.1:8080")
SHARE_TTL = 30 * 86400    # signed links stay valid 30 days after the render
DEFAULT_CONTEXT = 3.0     # s of the earlier video an extension re-creates first (H3 Ref2VA): 3 s keeps more of the
                          # original (that part is replaced by the re-creation) and leaves up to 12 s of new content
HOLD_WAIT_S = 15          # how long an async POST waits for its hold request to arrive
# Tiers. Times: 1344x768, 5 s.
#   best: exact. All 50 DiT blocks, every pass computed, dense attention (MPSGraph SDPA).
#   standard: 45 blocks, reuse 2, wide block-sparse attention ("relaxed" layout, BF16). User decision 2026-11-18 after
#         the hard-case study (reports/h3-optimization.md: whip pan, leave and re-enter, first frame, 15 s shot):
#         no visible defect, 0.2-3.9 dB below the noise floor of an equivalent dense kernel, ~25% faster than dense
#         (357 -> 270 s). Before that it was dense (set "sparse": false to get that back).
#   fast: standard with the tight sparse layout + token reduction (pooled video tokens in blocks 4-30): composition
#         drifts (2-5 dB below the noise floor), clean pictures.
#   turbo (user decision 2026-11-18): a separate DiT, ~/models/minimax-h3-turbo (larryvrh MiniMax-H3-Turbo-Lora v4
#         step600 EMA folded into the FL2VA weights with h3.c tools/fold_lora.py, then INT8-packed), 6 passes, all 50
#         blocks, reuse 1 (the distilled schedule has no redundancy to reuse), relaxed sparse layout. Measured: 768p 5 s
#         197-218 s (standard 288-318, best 689-761); sharp, first frame kept as well as best. Not with token reduction
#         or the tight layout (ghosted double images); 15 s shots drift (light, hair), hence the 8 s cap. The LoRA is
#         trained on FL2VA only: Ref2VA references and continuations fall back to standard.
PRESETS = {
    "fast": {"steps": 20, "layers": 45, "reuse": 2, "sparse": "fast", "token_reduction": True},    # 5 s ~3 min, 15 s ~13
    "turbo": {"steps": 6, "layers": 50, "reuse": 1, "sparse": "relaxed"},                           # 5 s ~3.3 min
    "standard": {"steps": 20, "layers": 45, "reuse": 2, "sparse": "relaxed"},                       # 5 s ~4.5 min
    "best": {"steps": 20, "layers": 50, "reuse": 1},                                                # 5 s ~12 min
}
PRESET_ALIASES = {"quality": "best", "draft": "fast"}   # names used before 2026-11-16
TURBO_MODELS = os.environ.get("H3_VIDEO_TURBO_MODELS")   # default: minimax-h3-turbo next to --models
TURBO_MAX_SECONDS = 8.0
DEFAULT_PRESET = "standard"
# Every preset: video VAE in BF16 on TensorOps (H3_VAE_BF16=2, PSNR ~52 dB vs F32, 2.3x faster decode).
H3_ENV = {"H3_VAE_BF16": "2"}
# Sparse attention (h3.c H3_SPARSE): a query block keeps the key blocks of the frames within t0 and, further away
# in time, those within a spatial radius that halves as the distance doubles (r0 >> level, at least rmin); sink=1 also
# keeps frame 0 for every query. Dense for the first evaluations (layout forms there) and the first/last block; with
# fewer than 10 passes (a custom "steps") only the first two stay dense.
# BF16 attention since 2026-11-17 (INT8 audit): int8 Q@K^T saved 5 of 160 s on a 768p 5 s fast render (DiT
# 128.4 vs 134.2 s) with 14x the attention error (rel. RMS 0.021 vs 0.0015).
SPARSE_LAYOUTS = {"fast": "t0=1,r0=4,rmin=0,sink=0", "relaxed": "t0=1,r0=12,rmin=2,sink=1"}


def sparse_env(layout, steps):
    dense_steps = 4 if steps >= 10 else 2
    return {"H3_ATTN": "nax",
            "H3_SPARSE": f"{SPARSE_LAYOUTS[layout]},dense_head=1,dense_tail=1,dense_steps={dense_steps}"}


def parse_sparse(value, preset_layout):
    """The "sparse" knob: false / "dense" / "off" -> None (exact attention); a layout name; true -> the preset's own
    layout, or "relaxed" where the preset is dense."""
    text = str(value).strip().lower()
    if text in SPARSE_LAYOUTS:
        return text
    if text in ("dense", "off", "none") or (not isinstance(value, bool) and text in ("0", "false", "no")):
        return None
    if isinstance(value, bool) or text in ("1", "true", "yes", "on"):
        return (preset_layout or "relaxed") if parse_bool(value) else None
    raise BadRequest(f"sparse must be true, false or one of {', '.join(SPARSE_LAYOUTS)}")

log = logging.getLogger("h3-video-server")
generate_lock = threading.Lock()
jobs = {}                 # id -> job dict (in memory; also persisted as job.json)
jobs_lock = threading.Lock()
holds = {}                # id -> (hold arrived, render done) events of the async job


MEDIA_PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport"
content="width=device-width,initial-scale=1"><title>{title}</title><style>
:root{{--bg:#fafafa;--fg:#1b1b1f;--muted:#6b6b76;--card:#fff;--line:#e3e3e8;--accent:#2f6fde}}
@media (prefers-color-scheme:dark){{:root{{--bg:#141416;--fg:#ececf1;--muted:#9a9aa6;--card:#1d1d21;--line:#2c2c33;--accent:#6b9cff}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,system-ui,"PingFang SC",sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:24px 16px}}h1{{font-size:18px;margin:0 0 12px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}}
video,img{{width:100%;max-height:78vh;border-radius:8px;background:#000;display:block}}
.meta{{color:var(--muted);margin-top:12px}}a.btn{{display:inline-block;margin-top:12px;padding:8px 14px;border-radius:8px;
background:var(--accent);color:#fff;text-decoration:none}}progress{{width:100%;height:10px}}</style></head>
<body><main><h1>{title}</h1><div class="card">{body}</div></main>{script}</body></html>"""


class BadRequest(Exception):
    pass


def legal_frames(n):
    """H3 aligns frame counts upward to 5 + 17k."""
    n = max(5, int(n))
    return 5 + 17 * math.ceil((n - 5) / 17)


def parse_size(value, name="size", default=DEFAULT_SIZE):
    if value in (None, "", "auto"):
        return default
    try:
        w, h = (int(v) for v in str(value).lower().split("x"))
    except ValueError:
        raise BadRequest(f"{name} must be WIDTHxHEIGHT, got {value!r}")
    if not all(32 <= v and v % 32 == 0 for v in (w, h)) or w * h > MAX_AREA:
        raise BadRequest(f"{name}: width and height must be multiples of 32 with width*height <= 768*1344")
    return w, h


def parse_num(value, name, default, lo, hi, kind=int):
    if value in (None, ""):
        return default
    try:
        v = kind(value)
    except (TypeError, ValueError):
        raise BadRequest(f"{name} must be a number")
    if not lo <= v <= hi:
        raise BadRequest(f"{name} must be between {lo} and {hi}")
    return v


def parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def decode_media(value, name):
    """JSON media field (image, video or audio): base64, data: URL or an https URL on an allowed host -> bytes."""
    if isinstance(value, dict):  # {"image_url": "..."} / {"url": "..."}
        value = value.get("image_url") or value.get("url") or value.get("b64_json")
    if not isinstance(value, str) or not value:
        raise BadRequest(f"{name} must be a base64 string or data: URL")
    if value.startswith("data:"):
        value = value.split(",", 1)[-1]
    elif value.startswith(("http://", "https://")):
        return fetch_media(value, name)
    try:
        return base64.b64decode(value, validate=False)
    except (binascii.Error, ValueError):
        raise BadRequest(f"{name} is not valid base64")


decode_image = decode_media


def fetch_media(url, name):
    import urllib.parse
    import urllib.request
    u = urllib.parse.urlsplit(url)
    if u.scheme != "https" or u.hostname not in FETCH_HOSTS:
        raise BadRequest(f"{name}: only https URLs on {', '.join(sorted(FETCH_HOSTS)) or 'no host'} are fetched "
                         f"(got {u.scheme}://{u.hostname}); send base64 or a data: URL")
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "h3-video-server"}),
                                    timeout=30) as r:
            raw = r.read(FETCH_MAX + 1)
    except Exception as e:
        raise BadRequest(f"{name}: cannot fetch {u.hostname}{u.path}: {e}")
    if len(raw) > FETCH_MAX:
        raise BadRequest(f"{name}: file larger than {FETCH_MAX >> 20} MB")
    return raw


def parse_model(asked, default_mode):
    """Body "model" -> (DiT mode, preset or None, echoed model name)."""
    m = MODEL_RE.fullmatch(asked) if isinstance(asked, str) else None
    if not m:
        return default_mode, None, MODEL_BY_MODE.get(default_mode)
    return MODE_BY_MODEL[m.group(1)], m.group(2), asked


def sniff_ext(raw, name, kind="image"):
    """File extension from the magic bytes; kind = image, video or audio."""
    ext = None
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        ext = ".png"
    elif raw[:3] == b"\xff\xd8\xff":
        ext = ".jpg"
    elif raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        ext = ".webp"
    elif raw[4:8] == b"ftyp":
        ext = ".m4a" if raw[8:11] == b"M4A" else (".mov" if raw[8:10] == b"qt" else ".mp4")
    elif raw[:4] == b"\x1aE\xdf\xa3":
        ext = ".webm"
    elif raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        ext = ".wav"
    elif raw[:3] == b"ID3" or raw[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        ext = ".mp3"
    elif raw[:4] == b"fLaC":
        ext = ".flac"
    elif raw[:4] == b"OggS":
        ext = ".ogg"
    ok = {"image": (".png", ".jpg", ".webp"), "video": (".mp4", ".mov", ".webm"),
          "audio": (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".mp4")}[kind]
    if ext not in ok:
        raise BadRequest(f"{name} must be a {kind} file ({', '.join(e.strip('.') for e in ok)})")
    return ext


def media_seconds(path):
    """Duration of a media file via ffprobe (None if unreadable)."""
    r = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of",
                        "default=nw=1:nk=1", str(path)], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


def has_audio_stream(path):
    r = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "a", "-show_entries",
                        "stream=index", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return bool(r.stdout.strip())


def continuation_prompt(text, ctx_s, total_s):
    """Ref2VA prompt (official ref guide structure) for continuing <Video 1> with its soundtrack <Audio 1>. H3 renders
    the reference first and then goes on, so the target is context + new seconds long."""
    text = text.strip() or "The action continues naturally."
    return (
        "subject_definitions:\n"
        "<Video 1> is the source video; the target video continues directly from its final frame.\n"
        "<Audio 1> is the synchronized audio track of <Video 1>; its ambience, music and voices carry on into the "
        "target video.\n\n"
        "summary:\n"
        f"[video continuation + audio reference] The {total_s:.1f}-second target video first follows <Video 1> exactly for "
        f"{ctx_s:.1f} s, then continues it seamlessly: {text}\n\n"
        "retention_analysis:\n"
        "<Video 1>: continuation - the subjects, environment, lighting, color grading and camera style of its final "
        "moments are preserved.\n"
        "<Audio 1>: reference - the soundscape and any music continue without a break in timbre, tempo or level.\n\n"
        "detailed_description:\n"
        f"[Shot 1] From 00:00 to 00:{ctx_s:06.3f} the target video follows <Video 1> unchanged. From 00:{ctx_s:06.3f} "
        f"it continues in the same shot: {text}\n\n"
        "overall_soundscape:\n"
        "The soundscape of <Audio 1> continues seamlessly.\n\n"
        "non_diegetic_music:\n"
        "Any music from <Audio 1> continues in the same style and tempo; N/A if it has none.")


_share_key = []


def share_sig(job_id, exp):
    import hashlib
    import hmac
    if not _share_key:
        _share_key.append(hashlib.sha256(b"h3-video-share\0" + LSW_KEY_FILE.read_bytes().strip()).digest())
    return hmac.new(_share_key[0], f"{job_id}:{exp}".encode(), hashlib.sha256).hexdigest()[:32]


def share_url(job):
    exp = int(job.get("completed_at") or time.time()) + SHARE_TTL
    return f"{SHARE_BASE}/v1/videos/share/{job['id']}?exp={exp}&sig={share_sig(job['id'], exp)}"


def public(job):
    keys = ("id", "object", "model", "status", "progress", "phase", "created_at", "completed_at", "seconds",
            "frames", "size", "seed", "settings", "timings", "error", "content_path", "bytes", "extends",
            "segment_seconds", "segment_path")
    out = {k: job[k] for k in keys if k in job}
    if job.get("status") == "completed":
        try:
            out["share_url"] = share_url(job)
        except OSError:
            pass
    return out


def save_job(job):
    d = Path(job["dir"])
    (d / "job.json").write_text(json.dumps(public(job), indent=1))


def prune_outputs(out_dir):
    """Delete job folders older than KEEP_DAYS, then the oldest ones while the total exceeds KEEP_BYTES."""
    dirs = sorted((p for p in out_dir.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
    size = {p: sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) for p in dirs}
    total, cutoff = sum(size.values()), time.time() - KEEP_DAYS * 86400
    for p in dirs[:-1]:                     # never the newest (it may be the job being finished)
        if p.stat().st_mtime >= cutoff and total <= KEEP_BYTES:
            break
        shutil.rmtree(p, ignore_errors=True)
        total -= size[p]
        with jobs_lock:
            jobs.pop(p.name, None)


def load_job(out_dir, job_id):
    if not re.fullmatch(r"video_[0-9a-f]{24}", job_id):
        return None
    with jobs_lock:
        if job_id in jobs:
            return jobs[job_id]
    f = out_dir / job_id / "job.json"
    if f.is_file():
        job = json.loads(f.read_text())
        job["dir"] = str(out_dir / job_id)
        return job
    return None


class Render:
    """One h3 CLI run plus a watchdog that kills it if this wrapper dies (even by SIGKILL)."""
    current = None

    def __init__(self, job, argv, env):
        self.job, self.argv, self.env = job, argv, env
        self.proc = None
        self.cancelled = False

    def run(self, on_progress):
        job = self.job
        logf = open(Path(job["dir"]) / "h3.log", "wb")
        self.proc = subprocess.Popen(self.argv, cwd=H3_DIR, env=self.env, stdin=subprocess.DEVNULL,
                                     stdout=logf, stderr=subprocess.PIPE)
        Render.current = self
        subprocess.Popen(["/bin/sh", "-c", f"while kill -0 {os.getpid()} 2>/dev/null && kill -0 {self.proc.pid} "
                          f"2>/dev/null; do sleep 1; done; kill -TERM {self.proc.pid} 2>/dev/null"],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
        phase, phase_t0, timings = None, time.time(), {}
        buf = b""
        tail = []
        try:
            while True:
                chunk = self.proc.stderr.read1(65536)
                if not chunk:
                    break
                logf.write(chunk)
                logf.flush()
                buf += chunk
                parts = re.split(rb"[\r\n]", buf)
                buf = parts.pop()
                for raw in parts:
                    line = raw.decode(errors="replace").rstrip()
                    if not line:
                        continue
                    tail = (tail + [line])[-30:]
                    m = re.match(r"^(\S.*?)\s+(\d+)/(\d+)\s*$", line)
                    if m and not line.startswith("h3:"):
                        name, done, total = m.group(1).strip(), int(m.group(2)), int(m.group(3))
                        if name != phase:
                            now = time.time()
                            if phase:
                                timings[phase] = round(timings.get(phase, 0) + now - phase_t0, 2)
                            phase, phase_t0 = name, now
                        on_progress(name, done, total)
            rc = self.proc.wait()
        finally:
            Render.current = None
            logf.close()
        if phase:
            timings[phase] = round(timings.get(phase, 0) + time.time() - phase_t0, 2)
        return rc, timings, tail

    def cancel(self):
        self.cancelled = True
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def server_bind(self):
        # HTTPServer.server_bind calls socket.getfqdn(), which blocks ~35 s on this Mac with
        # Homebrew Python 3.15; the server name is unused, so skip the reverse lookup.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


class Handler(BaseHTTPRequestHandler):
    server_version = "h3-video-server"
    protocol_version = "HTTP/1.1"

    # ---- plumbing -------------------------------------------------------------------------------
    def _send_json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, message, kind="invalid_request_error"):
        self._send_json(status, {"error": {"message": message, "type": kind}})

    def _send_file(self, path, ctype="video/mp4"):
        size = path.stat().st_size
        start, end = 0, size - 1
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", "").strip())
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start, end = int(m.group(1)), min(int(m.group(2) or size - 1), size - 1)
            else:                                   # suffix range: the last N bytes
                start = max(size - int(m.group(2)), 0)
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                return self.end_headers()
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        fname = path.name if path.suffix != ".mp4" or path.stem != "video" else f"{path.parent.name}.mp4"
        self.send_header("Content-Disposition", f'inline; filename="{fname}"')
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)

    def _chunk(self, data):
        self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
        self.wfile.flush()

    def log_message(self, fmt, *args):
        log.info("%s %s", self.address_string(), fmt % args)

    # ---- routes ---------------------------------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        out = self.server.out_dir
        if path == "/health":
            return self._send_json(200, {"status": "ok"})
        if path == "/v1/models":
            return self._send_json(200, {"object": "list", "data": [
                {"id": self.server.model_id, "object": "model", "owned_by": "local"}]})
        if path.startswith("/media/"):
            return self._media(path[len("/media/"):])
        if path == "/v1/videos":
            ids = sorted((p.name for p in out.iterdir() if p.is_dir()),
                         key=lambda n: (out / n).stat().st_mtime, reverse=True)
            data = [public(j) for j in (load_job(out, i) for i in ids) if j]
            return self._send_json(200, {"object": "list", "data": data})
        m = re.fullmatch(r"/v1/videos/share/(video_[0-9a-f]{24})", path)
        if m:
            import hmac
            import urllib.parse
            q = urllib.parse.parse_qs(self.path.partition("?")[2])
            exp, sig = (q.get("exp") or [""])[0], (q.get("sig") or [""])[0]
            if not exp.isdigit() or int(exp) < time.time() or not hmac.compare_digest(sig, share_sig(m.group(1), int(exp))):
                return self._error(403, "invalid or expired link")
            job = load_job(out, m.group(1))
            mp4 = Path(job["dir"]) / "video.mp4" if job else None
            if not job or job.get("status") != "completed" or not mp4.is_file():
                return self._error(404, "no such video")
            return self._send_file(mp4)
        m = re.fullmatch(r"/v1/videos/([^/]+)/hold", path)
        if m:
            return self._hold(m.group(1))
        m = re.fullmatch(r"/v1/videos/([^/]+)(/content)?", path)
        if m:
            job = load_job(out, m.group(1))
            if not job:
                return self._error(404, f"no video {m.group(1)}")
            if not m.group(2):
                return self._send_json(200, public(job))
            part = "segment.mp4" if job.get("extends") and "part=segment" in self.path.partition("?")[2] else "video.mp4"
            mp4 = Path(job["dir"]) / part
            if job.get("status") != "completed" or not mp4.is_file():
                return self._error(409, f"video {job['id']} is {job.get('status')}")
            return self._send_file(mp4)
        self._error(404, f"unknown path {self.path}")

    def _html(self, title, body, script=""):
        page = MEDIA_PAGE.format(title=title, body=body, script=script).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.end_headers()
        self.wfile.write(page)

    def _media(self, name):
        """Watch/download pages: /media/<id> (player, or progress until done), /media/<id>.mp4, /media/<id>/status."""
        out = self.server.out_dir
        im = re.fullmatch(r"(img_[0-9a-f]{16})(\.png)?", name)
        if im:       # images kept by the H3 image wrapper (same media dir convention, same retention)
            png = MEDIA_IMAGES / f"{im.group(1)}.png"
            if not png.is_file():
                return self._error(404, "no such image")
            if im.group(2):
                return self._send_file(png, "image/png")
            return self._html("图片", f'<img src="/media/{im.group(1)}.png" alt="{im.group(1)}"><div class="meta">{im.group(1)}'
                                      f'</div><a class="btn" href="/media/{im.group(1)}.png" download>下载 PNG</a>')
        m = re.fullmatch(r"(video_[0-9a-f]{24})(\.mp4|/status)?", name)
        job = load_job(out, m.group(1)) if m else None
        if not job:
            return self._error(404, "no such video")
        vid, kind = m.group(1), m.group(2)
        if kind == "/status":
            return self._send_json(200, {k: job.get(k) for k in ("id", "status", "progress", "phase", "seconds", "size")})
        mp4 = Path(job["dir"]) / "video.mp4"
        if kind == ".mp4":
            if job.get("status") != "completed" or not mp4.is_file():
                return self._error(409, f"video {vid} is {job.get('status')}")
            return self._send_file(mp4)
        if job.get("status") == "completed":
            st = job.get("settings") or {}
            info = (f"{job['size']} · {job['seconds']} 秒" + (f"（续写新增 {job.get('segment_seconds')} 秒）" if job.get("extends") else "")
                    + f" · {st.get('preset')} · seed {job.get('seed')} · 渲染 {(job.get('timings') or {}).get('render_s')} 秒 · {vid}")
            return self._html("视频", f'<video controls playsinline src="/media/{vid}.mp4"></video><div class="meta">{info}</div>'
                                      f'<a class="btn" href="/media/{vid}.mp4" download>下载 MP4</a>')
        if job.get("status") == "failed":
            return self._html("视频", f"渲染失败：{(job.get('error') or {}).get('message', '')}")
        body = (f'<p id="s">渲染中 {job.get("progress", 0)}%</p><progress id="p" max="100" value="{job.get("progress", 0)}">'
                f'</progress><div class="meta">{vid}：完成后本页自动显示视频</div>')
        script = ("<script>setInterval(async()=>{try{const d=await (await fetch(location.pathname+'/status')).json();"
                  "if(d.status==='completed'||d.status==='failed')location.reload();document.getElementById('p').value=d.progress||0;"
                  "document.getElementById('s').textContent='渲染中 '+(d.progress||0)+'%'+(d.phase?'（'+d.phase+'）':'')}catch(e){}},5000)"
                  "</script>")
        return self._html("视频", body, script)

    def do_POST(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        if path != "/v1/videos":
            return self._error(404, f"unknown path {self.path}")
        if self.server.files_only:
            return self._error(405, "this instance only serves finished videos (GET); POST /v1/videos renders on "
                                    "minimax-h3-video", "invalid_request_error")
        # one render at a time and no queue: a second POST gets 429 at once, so every request's wait is bounded by one
        # render (the public proxy timeout is sized to the worst render, not to a queue)
        if not generate_lock.acquire(blocking=False):
            self.send_response(429)
            self.send_header("Retry-After", "60")
            body = json.dumps({"error": {"message": "a video is being rendered; retry later", "type": "rate_limit_error",
                                         "code": "busy"}}).encode()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        handed_over = False   # an async render releases generate_lock itself when it ends
        try:
            try:
                job, argv, env, response_format, wait = self._prepare()
            except BadRequest as e:
                return self._error(400, str(e))
            except Exception as e:
                log.exception("bad request")
                return self._error(500, f"cannot prepare job: {e}", "server_error")
            if wait:
                self._run(job, argv, env, response_format)
            else:
                self._start_async(job, argv, env)
                handed_over = True
        finally:
            if not handed_over:
                generate_lock.release()

    def _hold(self, job_id):
        """Stays open while the async render of job_id runs, so llama-swap sees a request in flight."""
        ev = None if self.server.files_only else holds.get(job_id)
        if not ev:
            return self._error(404, f"no async render {job_id} running here")
        arrived, done = ev
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        self.wfile.flush()
        arrived.set()
        try:
            while not done.wait(KEEPALIVE_S):
                self._chunk(b" ")
            self._chunk(b"done\n")
            self._chunk(b"")
        except OSError:
            log.warning("hold for %s dropped before the render ended", job_id)
            self.close_connection = True

    # ---- request parsing ------------------------------------------------------------------------
    def _read_request(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length > 200 << 20:
            raise BadRequest("request body too large (max 200 MB)")
        body = self.rfile.read(length)
        ctype = self.headers.get("Content-Type", "")
        fields, files = {}, {}
        if ctype.startswith("multipart/form-data"):
            msg = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
                f"Content-Type: {ctype}\r\n\r\n".encode() + body)
            for part in msg.iter_parts():
                name = part.get_param("name", header="content-disposition")
                if not name:
                    continue
                if part.get_filename() is not None or part.get_content_maintype() != "text":
                    name, raw = name.removesuffix("[]"), part.get_payload(decode=True)
                    if name == "reference_videos":
                        files.setdefault("reference_videos", []).append((raw, True))
                    elif name == "reference_silent_videos":
                        files.setdefault("reference_videos", []).append((raw, False))
                    else:
                        files.setdefault(name, []).append(raw)
                else:
                    fields[name] = part.get_content().strip()
        else:
            try:
                fields = json.loads(body or b"{}")
            except json.JSONDecodeError:
                raise BadRequest("body must be JSON or multipart/form-data")
            if not isinstance(fields, dict):
                raise BadRequest("JSON body must be an object")
            for key in ("first_frame", "last_frame", "input_reference"):
                if fields.get(key):
                    files[key] = [decode_image(fields.pop(key), key)]
            refs = fields.pop("reference_images", None)
            if refs:
                if not isinstance(refs, list):
                    refs = [refs]
                files["reference_images"] = [decode_image(r, "reference_images") for r in refs]
            vids = fields.pop("reference_videos", None)
            if vids:
                out = []
                for v in vids if isinstance(vids, list) else [vids]:
                    if isinstance(v, dict) and "video" in v:   # {"video": ..., "audio": true | false | <audio>}
                        audio = v.get("audio", True)
                        if not isinstance(audio, bool):
                            audio = decode_media(audio, "reference_videos[].audio")
                        out.append((decode_media(v["video"], "reference_videos"), audio))
                    else:
                        out.append((decode_media(v, "reference_videos"), True))
                files["reference_videos"] = out
            auds = fields.pop("reference_audio", None)
            if auds:
                files["reference_audio"] = [decode_media(a, "reference_audio")
                                            for a in (auds if isinstance(auds, list) else [auds])]
            if fields.get("video"):
                files["video"] = [decode_media(fields.pop("video"), "video")]
        return fields, files

    def _prepare(self):
        fields, files = self._read_request()
        prompt = fields.get("prompt")
        prev = None
        if fields.get("extend_from"):
            prev = load_job(self.server.out_dir, str(fields["extend_from"]))
            if not prev or prev.get("status") != "completed" or not (Path(prev["dir"]) / "video.mp4").is_file():
                raise BadRequest(f"extend_from: {fields['extend_from']} is not a completed video on this server")
            if fields.get("size") not in (None, "") and parse_size(fields.get("size")) != parse_size(prev["size"]):
                raise BadRequest(f"extend_from: a continuation keeps the size of the video ({prev['size']})")
            fields["size"] = prev["size"]
            prompt = prompt if isinstance(prompt, str) and prompt.strip() else "The action continues naturally."
        job_id, d = "video_" + secrets.token_hex(12), None
        upload = (files.get("video") or [None])[0]
        if upload is not None:   # the user's own video is the start: normalize it, then continue it like extend_from
            if prev:
                raise BadRequest("send either extend_from or video, not both")
            d = self.server.out_dir / job_id
            d.mkdir(parents=True)
            raw = d / ("upload" + sniff_ext(upload, "video", "video"))
            raw.write_bytes(upload)
            size = parse_size(fields.get("size")) if fields.get("size") not in (None, "") else None
            src, size = normalize_upload(raw, d / "source.mp4", size)
            prev = {"id": "upload", "dir": str(d), "size": f"{size[0]}x{size[1]}", "status": "completed",
                    "parts": [[str(src), 0.0, None]], "full": str(src)}
            fields["size"] = prev["size"]
            prompt = prompt if isinstance(prompt, str) and prompt.strip() else "The action continues naturally."
        if not isinstance(prompt, str) or not prompt.strip():
            raise BadRequest("prompt is required")
        width, height = parse_size(fields.get("size"))
        if fields.get("frames") not in (None, ""):
            frames = legal_frames(parse_num(fields.get("frames"), "frames", None, 5, MAX_FRAMES))
        else:
            secs = parse_num(fields.get("seconds"), "seconds", DEFAULT_SECONDS, 0.2, 15.1, float)
            frames = legal_frames(math.ceil(secs * FPS - 1e-9))
        ctx_frames = 0
        if prev:   # Ref2VA continuation re-creates <Video 1> first, then goes on: render context + new seconds
            ctx_s = parse_num(fields.get("extend_context_seconds"), "extend_context_seconds", DEFAULT_CONTEXT, 2.0, 15.0, float)
            ctx_frames = round(min(ctx_s, parts_seconds(job_parts(prev))) * FPS)
            new_frames = (frames if fields.get("frames") not in (None, "") else
                          math.ceil(parse_num(fields.get("seconds"), "seconds", DEFAULT_SECONDS, 0.2, 15.1, float) * FPS - 1e-9))
            frames = legal_frames(ctx_frames + new_frames)   # round once, after adding the context
            if frames > MAX_FRAMES:
                raise BadRequest(f"with {ctx_frames / FPS:.1f} s of context a continuation adds at most "
                                 f"{(MAX_FRAMES - ctx_frames) / FPS:.1f} s (lower seconds or extend_context_seconds)")
            if "subject_definitions:" not in prompt:
                prompt = continuation_prompt(prompt, ctx_frames / FPS, frames / FPS)
        if frames > MAX_FRAMES:
            raise BadRequest(f"at most 15 seconds ({MAX_FRAMES} frames)")
        mode, model_preset, model_name = parse_model(fields.get("model"), self.server.mode)
        preset_name = str(fields.get("preset") or model_preset or DEFAULT_PRESET)
        preset_name = PRESET_ALIASES.get(preset_name, preset_name)
        if preset_name not in PRESETS:
            raise BadRequest(f"preset must be one of {', '.join(PRESETS)}")
        requested_preset, fallback = preset_name, None
        turbo_dir = Path(TURBO_MODELS or Path(self.server.models).parent / "minimax-h3-turbo")
        if preset_name == "turbo":
            if prev or any(files.get(k) for k in ("reference_images", "reference_videos", "reference_audio")):
                fallback = ("turbo covers text and first/last-frame videos only (its LoRA is trained on FL2VA); "
                            "references and continuations use standard")
            elif frames > legal_frames(math.ceil(TURBO_MAX_SECONDS * FPS - 1e-9)):
                fallback = f"turbo is limited to {TURBO_MAX_SECONDS:g} s; longer videos use standard"
            elif not (turbo_dir / "FL2VA/transformer-int8/h3-int8-pack.json").is_file():
                fallback = "the turbo weights are not installed; standard is used"
            if fallback:
                preset_name = "standard"
        models_dir = str(turbo_dir) if preset_name == "turbo" else self.server.models
        p = dict(PRESETS[preset_name])
        p["steps"] = parse_num(fields.get("steps"), "steps", p["steps"], 1, MAX_STEPS)
        if fields.get("steps") not in (None, "") and p["steps"] < 8 and fields.get("reuse") in (None, ""):
            p["reuse"] = 1   # README: keep reuse 1 at very small step counts
        p["reuse"] = parse_num(fields.get("reuse"), "reuse", p["reuse"], 1, 3)
        p["layers"] = parse_num(fields.get("layers"), "layers", p["layers"], 35, 50)
        core_reuse = parse_num(fields.get("core_reuse"), "core_reuse", 1, 1, 6)
        if core_reuse > 1:
            p["core_reuse"], p["reuse"] = core_reuse, 1
        if fields.get("token_reduction") not in (None, ""):
            p["token_reduction"] = parse_bool(fields.get("token_reduction"))
        if fields.get("sparse") not in (None, ""):
            p["sparse"] = parse_sparse(fields.get("sparse"), p.get("sparse"))
        render = None
        if fields.get("render_size"):
            render = parse_size(fields.get("render_size"), "render_size")
            if render[0] > width or render[1] > height or render[0] * height != render[1] * width:
                raise BadRequest("render_size must keep the output aspect ratio and not exceed size")
            p["render_size"] = f"{render[0]}x{render[1]}"
        seed = parse_num(fields.get("seed"), "seed", None, 0, 2**63 - 1)
        if seed is None:
            seed = secrets.randbelow(2**31)

        response_format = str(fields.get("response_format") or "").lower()
        if not response_format:
            response_format = "mp4" if "video/mp4" in self.headers.get("Accept", "") else "json"
        if response_format not in ("json", "b64_json", "mp4"):
            raise BadRequest("response_format must be json, b64_json or mp4")
        wait = parse_bool(fields.get("wait", False)) or response_format in ("b64_json", "mp4")

        first = (files.get("first_frame") or files.get("input_reference") or [None])[0]
        last = (files.get("last_frame") or [None])[0]
        refs = files.get("reference_images") or []
        vids = files.get("reference_videos") or []
        auds = files.get("reference_audio") or []
        ref2va = bool(refs or vids or auds or prev)
        if ref2va and (first or last):
            raise BadRequest("Ref2VA references (reference_*, extend_from) cannot be combined with first_frame/last_frame")
        if ref2va and not self.server.has_ref2va:
            raise BadRequest("references need the Ref2VA checkpoint, which is not installed on this server")
        if len(refs) > 9 or len(vids) + (1 if prev else 0) > 3 or len(auds) > 3:
            raise BadRequest("Ref2VA limits: 9 images, 3 videos (extend_from counts as one), 3 audio clips")

        if d is None:
            d = self.server.out_dir / job_id
            d.mkdir(parents=True)
        argv = [str(H3_BIN), "--profile", "-d", models_dir, "-p", prompt,
                "--width", str(width), "--height", str(height), "--frames", str(frames),
                "--steps", str(p["steps"]), "--layers", str(p["layers"]), "--seed", str(seed),
                "-o", str(d / ("segment.mp4" if prev else "video.mp4"))]
        argv += ["--core-reuse", str(p["core_reuse"])] if "core_reuse" in p else ["--reuse", str(p["reuse"])]
        if p.get("token_reduction"):
            argv.append("--token-reduction")
        if render:
            argv += ["--render-width", str(render[0]), "--render-height", str(render[1])]
        if mode == "stream":
            argv.append("--ssd-streaming")
        for key, raw in (("first-frame", first), ("last-frame", last)):
            if raw:
                f = d / (key + sniff_ext(raw, key))
                f.write_bytes(raw)
                argv += ["--" + key, str(f)]
        for i, raw in enumerate(refs):
            f = d / f"ref{i + 1}{sniff_ext(raw, 'reference_images')}"
            f.write_bytes(raw)
            argv += ["--ref-image", str(f)]
        vid_paths = []          # (path, audio) in <Video N> order; an extended video's tail is <Video 1>
        if prev:
            full = Path(prev.get("full") or Path(prev["dir"]) / "video.mp4")
            tail = d / "context.mp4"
            r = subprocess.run([FFMPEG, "-v", "error", "-y", "-sseof", f"-{ctx_frames / FPS:.4f}", "-i", str(full),
                                "-frames:v", str(ctx_frames), "-c:v", "libx264", "-crf", "12", "-preset", "veryfast",
                                "-c:a", "aac", "-b:a", "192k", str(tail)], capture_output=True, text=True)
            if r.returncode != 0 or not tail.is_file():
                raise BadRequest(f"extend_from: cannot cut the last {ctx_frames / FPS:.1f} s: {r.stderr.strip()[-200:]}")
            vid_paths.append((tail, True))
        for i, (raw, audio) in enumerate(vids):
            f = d / f"refvideo{i + 1}{sniff_ext(raw, 'reference_videos', 'video')}"
            f.write_bytes(raw)
            if not isinstance(audio, bool):
                a = d / f"refvideo{i + 1}-audio{sniff_ext(audio, 'reference_videos[].audio', 'audio')}"
                a.write_bytes(audio)
                audio = a
            vid_paths.append((f, audio))
        total = 0.0
        for f, audio in vid_paths:
            secs = media_seconds(f)
            if secs is None or secs < 1.95 or secs > 15.1:
                raise BadRequest(f"each reference video must be 2-15 s long ({f.name}: {secs})")
            total += secs
            if audio is True and not has_audio_stream(f):
                audio = False
            argv += (["--ref-video", str(f)] if audio is True else ["--ref-silent-video", str(f)] if audio is False
                     else ["--ref-video-audio", str(f), str(audio)])
        if total > 15.1:
            raise BadRequest(f"reference videos may total at most 15 s (got {total:.1f} s)")
        for i, raw in enumerate(auds):
            f = d / f"refaudio{i + 1}{sniff_ext(raw, 'reference_audio', 'audio')}"
            f.write_bytes(raw)
            argv += ["--ref-audio", str(f)]
        env = dict(os.environ, **H3_ENV)
        if p.get("sparse"):
            env.update(sparse_env(p["sparse"], p["steps"]))
        job = {"id": job_id, "object": "video", "model": model_name or self.server.model_id, "status": "queued", "progress": 0,
               "created_at": int(time.time()), "seconds": round(frames / FPS, 3), "frames": frames,
               "size": f"{width}x{height}", "seed": seed,
               "settings": dict(p, preset=preset_name, mode=mode,
                                **({"requested_preset": requested_preset, "fallback": fallback} if fallback else {}),
                                conditioning="extension" if prev else "ref2va" if ref2va else
                                ("fl2va" if first or last else "t2va")),
               "dir": str(d), "content_path": f"/v1/videos/{job_id}/content"}
        if prev:   # earlier parts minus the re-created context, then the whole new render (finish -> build_video)
            parts = trim_parts_end(job_parts(prev), ctx_frames / FPS) + [[str(d / "segment.mp4"), 0.0, None]]
            (d / "parts.json").write_text(json.dumps(parts))
            job.update(extends=prev["id"], segment_seconds=round((frames - ctx_frames) / FPS, 3),
                       seconds=round(parts_seconds(parts[:-1]) + frames / FPS, 3),
                       segment_path=f"/v1/videos/{job_id}/content?part=segment", parts=parts)
            job["settings"]["context_seconds"] = round(ctx_frames / FPS, 3)
        with jobs_lock:
            jobs[job_id] = job
        save_job(job)
        return job, argv, env, response_format, wait

    # ---- execution ------------------------------------------------------------------------------
    def _start_async(self, job, argv, env):
        """Answers the POST with the queued job; the render runs in a thread that owns generate_lock."""
        arrived, done = threading.Event(), threading.Event()
        holds[job["id"]] = (arrived, done)
        out_dir = self.server.out_dir

        def work():
            t0 = time.time()
            try:
                execute(job, Render(job, argv, env), t0)
            finally:
                finish(job, out_dir, t0)
                done.set()
                holds.pop(job["id"], None)
                generate_lock.release()
        threading.Thread(target=hold_open, args=(self.server.model_id, job["id"]), daemon=True).start()
        if not arrived.wait(HOLD_WAIT_S):
            log.warning("hold for %s did not arrive within %d s; llama-swap may unload this model mid-render",
                        job["id"], HOLD_WAIT_S)
        threading.Thread(target=work, daemon=True).start()
        self._send_json(200, public(job))

    def _run(self, job, argv, env, response_format):
        streaming = response_format in ("json", "b64_json")
        disconnected = threading.Event()
        done = threading.Event()
        render = Render(job, argv, env)
        if streaming:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("X-Video-Id", job["id"])
            self.end_headers()
            self.wfile.flush()

            def keepalive():
                while not done.wait(KEEPALIVE_S):
                    try:
                        self._chunk(b" ")
                    except OSError:
                        disconnected.set()
                        render.cancel()
                        return
            threading.Thread(target=keepalive, daemon=True).start()
        else:
            def watch_socket():  # detect a client that hangs up while we render (no bytes are sent yet)
                import select
                while not done.wait(5):
                    try:
                        r, _, _ = select.select([self.connection], [], [], 0)
                        if r and not self.connection.recv(1, socket_peek()):
                            disconnected.set()
                            render.cancel()
                            return
                    except OSError:
                        return
            threading.Thread(target=watch_socket, daemon=True).start()

        t0 = time.time()
        execute(job, render, t0, disconnected)   # generate_lock is held by do_POST for the whole request
        done.set()
        finish(job, self.server.out_dir, t0)
        if disconnected.is_set():
            self.close_connection = True
            return
        mp4 = Path(job["dir"]) / "video.mp4"
        try:
            if streaming:
                payload = public(job)
                if response_format == "b64_json" and job["status"] == "completed":
                    payload["b64_json"] = base64.b64encode(mp4.read_bytes()).decode()
                self._chunk(json.dumps(payload).encode())
                self._chunk(b"")
            elif job["status"] == "completed":
                self._send_file(mp4)
            else:
                self._send_json(500, {"error": {"message": job["error"]["message"], "type": "server_error",
                                                "code": job["error"]["code"]}, "video": public(job)})
        except OSError:
            self.close_connection = True


def execute(job, render, t0, disconnected=None):
    """Runs one render and records the outcome in job; the caller holds generate_lock."""
    if disconnected is not None and disconnected.is_set():
        job.update(status="failed", error={"code": "cancelled", "message": "client disconnected"})
        return
    job["status"] = "in_progress"
    t_start = time.time()
    save_job(job)
    log.info("render %s %s %d frames %s", job["id"], job["size"], job["frames"], job["settings"])
    last = [None]

    def on_progress(name, n, total):
        job["phase"] = f"{name} {n}/{total}"
        if name.startswith("denoise") and total:
            job["progress"] = int(100 * n / total)
        if (name, job["progress"]) != last[0]:   # the read-only instance reads progress from job.json
            last[0] = (name, job["progress"])
            try:
                save_job(job)
            except OSError:
                pass
    try:
        rc, timings, tail = render.run(on_progress)
    except Exception as e:  # spawn failure etc.
        rc, timings, tail = -1, {}, [repr(e)]
    job["timings"] = dict(timings, queued_s=round(t_start - t0, 2), render_s=round(time.time() - t_start, 2))
    mp4 = Path(job["dir"]) / "video.mp4"
    if rc == 0 and job.get("parts"):
        job["phase"] = "joining"
        if not build_video(job["parts"], mp4):
            rc, tail = -2, ["h3: rendered the continuation, but joining it to the earlier video failed"]
    if rc == 0 and mp4.is_file():
        job.update(status="completed", progress=100, bytes=mp4.stat().st_size)
    else:
        msg = "client disconnected" if render.cancelled else \
            next((l for l in reversed(tail) if l.startswith("h3:")), f"h3 exited with {rc}")
        job.update(status="failed", error={"code": "cancelled" if render.cancelled else "render_failed",
                                           "message": msg})


def job_parts(job):
    """[[path, start_s, end_s|None], ...] that make up a finished video (parts.json; plain renders are one part)."""
    whole = [[str(Path(job["dir"]) / "video.mp4"), 0.0, None]]
    if job.get("parts"):
        return job["parts"]
    f = Path(job["dir"]) / "parts.json"
    parts = json.loads(f.read_text()) if f.is_file() else whole
    # an earlier part pruned away (retention): continue from the finished video itself (one more encode)
    return parts if all(Path(p[0]).is_file() for p in parts) else whole


def part_seconds(part):
    path, start, end = part
    return (end if end is not None else media_seconds(path) or 0.0) - start


def parts_seconds(parts):
    return sum(part_seconds(p) for p in parts)


def trim_parts_end(parts, seconds):
    """Drop the last `seconds` of the parts' timeline."""
    parts = [list(p) for p in parts]
    while parts and seconds > 1e-6:
        length = part_seconds(parts[-1])
        if length <= seconds + 1e-6:
            seconds -= length
            parts.pop()
        else:
            parts[-1][2] = parts[-1][1] + length - seconds
            seconds = 0
    return parts


def fit_size(w, h):
    """Largest supported size (multiples of 32, area <= 1344x768) with about the aspect ratio of w x h, not larger
    than the source (small uploads are not upscaled; at least 256 on the short side)."""
    cap = min(MAX_AREA, max(w * h, 256 * 256))
    best = None
    for hh in range(256, 1345, 32):
        ww = round(w / h * hh / 32) * 32
        if 256 <= ww <= 1344 and ww * hh <= cap and (best is None or ww * hh > best[0] * best[1]):
            best = (ww, hh)
    return best or (512, 512)


def normalize_upload(src, out, size=None):
    """An uploaded starting video -> 24 fps H.264 + 32 kHz stereo AAC (silence if it has no sound) at a supported size:
    its own aspect ratio by default, else `size` (scaled to cover, then centre-cropped)."""
    r = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,duration:format=duration", "-of", "json", str(src)],
                       capture_output=True, text=True)
    try:
        info = json.loads(r.stdout)
        w, h = int(info["streams"][0]["width"]), int(info["streams"][0]["height"])
        dur = float(info["streams"][0].get("duration") or info["format"]["duration"])   # the picture's length
    except (ValueError, KeyError, IndexError, TypeError):
        raise BadRequest("video: not a readable video file")
    if not 2.0 <= dur <= 600:
        raise BadRequest(f"video: must be 2 s to 10 min long (got {dur:.1f} s)")
    W, H = size or fit_size(w, h)
    audio = has_audio_stream(src)
    args = [FFMPEG, "-v", "error", "-y", "-i", str(src)]
    if not audio:
        args += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=32000"]
    args += ["-map", "0:v:0", "-map", "0:a:0" if audio else "1:a:0",
             "-vf", f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={FPS},setsar=1"]
    if audio:
        args += ["-af", "apad"]            # a shorter soundtrack is padded with silence ...
    # ... and everything is cut to the picture's length (-shortest does not stop apad, and a longer soundtrack
    # would otherwise leave seconds without picture at the end, where the continuation context is cut)
    args += ["-t", f"{dur:.3f}", "-c:v", "libx264", "-crf", "14", "-preset", "fast", "-pix_fmt", "yuv420p", "-c:a", "aac",
             "-b:a", "192k", "-ar", "32000", "-ac", "2", "-movflags", "+faststart", str(out)]
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0 or not out.is_file():
        raise BadRequest(f"video: cannot convert it: {r.stderr.strip()[-200:]}")
    return out, (W, H)


def build_video(parts, out):
    """Encode the parts (h3 renders or normalized uploads, trimmed) into one MP4. Every source is encoded only here,
    so chained extensions do not pile up re-encoding losses."""
    args, chains = [FFMPEG, "-v", "error", "-y"], []
    for i, (path, start, end) in enumerate(parts):
        args += ["-i", path]
        rng = f"start={start:.4f}" + (f":end={end:.4f}" if end is not None else "")
        chains.append(f"[{i}:v]trim={rng},setpts=PTS-STARTPTS[v{i}];[{i}:a]atrim={rng},asetpts=PTS-STARTPTS[a{i}]")
    graph = ";".join(chains) + ";" + "".join(f"[v{i}][a{i}]" for i in range(len(parts))) + \
        f"concat=n={len(parts)}:v=1:a=1[v][a]"
    tmp = out.with_suffix(".tmp.mp4")
    r = subprocess.run(args + ["-filter_complex", graph, "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "16",
                               "-preset", "fast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "32000",
                               "-movflags", "+faststart", str(tmp)], capture_output=True, text=True)
    if r.returncode != 0:
        log.warning("joining parts failed: %s", r.stderr.strip()[-300:])
        return False
    tmp.replace(out)
    return True


def finish(job, out_dir, t0):
    job["completed_at"] = int(time.time())
    job.pop("phase", None)
    save_job(job)
    log.info("render %s %s in %.1fs %s", job["id"], job["status"], time.time() - t0, job.get("timings"))
    try:
        prune_outputs(out_dir)
    except OSError:
        pass


def hold_open(model_id, job_id):
    """The async render's in-flight request: GET .../hold through llama-swap, open until the render ends."""
    import http.client
    import urllib.parse
    u = urllib.parse.urlsplit(LSW_URL)
    try:
        conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=None)
        conn.request("GET", f"/upstream/{model_id}/v1/videos/{job_id}/hold",
                     headers={"Authorization": "Bearer " + LSW_KEY_FILE.read_text().strip()})
        r = conn.getresponse()
        r.read()
        if r.status != 200:
            log.warning("hold for %s: llama-swap answered %d", job_id, r.status)
    except Exception as e:
        log.warning("hold for %s failed: %r (llama-swap may unload this model mid-render)", job_id, e)


def socket_peek():
    import socket
    return socket.MSG_PEEK


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--models", default=str(Path.home() / "models/minimax-h3"))
    parser.add_argument("--model-id", default="minimax-h3-video")
    parser.add_argument("--mode", choices=("resident", "stream"), default="resident",
                        help="resident: int8 DiT in memory (~40 GB peak); stream: BF16 DiT blocks read from SSD "
                             "per step (--ssd-streaming, small DiT footprint, slower)")
    parser.add_argument("--out-dir", default=str(HERE / "outputs"))
    parser.add_argument("--files-only", action="store_true", help="serve finished videos (GET) only; refuse POST")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[h3-video] %(message)s")
    if not H3_BIN.is_file():
        sys.exit(f"missing {H3_BIN}; run make in {H3_DIR}")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    server = Server((args.host, args.port), Handler)
    server.models, server.model_id, server.mode, server.out_dir = args.models, args.model_id, args.mode, out
    server.files_only = args.files_only
    server.has_ref2va = (Path(args.models) / "Ref2VA/transformer/model.safetensors.index.json").is_file()

    def stop(signum, _frame):
        if Render.current:
            Render.current.cancel()
        os._exit(0)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    log.info("listening on http://%s:%d (models: %s, mode: %s, ref2va: %s)", args.host, args.port, args.models,
             args.mode, server.has_ref2va)
    server.serve_forever()


if __name__ == "__main__":
    main()
