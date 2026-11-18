"""llm-mcp: the local models as MCP tools, plus media pages. Runs anywhere that reaches the gateway (e.g. docker).

  /mcp              MCP (streamable HTTP). Authorization: Bearer <token>, tokens from MCP_TOKENS (one per client,
                    comma-separated, so one can be revoked alone). The llama-swap key stays here.
  /healthz
Media (videos, images) is stored and served by the Mac Studio itself (its gateway: /v1/videos, /media/<id>); the
links handed out are PUBLIC/media/<id> (behind an optional authenticating proxy). Nothing is cached here.

Tools: list_models, chat, web_search, fetch_url, generate_image, edit_image, generate_video, extend_video, get_video,
transcribe, speak, embed, rerank, decide, list_attachments, inspect_file, server_status.
- Models come from llama-swap's /v1/models (refreshed every minute, classified by its modalities); a "model" argument
  is a free string with a default, checked against the live list (an unknown id gets the list back), so a changed
  llama-swap config needs no restart here and no client has to re-fetch tool schemas.
- Attachments: Open WebUI sends X-Chat-Id / X-User-Id / X-User-Message-Id (header placeholders of its tool-server
  connection). Only requests with one of CONTEXT_TOKENS keep these headers (other clients cannot read chats). The
  server then reads that chat's files through Open WebUI's API (OWUI_API_KEY) after checking the chat belongs to the
  user; file arguments left empty pick the newest fitting attachment, "att:N" picks the N-th newest.
- Video tools wait for the render (progress notifications keep the call alive; the GPU is the video's alone
  meanwhile, so the calling chat model is idle anyway) up to VIDEO_MAX_WAIT seconds, then return the watch page.
"""
import asyncio
import base64
import functools
import hashlib
import io
import ipaddress
import json
import logging
import math
import os
import re
import secrets
import socket
import subprocess
import tempfile
import time
import urllib.parse
from pathlib import Path
from typing import Literal

import httpx
from mcp.server.mcpserver import Audio, Context, Image, MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import PlainTextResponse

LSW = os.environ.get("LLAMA_SWAP_URL", "http://127.0.0.1:8080").rstrip("/")
KEY = os.environ.get("LLAMA_SWAP_KEY", "")
SEARX = os.environ.get("SEARXNG_URL", "http://searxng:8080").rstrip("/")
PUBLIC = os.environ.get("PUBLIC_BASE", "http://127.0.0.1:8080").rstrip("/")
TOKENS = {t.strip() for t in os.environ.get("MCP_TOKENS", "").split(",") if t.strip()}
CONTEXT_TOKENS = {t.strip() for t in os.environ.get("CONTEXT_TOKENS", "").split(",") if t.strip()}
OWUI = os.environ.get("OWUI_URL", "http://open-webui:8080").rstrip("/")
OWUI_KEY = os.environ.get("OWUI_API_KEY", "")
VIDEO_MAX_WAIT = int(os.environ.get("VIDEO_MAX_WAIT", "7200"))
# hosts whose files tools may download as inputs besides public internet hosts (e.g. a chat UI's presigned S3 links)
INPUT_HOSTS = {h for h in os.environ.get("INPUT_HOSTS", "").split(",") if h}
MAX_INPUT = 200 << 20
VIDEO_POST = VIDEO_GET = f"{LSW}/v1/videos"       # the Mac Studio gateway routes POST to the renderer
VIDEO_ID = re.compile(r"video_[0-9a-f]{24}")
MEDIA_ID = re.compile(r"(video_[0-9a-f]{24}|img_[0-9a-f]{16})")
CONTEXT_HEADERS = (b"x-chat-id", b"x-user-id", b"x-user-message-id")
PRESET_MODEL = {"fast": "minimax-h3-video-fast", "standard": "minimax-h3-video", "best": "minimax-h3-video-best"}
DEFAULT_MODEL = {"chat": "qwen3.8-flash-next", "asr": "qwen3-asr-1.7b", "embedding": "embeddinggemma-2",
                 "image": "minimax-h3-image", "tts": "qwen3-tts", "rerank": "qwen3-reranker-4b-int8",
                 "decision": "elephant-1"}
KINDS = ("chat", "image", "video", "asr", "tts", "embedding", "rerank", "decision")
CHAT_ENDPOINTS = "/v1/chat/completions (OpenAI), /v1/messages (Anthropic), /v1/responses (OpenAI Responses)"
API_BASES = os.environ.get("API_BASES", "local http://127.0.0.1:8080/v1")
log = logging.getLogger("llm-mcp")
AUTH = {"Authorization": f"Bearer {KEY}"}

mcp = MCPServer(
    "local-models",
    instructions=(
        "Tools backed by the user's own models on a Mac Studio: chat models, web search and page fetching, image "
        "generation and editing (MiniMax-H3 up to 4096x4096; Qwen-Image 2.1 for text in images, masks and transparent "
        "PNGs), video with synchronized sound (up to 15 s per render; longer by continuing with extend_video), "
        "speech-to-text, text-to-speech (speak), text embeddings, reranking, decision probabilities (decide) and file "
        "inspection. Files the user attached "
        "in this chat are used automatically when a file argument is left empty (list_attachments shows them). "
        "Videos take minutes and use the GPU exclusively: agree on the content, length and quality with the user, "
        "tell them the expected time, then call the tool once. Model arguments may be left empty for the default; "
        "list_models shows what is available right now."),
)


class ToolInputError(ValueError):
    """A problem the calling model can fix; reported as the tool's text result."""


def friendly(fn):
    """Turn input problems and backend refusals into a readable tool result instead of an opaque tool error."""
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except (ToolInputError, RuntimeError) as e:
            return f"Error: {e}"
        except httpx.HTTPError as e:
            return f"Error: backend request failed: {e}"
    return wrapper


# ---------------------------------------------------------------------------------------------------- models
_models = {"t": 0.0, "data": []}


async def live_models():
    """llama-swap's model list, cached for a minute."""
    if time.time() - _models["t"] > 60 or not _models["data"]:
        try:
            async with httpx.AsyncClient(timeout=15) as c:
                r = await c.get(f"{LSW}/v1/models", headers=AUTH)
                r.raise_for_status()
            _models.update(t=time.time(), data=r.json().get("data", []))
        except httpx.HTTPError as e:
            log.warning("model list refresh failed: %s", e)
    return _models["data"]


def kind_of(m):
    a = m.get("architecture") or {}
    ins, outs, mid = a.get("input_modalities") or [], a.get("output_modalities") or [], m["id"].lower()
    if "rerank" in mid:
        return "rerank"
    if "embed" in mid or "voyage" in mid or "embedding" in outs:
        return "embedding"
    if "decisions" in outs or mid.startswith("drex"):
        return "decision"
    if "audio" in outs:
        return "tts"
    if "video" in outs or "-video" in mid:
        return "video"
    if "image" in outs or "-image" in mid:
        return "image"
    if a.get("modality") == "audio->text" or ins == ["audio"]:
        return "asr"
    return "chat"


async def pick_model(kind, asked):
    ids = [m["id"] for m in await live_models() if kind_of(m) == kind]
    if not asked:
        return DEFAULT_MODEL[kind] if DEFAULT_MODEL[kind] in ids or not ids else ids[0]
    if asked in ids:
        return asked
    raise ToolInputError(f"unknown {kind} model {asked!r}; available now: {', '.join(ids) or 'none'}")


# ---------------------------------------------------------------------------------------------------- inputs
def _public_host_ok(host):
    """Refuse hosts that resolve to private, loopback or link-local addresses (no LAN probing through fetch tools)."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    return True


def own_media(ref):
    """(id, URL on the Mac Studio) for a media id or one of the PUBLIC/media URLs, else None."""
    m = MEDIA_ID.search(ref or "")
    if not m or not (ref == m.group(1) or ref.startswith(f"{PUBLIC}/media/")):
        return None
    mid = m.group(1)
    return mid, f"{LSW}/media/{mid}." + ("mp4" if mid.startswith("video_") else "png")


def file_kind(name, ctype):
    ctype, ext = (ctype or "").lower(), Path(name or "").suffix.lower()
    if ctype.startswith("image/") or ext in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".heic"):
        return "image"
    if ctype.startswith("video/") or ext in (".mp4", ".mov", ".webm", ".m4v", ".mkv", ".avi"):
        return "video"
    if ctype.startswith("audio/") or ext in (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac"):
        return "audio"
    return "file"


def chat_context(ctx):
    h = (ctx.headers if ctx else None) or {}
    return h.get("x-chat-id") or "", h.get("x-user-id") or "", h.get("x-user-message-id") or ""


async def chat_files(ctx):
    """Files of the calling Open WebUI chat, newest first: [{n, id, name, kind, content_type, from}], or None when
    the request carries no chat context."""
    chat_id, user_id, _ = chat_context(ctx)
    if not chat_id or not OWUI_KEY:
        return None
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(f"{OWUI}/api/v1/chats/{urllib.parse.quote(chat_id)}",
                        headers={"Authorization": f"Bearer {OWUI_KEY}"})
    if r.status_code != 200:
        raise ToolInputError(f"cannot read this chat's attachments (Open WebUI answered {r.status_code})")
    chat = r.json()
    if user_id and chat.get("user_id") != user_id:
        raise ToolInputError("this chat does not belong to the calling user")
    msgs = sorted(((chat.get("chat") or {}).get("history") or {}).get("messages", {}).values(),
                  key=lambda m: m.get("timestamp") or 0)
    out = []
    for m in msgs:
        for f in m.get("files") or []:
            info = f.get("file") if isinstance(f.get("file"), dict) else {}
            fid = f.get("id") or info.get("id")
            if not fid and isinstance(f.get("url"), str):
                fid = (re.search(r"/files/([0-9a-f-]{36})", f["url"]) or [None, None])[1]
            if not fid:
                continue
            name = f.get("name") or info.get("filename") or ""
            ctype = f.get("content_type") or (info.get("meta") or {}).get("content_type") or ""
            out.append({"id": fid, "name": name, "content_type": ctype, "kind": file_kind(name, ctype),
                        "from": "user" if m.get("role") == "user" else "assistant"})
    out.reverse()
    for i, f in enumerate(out, 1):
        f["n"] = i
    return out


async def owui_file(fid):
    async with httpx.AsyncClient(timeout=300) as c:
        r = await c.get(f"{OWUI}/api/v1/files/{fid}/content", headers={"Authorization": f"Bearer {OWUI_KEY}"})
        r.raise_for_status()
        return r.content


async def resolve_input(ref, ctx, want, what):
    """(bytes, name, content_type) of an input. ref: "" (newest attachment of kind `want`), "att:N", one of this
    server's media ids/URLs, a data: URL, an allowed or public http(s) URL."""
    ref = (ref or "").strip()
    m = re.fullmatch(r"(?:att(?:achment)?:)?(\d+)", ref) if ref else None
    if not ref or m:
        files = await chat_files(ctx)
        if files is None:
            raise ToolInputError(f"{what}: give a URL (this client passes no chat attachments)")
        cands = files if want == "any" else [f for f in files if f["kind"] == want]
        if m:
            cands = [f for f in files if f["n"] == int(m.group(1))]
        if not cands:
            raise ToolInputError(f"{what}: no {'' if want == 'any' else want + ' '}attachment in this chat"
                                 + ("" if m else "; attach one or give a URL"))
        f = cands[0]
        return await owui_file(f["id"]), f["name"], f["content_type"]
    if ref.startswith("data:"):
        head, _, b64 = ref.partition(",")
        return base64.b64decode(b64), "data", head[5:].split(";")[0]
    own = own_media(ref)
    if own:
        async with httpx.AsyncClient(timeout=300) as c:
            r = await c.get(own[1], headers=AUTH)
        if r.status_code != 200:
            raise ToolInputError(f"{what}: {own[0]} is not available (HTTP {r.status_code})")
        return r.content, own[1].rsplit("/", 1)[1], r.headers.get("content-type", "").split(";")[0]
    u = urllib.parse.urlsplit(ref)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ToolInputError(f"{what}: need an attachment (empty or att:N), a URL or a data: URL")
    if u.hostname not in INPUT_HOSTS and not _public_host_ok(u.hostname):
        raise ToolInputError(f"{what}: {u.hostname} is not a public host")
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as c:
        r = await c.get(ref)
        r.raise_for_status()
    if len(r.content) > MAX_INPUT:
        raise ToolInputError(f"{what}: larger than {MAX_INPUT >> 20} MB")
    return r.content, Path(u.path).name, r.headers.get("content-type", "").split(";")[0]


def data_url(data, ctype):
    return f"data:{ctype or 'application/octet-stream'};base64," + base64.b64encode(data).decode()


# ---------------------------------------------------------------------------------------------------- media helpers
def links(mid):
    ext = "mp4" if mid.startswith("video_") else "png"
    return f"{PUBLIC}/media/{mid}", f"{PUBLIC}/media/{mid}.{ext}"


async def video_job(vid):
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.get(f"{VIDEO_GET}/{vid}", headers=AUTH)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()


def describe_video(job):
    st = job.get("settings") or {}
    length = (f"{job['seconds']} s in total; this step added {job.get('segment_seconds')} s, guided by the last "
              f"{st.get('context_seconds')} s of {job.get('extends')}" if job.get("extends") else f"{job['seconds']} s")
    return (f"id {job['id']}, {job['size']}, {length}, preset {st.get('preset')}, seed {job.get('seed')}, "
            f"rendered in {(job.get('timings') or {}).get('render_s')} s")


def video_done_text(job):
    page, file = links(job["id"])
    return (f"Video ready: {describe_video(job)}.\n"
            f"To show it in the chat, put this line on its own line in your reply: <video>{file}</video>\n"
            f"Watch/download page: {page} (download: {file}).\n"
            f"Tell the user what was made, give the page link, and offer to continue it: extend_video adds up to about "
            f"12 s per step from a description of what happens next (keep the id {job['id']}).")


async def wait_video(ctx, job, t0):
    """Poll a render until it ends or VIDEO_MAX_WAIT passes, sending progress (and a log line each minute)."""
    last_log = 0.0
    while job.get("status") not in ("completed", "failed"):
        el = time.time() - t0
        if el > VIDEO_MAX_WAIT:
            page, _ = links(job["id"])
            return (f"Still rendering after {int(el)} s (video {job['id']}, {job.get('progress', 0)}%). The user can "
                    f"follow it on {page}; call get_video with this id later to fetch the result.")
        msg = f"rendering {job.get('progress', 0)}% ({job.get('phase') or job.get('status')}), {int(el)} s"
        if ctx:
            await ctx.report_progress(float(job.get("progress") or 0), 100.0, msg)
            if time.time() - last_log > 60:
                await ctx.info(msg)          # a server->client message also keeps the HTTP stream busy
                last_log = time.time()
        await asyncio.sleep(5)
        job = await video_job(job["id"]) or job
    if job["status"] == "failed":
        return f"The render failed: {(job.get('error') or {}).get('message', '')}"
    return video_done_text(job)


async def post_video(req):
    async with httpx.AsyncClient(timeout=600) as c:
        r = await c.post(VIDEO_POST, json=req, headers=AUTH)
    if r.status_code == 429:
        raise RuntimeError("another video is being rendered right now (one at a time); try again later")
    body = r.json()
    if r.status_code != 200:
        raise RuntimeError(f"the video request was rejected: {(body.get('error') or {}).get('message', body)}")
    return body


# ---------------------------------------------------------------------------------------------------- models, chat
@mcp.tool()
@friendly
async def list_models() -> str:
    """The models available on the Mac Studio right now, by use (chat, image, video, asr, embedding), with their
    names and descriptions, plus the HTTP API they are served on (for agents that can make HTTP requests themselves).
    Model ids for the other tools come from here; left empty, each tool uses its default."""
    groups = {}
    for m in await live_models():
        groups.setdefault(kind_of(m), []).append(m)
    lines = []
    for kind in KINDS:
        if kind not in groups:
            continue
        note = {"chat": f"tool: chat; HTTP API: {CHAT_ENDPOINTS}", "image": "tools: generate_image, edit_image",
                "video": "tools: generate_video, extend_video (choose quality, not a model id)",
                "asr": "tool: transcribe", "tts": "tool: speak; HTTP API: /v1/audio/speech",
                "embedding": "tool: embed", "rerank": "tool: rerank; HTTP API: /v1/rerank",
                "decision": "tool: decide; HTTP API: /v1/decisions (OpenAI) and /v1/systemone"}[kind]
        lines.append(f"{kind} ({note}; default {DEFAULT_MODEL.get(kind, '-')}):")
        for m in groups[kind]:
            lines.append(f"  - {m['id']}: {m.get('name', '')}" + (f" - {m['description']}" if m.get("description") else ""))
    lines.append(f"\nHTTP API (OpenAI-compatible, same models): {API_BASES}. It needs an API key (Authorization: "
                 "Bearer ...) that is not handed out here: use the key configured for you, or these tools.")
    return "\n".join(lines)


@mcp.tool()
@friendly
async def chat(prompt: str, model: str = "", system: str = "", images: list[str] = [],
               json_schema: dict = {}, temperature: float = -1, max_tokens: int = 4096,
               ctx: Context = None) -> str:
    """Ask a local chat model (Qwen3.8 Flash-Next on the Mac Studio, 262k context, vision; the -uncensored variant
    answers without refusals). Useful for a second opinion, for private text that should stay local, or to delegate.

    Args:
        prompt: the message
        model: a chat model id from list_models; empty = qwen3.8-flash-next
        system: optional system prompt
        images: images for the model to look at: attachments ("" is not allowed here; use "att:N"), media ids, URLs
        json_schema: a JSON Schema object to force a structured JSON answer; {} for free text
        temperature: 0-2, or -1 for the model default
        max_tokens: answer length limit, up to 32768
    """
    model = await pick_model("chat", model)
    content = prompt
    if images:
        content = [{"type": "text", "text": prompt}]
        for i, ref in enumerate(images[:8]):
            data, _, ctype = await resolve_input(ref, ctx, "image", f"images[{i}]")
            content.append({"type": "image_url", "image_url": {"url": data_url(data, ctype or "image/png")}})
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
    body = {"model": model, "messages": msgs, "max_tokens": max(16, min(32768, max_tokens))}
    if temperature >= 0:
        body["temperature"] = min(2.0, temperature)
    if json_schema:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "result", "schema": json_schema}}
    async with httpx.AsyncClient(timeout=1800) as c:
        r = await c.post(f"{LSW}/v1/chat/completions", headers=AUTH, json=body)
    if r.status_code != 200:
        return f"Error: the model call failed (HTTP {r.status_code}): {r.text[:300]}"
    return r.json()["choices"][0]["message"].get("content") or ""


# ---------------------------------------------------------------------------------------------------- search
@mcp.tool()
@friendly
async def web_search(query: str, count: int = 8, language: str = "auto",
                     time_range: Literal["", "day", "week", "month", "year"] = "") -> str:
    """Search the web (the user's SearXNG). Returns title, URL and snippet per result; use fetch_url to read a page.

    Args:
        query: search terms
        count: number of results, 1-20
        language: language code such as "zh-CN" or "en", or "auto"
        time_range: limit to the last day/week/month/year, or "" for any time
    """
    params = {"q": query, "format": "json", "language": language}
    if time_range:
        params["time_range"] = time_range
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(f"{SEARX}/search", params=params)
        r.raise_for_status()
    results = r.json().get("results", [])[:max(1, min(20, count))]
    if not results:
        return "No results."
    return "\n\n".join(f"{i}. {x.get('title', '')}\n{x.get('url', '')}\n{(x.get('content') or '').strip()}"
                       for i, x in enumerate(results, 1))


@mcp.tool()
@friendly
async def fetch_url(url: str, max_chars: int = 20000) -> str:
    """Fetch a public web page and return its main text (boilerplate removed), or the raw text for plain files.
    For binary files use inspect_file.

    Args:
        url: http(s) URL on the public internet
        max_chars: cut the text after this many characters (1000-100000)
    """
    import trafilatura
    u = urllib.parse.urlsplit(url)
    if u.scheme not in ("http", "https") or not u.hostname or not _public_host_ok(u.hostname):
        return "Only public http(s) URLs can be fetched."
    async with httpx.AsyncClient(timeout=30, follow_redirects=True,
                                 headers={"User-Agent": "Mozilla/5.0 (llm-mcp fetch_url)"}) as c:
        r = await c.get(url)
    if not _public_host_ok(urllib.parse.urlsplit(str(r.url)).hostname or ""):
        return "The page redirected to a non-public host."
    ctype = r.headers.get("content-type", "")
    if "html" in ctype:
        text = trafilatura.extract(r.text, include_links=False, include_tables=True, favor_recall=True) or ""
    elif ctype.startswith("text/") or "json" in ctype or "xml" in ctype:
        text = r.text
    else:
        return f"Not a text page ({ctype or 'unknown type'}, HTTP {r.status_code}); inspect_file can analyse it."
    limit = max(1000, min(100000, max_chars))
    head = f"HTTP {r.status_code} {r.url}\n\n"
    return head + (text[:limit] + f"\n\n[cut at {limit} of {len(text)} characters]" if len(text) > limit else text)


# ---------------------------------------------------------------------------------------------------- images
def _image_result(item, size):
    """MCP result for one image of the image API: a downscaled preview plus the Mac Studio's links (id kept there)."""
    png, mid = base64.b64decode(item["b64_json"]), item.get("id")
    from PIL import Image as PILImage
    im = PILImage.open(io.BytesIO(png))
    w, h = im.size
    im = im.convert("RGB")
    im.thumbnail((1024, 1024))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    if not mid:
        return [Image(data=buf.getvalue(), format="jpeg"), f"Image ready ({w}x{h}); the server kept no copy, so there is no link."]
    page, file = links(mid)
    revised = f"\nThe prompt was rewritten to: {item['revised_prompt']}" if item.get("revised_prompt") else ""
    return [Image(data=buf.getvalue(), format="jpeg"),
            f"Image ready: id {mid}, {w}x{h}. The preview above is downscaled; full size: {file} (page {page}). "
            f"To show it in the chat put ![image]({file}) in your reply. edit_image can change it further (pass {mid})."
            + revised]


def _qwen_extras(model, prompt_enhance, background):
    """Extra fields only the Qwen-Image server understands (MiniMax-H3 would reject or ignore them)."""
    if not model.startswith("qwen-image"):
        if prompt_enhance not in ("", "off") or background != "auto":
            raise ToolInputError("prompt_enhance and background are Qwen-Image options; pick a qwen-image-2.1 model")
        return {}
    out = {}
    if prompt_enhance:
        out["prompt_enhance"] = prompt_enhance
    if background != "auto":
        out["background"] = background
    return out


@mcp.tool()
@friendly
async def generate_image(prompt: str, size: str = "1024x1024", seed: int = -1, model: str = "",
                         prompt_enhance: Literal["", "off", "llm", "mflux"] = "",
                         background: Literal["auto", "transparent"] = "auto", ctx: Context = None) -> list:
    """Generate an image. 1024x1024 takes ~10-20 s (MiniMax-H3) or ~9 s (Qwen-Image turbo).

    Prompting: describe subject, composition, style, lighting and camera; put any text to render in quotes. For
    Qwen-Image a detailed English or Chinese description works best; short prompts gain from prompt_enhance="llm".

    Args:
        prompt: detailed description: subject, composition, style, lighting, text to render (if any)
        size: WxH, e.g. 1024x1024, 1536x1024, 1024x1536, 2048x2048 (MiniMax-H3 up to 4096 per side), or "auto"
            for Qwen-Image with prompt_enhance (the rewrite picks the aspect ratio)
        seed: -1 for random
        model: an image model id from list_models; empty = minimax-h3-image
        prompt_enhance: Qwen-Image only: "" = the model's default, "off", "llm" (rewritten by the chat model, best),
            "mflux" (rewritten by Qwen-Image's own text encoder)
        background: Qwen-Image only: "transparent" for an RGBA PNG with a transparent background

    Image models (pick with `model`; list_models shows what is up):
        - minimax-h3-image (default): MiniMax-H3, any size up to 4096x4096, strong photorealism.
        - qwen-image-2.1-turbo: Qwen-Image 2.1, 8 steps, ~9 s at 1024x1024; best at rendering text (Chinese and
          English) inside images, masks and transparent backgrounds; 256-2752 per side, area up to 2400x1792.
        - qwen-image-2.1-turbo-pe: the same, but a short prompt is first expanded by the chat model (+20-30 s).
        - qwen-image-2.1: Qwen-Image 2.1 base, 40 steps (~30 s at 1024x1024).
    """
    model = await pick_model("image", model)
    body = {"model": model, "prompt": prompt, "size": size, "n": 1, **_qwen_extras(model, prompt_enhance, background)}
    if seed >= 0:
        body["seed"] = seed
    if ctx:
        await ctx.info(f"generating {size}")
    async with httpx.AsyncClient(timeout=900) as c:
        r = await c.post(f"{LSW}/v1/images/generations", json=body, headers=AUTH)
    if r.status_code != 200:
        return [f"Error: image generation failed (HTTP {r.status_code}): {r.text[:300]}"]
    return _image_result(r.json()["data"][0], size)


@mcp.tool()
@friendly
async def edit_image(prompt: str, images: list[str] = [], size: str = "auto", seed: int = -1, model: str = "",
                     prompt_enhance: Literal["", "off", "llm", "mflux"] = "",
                     background: Literal["auto", "transparent"] = "auto", ctx: Context = None) -> list:
    """Edit or combine images: change, add or remove things, restyle, or compose several inputs. Refer to the
    inputs as <Picture 1>, <Picture 2>, ... in the prompt (MiniMax-H3); Qwen-Image also understands "the first
    image", "the second image".

    Args:
        prompt: the edit, e.g. "Make <Picture 1> a watercolor" or "Put the cat from <Picture 1> on the sofa from
            <Picture 2>"
        images: inputs, in order: "att:N" (attachment N of this chat, see list_attachments), an image id
            from generate_image/edit_image, a URL or a data: URL. Empty = the newest image in this chat (an upload
            or a generated one). MiniMax-H3 takes up to 16, Qwen-Image up to 10.
        size: WxH or "auto" (the first image's size)
        seed: -1 for random
        model: an image model id from list_models; empty = minimax-h3-image. qwen-image-2.1-turbo is fast (~10 s)
            and keeps everything outside the edited region unchanged; good for text edits.
        prompt_enhance: Qwen-Image only: "" = the model's default, "off", "llm" (the chat model looks at the images
            and rewrites the instruction), "mflux"
        background: Qwen-Image only: "transparent" for an RGBA PNG
    """
    refs = images or [""]
    model = await pick_model("image", model)
    files = [("prompt", (None, prompt)), ("size", (None, size)), ("n", (None, "1")), ("model", (None, model))]
    files += [(k, (None, v)) for k, v in _qwen_extras(model, prompt_enhance, background).items()]
    if seed >= 0:
        files.append(("seed", (None, str(seed))))
    for i, ref in enumerate(refs[:10 if model.startswith("qwen-image") else 16]):
        data, name, ctype = await resolve_input(ref, ctx, "image", f"images[{i}]")
        files.append(("image[]", (name or f"ref{i + 1}.png", data, ctype or "image/png")))
    async with httpx.AsyncClient(timeout=900) as c:
        r = await c.post(f"{LSW}/v1/images/edits", files=files, headers=AUTH)
    if r.status_code != 200:
        return [f"Error: image edit failed (HTTP {r.status_code}): {r.text[:300]}"]
    return _image_result(r.json()["data"][0], size)


# ---------------------------------------------------------------------------------------------------- video
@mcp.tool()
@friendly
async def generate_video(prompt: str, seconds: float = 5,
                         size: Literal["1344x768", "768x1344", "768x768", "512x512"] = "1344x768",
                         quality: Literal["fast", "standard", "best"] = "standard",
                         first_frame: str = "none", ctx: Context = None) -> str:
    """Generate a video WITH synchronized sound (MiniMax-H3), up to 15 s. Before calling: agree on content, length,
    orientation and quality with the user and tell them the expected time. Render time at 1344x768 "standard":
    5 s ~4.5 min (15 s not measured yet with the new sparse layout; roughly 30-40 min); "fast" 5 s ~3 min, 15 s
    ~13 min; "best" 5 s ~12 min (~2.7x standard); 512x512 is ~4x faster. The call waits
    for the render. For longer videos generate up to 15 s, then use extend_video. To continue a video the user
    uploaded, use extend_video instead.

    Args:
        prompt: English, in this structure: "integrated_multimodal_description: [Shot 1] subjects and their look,
            the actions in order, camera framing and movement, lighting, visual style ([Shot 2] ... for cuts).
            overall_soundscape: ambience and physical sounds. non_diegetic_music: background music, or none."
            Put spoken lines with who says them in the description.
        seconds: 1-15
        size: 1344x768 landscape, 768x1344 portrait, 768x768 square, 512x512 small and fast
        quality: fast (token reduction, composition may drift), standard (default), best (exact full
            computation, slowest)
        first_frame: "none", or an image to start from: "" = the newest image in this chat, "att:N", an image id,
            a URL
    """
    req = {"model": PRESET_MODEL[quality], "prompt": prompt, "size": size, "seconds": max(1.0, min(15.0, seconds))}
    if first_frame != "none":
        data, _, ctype = await resolve_input(first_frame, ctx, "image", "first_frame")
        req["input_reference"] = data_url(data, ctype or "image/png")
    t0 = time.time()
    return await wait_video(ctx, await post_video(req), t0)


@mcp.tool()
@friendly
async def extend_video(prompt: str, video: str = "", seconds: float = 5, context_seconds: float = 3,
                       quality: Literal["fast", "standard", "best"] = "standard", ctx: Context = None) -> str:
    """Continue a video: the model re-creates its last context_seconds (picture and sound) and goes on, and the
    result is the whole, longer video (same size). Works on videos made here and on videos the user uploaded.
    Each step adds up to 15 - context_seconds seconds; chain steps for longer videos. Same render times as
    generate_video (plus ~40%). Call it after the user said what should happen next.

    Args:
        prompt: what happens next (English): the actions in order, camera, how the sound continues or changes.
            Do not describe the earlier video again; it is given to the model.
        video: "" = the newest video in this chat (an upload, else the last generated one), "att:N", a video id
            (video_...), this server's media URL or another video URL
        seconds: length of the new part, 1-12
        context_seconds: final seconds of the earlier video that guide the continuation, 2-8 (default 3)
        quality: fast, standard or best
    """
    req = {"model": PRESET_MODEL[quality], "prompt": prompt, "seconds": max(1.0, min(13.0, seconds)),
           "extend_context_seconds": max(2.0, min(8.0, context_seconds))}
    m = VIDEO_ID.search(video or "")
    if m and (video == m.group(0) or video.startswith(f"{PUBLIC}/media/")):
        req["extend_from"] = m.group(0)
    else:
        try:
            data, _, ctype = await resolve_input(video, ctx, "video", "video")
            if file_kind("", ctype) == "video" or not ctype:
                req["video"] = data_url(data, ctype or "video/mp4")
            else:
                raise ToolInputError("video: that attachment is not a video")
        except ToolInputError:
            if video:
                raise
            last = await last_generated_video(ctx)        # no uploaded video: continue the chat's last render
            if not last:
                raise
            req["extend_from"] = last
    t0 = time.time()
    return await wait_video(ctx, await post_video(req), t0)


async def last_generated_video(ctx):
    """Newest video id mentioned by the assistant in the calling Open WebUI chat."""
    chat_id, user_id, _ = chat_context(ctx)
    if not chat_id or not OWUI_KEY:
        return None
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(f"{OWUI}/api/v1/chats/{urllib.parse.quote(chat_id)}",
                        headers={"Authorization": f"Bearer {OWUI_KEY}"})
    if r.status_code != 200 or (user_id and r.json().get("user_id") != user_id):
        return None
    msgs = sorted(((r.json().get("chat") or {}).get("history") or {}).get("messages", {}).values(),
                  key=lambda m: m.get("timestamp") or 0, reverse=True)
    for m in msgs:
        ids = VIDEO_ID.findall(json.dumps(m.get("content") or "") + json.dumps(m.get("output") or ""))
        if m.get("role") == "assistant" and ids:
            return ids[-1]
    return None


@mcp.tool()
@friendly
async def get_video(video_id: str) -> str:
    """Status of a video render that was still running when generate_video/extend_video returned, and its links
    once done.

    Args:
        video_id: video_...
    """
    job = await video_job(video_id)
    if not job:
        return f"No video {video_id}."
    if job["status"] == "completed":
        return video_done_text(job)
    if job["status"] == "failed":
        return f"The render failed: {(job.get('error') or {}).get('message', '')}"
    return (f"Still rendering: {job.get('progress', 0)}% ({job.get('phase') or job.get('status')}). "
            f"Page: {links(video_id)[0]}")


# ---------------------------------------------------------------------------------------------------- audio, text
@mcp.tool()
@friendly
async def transcribe(audio: str = "", language: str = "", timestamps: bool = False, model: str = "",
                     ctx: Context = None) -> str:
    """Speech to text (Qwen3-ASR by default; Chinese, English and many more; long recordings are fine). Takes audio
    or the soundtrack of a video.

    Args:
        audio: "" = the newest audio or video attachment in this chat, "att:N", or a URL / data: URL
        language: language code ("zh", "en", ...) or "" to detect
        timestamps: true for segment timestamps
        model: an asr model id from list_models; empty = qwen3-asr-1.7b
    """
    try:
        data, name, _ = await resolve_input(audio, ctx, "audio", "audio")
    except ToolInputError:
        if audio:
            raise
        data, name, _ = await resolve_input("", ctx, "video", "audio")
    form = {"model": await pick_model("asr", model), "response_format": "verbose_json" if timestamps else "json"}
    if language:
        form["language"] = language
    async with httpx.AsyncClient(timeout=1800) as c:
        r = await c.post(f"{LSW}/v1/audio/transcriptions", data=form, files={"file": (name or "audio", data)},
                         headers=AUTH)
    if r.status_code != 200:
        return f"Error: transcription failed (HTTP {r.status_code}): {r.text[:300]}"
    d = r.json()
    if timestamps and d.get("segments"):
        return "\n".join(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text'].strip()}" for s in d["segments"])
    return d.get("text", "")


@mcp.tool()
@friendly
async def embed(texts: list[str], input_type: Literal["", "query", "document"] = "", dimensions: int = 768,
                model: str = "") -> str:
    """Text embeddings (EmbeddingGemma 2, 768 dimensions; Matryoshka: 512/256/128 keep most quality). Returns JSON
    {"model", "dimensions", "vectors": [[...], ...]}, L2-normalized.

    Args:
        texts: up to 256 texts
        input_type: "query" for search queries, "document" for passages to index, "" for neither
        dimensions: 128-768 (truncated and re-normalized below 768)
        model: an embedding model id from list_models; empty = embeddinggemma-2
    """
    model = await pick_model("embedding", model)
    body = {"model": model, "input": texts[:256], "dimensions": max(128, min(768, dimensions))}
    if input_type:
        body["input_type"] = input_type
    async with httpx.AsyncClient(timeout=300) as c:
        r = await c.post(f"{LSW}/v1/embeddings", json=body, headers=AUTH)
        r.raise_for_status()
    vecs = [x["embedding"] for x in r.json()["data"]]
    return json.dumps({"model": model, "dimensions": len(vecs[0]) if vecs else 0, "vectors": vecs})


async def owui_upload(ctx, data, name, ctype):
    """Store a generated file in the calling Open WebUI chat's files; its content URL, or None without chat context."""
    chat_id, _, _ = chat_context(ctx)
    if not chat_id or not OWUI_KEY:
        return None
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(f"{OWUI}/api/v1/files/", headers={"Authorization": f"Bearer {OWUI_KEY}"},
                         files={"file": (name, data, ctype)}, params={"process": "false"})
    if r.status_code != 200:
        log.warning("Open WebUI upload failed: %s %s", r.status_code, r.text[:200])
        return None
    return f"/api/v1/files/{r.json()['id']}/content"


@mcp.tool()
@friendly
async def speak(text: str, voice: str = "", instructions: str = "", model: str = "", speed: float = 1.0,
                reference_audio: str = "", reference_text: str = "", ctx: Context = None) -> list:
    """Text to speech (Qwen3-TTS 1.7B: Chinese, English, Japanese, Korean, German, French, Russian, Italian,
    Spanish, Portuguese). Returns an MP3. ~4x faster than real time; long texts are fine.

    Models:
        - qwen3-tts (default): 9 preset voices; `instructions` sets emotion, style or dialect, e.g. "开心地说" or
          "Speak slowly and calmly". Voices: vivian, serena (Chinese female), uncle_fu (Chinese male, older), dylan
          (Beijing), eric (Chengdu / Sichuanese), ryan, aiden (English male), ono_anna (Japanese), sohee (Korean);
          OpenAI names (alloy, nova, echo, onyx, ...) are mapped onto them.
        - qwen3-tts-design: creates a new voice from a description in `instructions`, e.g. "a deep, warm male voice
          speaking fluent Cantonese" (for Cantonese use this model).
        - qwen3-tts-clone: copies the voice of `reference_audio` (3-20 s), which needs its exact `reference_text`.

    Args:
        text: what to say
        voice: preset voice for qwen3-tts; empty = vivian
        instructions: speaking style for qwen3-tts, or the voice description for qwen3-tts-design
        model: a tts model id from list_models; empty = qwen3-tts
        speed: 0.25-4.0
        reference_audio: qwen3-tts-clone only: "att:N", a URL or data: URL of the voice to copy
        reference_text: qwen3-tts-clone only: the exact words spoken in reference_audio
    """
    model = await pick_model("tts", model)
    body = {"model": model, "input": text, "response_format": "mp3", "speed": max(0.25, min(4.0, speed))}
    if voice:
        body["voice"] = voice
    if instructions:
        body["instructions"] = instructions
    if reference_audio:
        data, _, ctype = await resolve_input(reference_audio, ctx, "audio", "reference_audio")
        body["ref_audio"] = data_url(data, ctype or "audio/wav")
        body["ref_text"] = reference_text
    async with httpx.AsyncClient(timeout=1800) as c:
        r = await c.post(f"{LSW}/v1/audio/speech", json=body, headers=AUTH)
    if r.status_code != 200:
        return [f"Error: speech synthesis failed (HTTP {r.status_code}): {r.text[:300]}"]
    mp3 = r.content
    seconds = len(mp3) * 8 / 128000      # the server encodes 128 kbit/s MP3
    out = [Audio(data=mp3, format="mpeg")]
    url = await owui_upload(ctx, mp3, f"speech-{int(time.time())}.mp3", "audio/mpeg")
    if url:
        out.append(f"Speech ready (~{seconds:.0f} s, MP3), saved in this chat: {url}. To let the user play it, put "
                   f"<audio controls src=\"{url}\"></audio> on its own line in your reply.")
    else:
        out.append(f"Speech ready (~{seconds:.0f} s, MP3), attached above as audio.")
    return out


@mcp.tool()
@friendly
async def rerank(query: str, documents: list[str], top_n: int = 0, instruction: str = "", model: str = "") -> str:
    """Rank passages by relevance to a query (Qwen3-Reranker). Returns JSON [{"index", "score", "text"}], best first;
    score is a 0-1 relevance probability. Use it to pick the best search results or chunks before answering.

    Args:
        query: the question or search query
        documents: up to 200 passages
        top_n: keep only the best N (0 = all)
        instruction: optional task description, e.g. "Given a legal question, find the passage that answers it"
        model: a rerank model id from list_models; empty = qwen3-reranker-4b-int8 (-0.6b fastest, -8b most accurate)
    """
    model = await pick_model("rerank", model)
    body = {"model": model, "query": query, "documents": documents[:200]}
    if top_n > 0:
        body["top_n"] = top_n
    if instruction:
        body["instruction"] = instruction
    async with httpx.AsyncClient(timeout=600) as c:
        r = await c.post(f"{LSW}/v1/rerank", json=body, headers=AUTH)
    if r.status_code != 200:
        return f"Error: rerank failed (HTTP {r.status_code}): {r.text[:300]}"
    res = sorted(r.json()["results"], key=lambda x: -x["relevance_score"])
    return json.dumps([{"index": x["index"], "score": round(x["relevance_score"], 4),
                        "text": documents[x["index"]][:200]} for x in res], ensure_ascii=False)


@mcp.tool()
@friendly
async def decide(input: str, questions: list[dict], model: str = "", images: list[str] = [],
                 ctx: Context = None) -> str:
    """Calibrated probabilities for yes/no, multiple-choice and rating questions about a text (OpenAI Decisions
    API on local decision models). Much faster and better calibrated than asking a chat model; use it for
    classification, routing, moderation, triage, grading.

    Models: elephant-1 (default, 9B, fits next to the chat model), mammoth-1 (27B, most
    accurate; unloads the chat model while it runs), pplx-decider-flash (fast 4B; -multimodal also reads images),
    drex-v1.5.

    Args:
        input: the text to judge (a conversation, a document, a ticket, ...)
        questions: list of questions, each one of
            {"type": "predicate", "name": "is_spam", "instructions": "Is this message spam?"}
            {"type": "choice", "name": "topic", "instructions": "...", "choices": [{"value": "billing",
             "description": "..."}, {"value": "bug"}]}
            {"type": "score", "name": "urgency", "instructions": "...", "levels": [{"label": "low"},
             {"label": "medium"}, {"label": "high"}]} (lowest first)
        model: a decision model id from list_models; empty = elephant-1
        images: pplx-decider-flash-multimodal only: images about the input ("att:N", ids, URLs)
    """
    model = await pick_model("decision", model)
    content = [{"type": "input_text", "text": input}]
    for i, ref in enumerate(images[:8]):
        data, _, ctype = await resolve_input(ref, ctx, "image", f"images[{i}]")
        content.append({"type": "input_image", "image_url": data_url(data, ctype or "image/png")})
    body = {"model": model, "input": [{"role": "user", "content": content}] if images else input,
            "questions": questions}
    async with httpx.AsyncClient(timeout=900) as c:
        r = await c.post(f"{LSW}/v1/decisions", json=body, headers=AUTH)
    if r.status_code != 200:
        return f"Error: decision failed (HTTP {r.status_code}): {r.text[:400]}"
    return json.dumps(r.json().get("answers", []), ensure_ascii=False)


# ---------------------------------------------------------------------------------------------------- files
@mcp.tool()
@friendly
async def list_attachments(ctx: Context = None) -> str:
    """Files in this chat (uploaded by the user or generated here), newest first, as att:1, att:2, ... for the
    file arguments of the other tools."""
    files = await chat_files(ctx)
    if files is None:
        return "This client passes no chat attachments; give files to the tools as URLs."
    if not files:
        return "No files in this chat."
    return "\n".join(f"att:{f['n']}  {f['kind']:5s}  {f['name'] or '(no name)'}  {f['content_type']}  "
                     f"({'uploaded by the user' if f['from'] == 'user' else 'generated'})" for f in files)


def _hexdump(data, base):
    rows = []
    for i in range(0, len(data), 16):
        chunk = data[i:i + 16]
        hx = " ".join(f"{b:02x}" for b in chunk)
        rows.append(f"{base + i:08x}  {hx:<47}  |{''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)}|")
    return "\n".join(rows)


def _entropy(data):
    if not data:
        return 0.0
    counts = [0] * 256
    for b in data:
        counts[b] += 1
    return -sum(c / len(data) * math.log2(c / len(data)) for c in counts if c)


def _structure(data, mime):
    """Short format-specific summary for common types (best effort)."""
    try:
        if mime == "application/pdf":
            from pypdf import PdfReader
            pdf = PdfReader(io.BytesIO(data))
            text = "\n".join((p.extract_text() or "") for p in pdf.pages[:3])
            return (f"PDF: {len(pdf.pages)} pages, metadata {dict(pdf.metadata or {})}\n"
                    f"text of the first pages:\n{text[:3000]}")
        if data[:2] == b"PK":
            import zipfile
            z = zipfile.ZipFile(io.BytesIO(data))
            items = z.infolist()
            return f"ZIP with {len(items)} entries:\n" + "\n".join(
                f"  {i.filename}  {i.file_size} B" for i in items[:60])
        if data[:16] == b"SQLite format 3\x00":
            import sqlite3
            with tempfile.NamedTemporaryFile(suffix=".db") as t:
                t.write(data)
                t.flush()
                rows = sqlite3.connect(t.name).execute("select type, name, sql from sqlite_master").fetchall()
            return "SQLite schema:\n" + "\n".join(f"  {r[0]} {r[1]}: {r[2]}" for r in rows[:60])
        if mime.startswith("image/"):
            from PIL import Image as PILImage
            im = PILImage.open(io.BytesIO(data))
            exif = {k: v for k, v in (im.getexif() or {}).items()} if hasattr(im, "getexif") else {}
            return f"image: {im.format} {im.size[0]}x{im.size[1]} mode {im.mode}, {len(exif)} EXIF tags"
        if mime.startswith(("audio/", "video/")):
            with tempfile.NamedTemporaryFile() as t:
                t.write(data)
                t.flush()
                p = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                    "format=format_name,duration,bit_rate:stream=codec_type,codec_name,width,height,"
                                    "sample_rate,channels,r_frame_rate", "-of", "json", t.name],
                                   capture_output=True, text=True, timeout=60)
            return "media: " + p.stdout.strip().replace("\n", " ")[:1500]
        if data[:2] == b"\x1f\x8b":
            import gzip
            import magic
            import tarfile
            inner = gzip.GzipFile(fileobj=io.BytesIO(data)).read(1 << 20)
            if inner[257:262] == b"ustar":
                with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as t:
                    return "tar.gz:\n" + "\n".join(f"  {m.name}  {m.size} B" for m in t.getmembers()[:60])
            kind = magic.from_buffer(inner[:65536])
            text = inner[:3000].decode("utf-8", errors="replace") if "text" in kind or "JSON" in kind else ""
            return f"gzip -> {len(inner)} B of: {kind}" + (f"\ndecompressed text:\n{text}" if text else "")
    except Exception as e:  # a malformed file is a finding, not a failure
        return f"({mime} parser: {e})"
    return ""


@mcp.tool()
@friendly
async def inspect_file(file: str = "", offset: int = 0, length: int = 256,
                       encodings: list[str] = ["utf-8", "utf-16-le", "gb18030", "big5", "shift_jis", "latin-1"],
                       ctx: Context = None) -> str:
    """Analyse any file, also unknown or binary ones: detected type (magic bytes, like the `file` command), size,
    SHA-256, entropy (high = compressed or encrypted), a hex dump of a byte range (plus the last bytes), that range
    decoded with several text encodings, printable strings, and a summary for common formats (PDF text and
    metadata, ZIP/Office listing, images, audio/video streams, SQLite schema, gzip). Call again with another offset
    to look further into the file.

    Args:
        file: "" = the newest attachment in this chat, "att:N", a media id, or a URL
        offset: first byte of the range to dump (negative = from the end)
        length: bytes to dump, up to 4096
        encodings: text encodings to try on the range
    """
    import magic
    data, name, declared = await resolve_input(file, ctx, "any", "file")
    size = len(data)
    start = max(0, size + offset if offset < 0 else min(offset, size))
    chunk = data[start:start + max(16, min(4096, length))]
    mime = magic.from_buffer(data[:65536], mime=True)
    out = [f"name: {name or '-'}   size: {size} B   declared type: {declared or '-'}",
           f"detected: {magic.from_buffer(data[:65536])}  ({mime})",
           f"sha256: {hashlib.sha256(data).hexdigest()}",
           f"entropy: {_entropy(data[:1 << 20]):.2f} bits/byte over the first {min(size, 1 << 20)} B (8 = random)",
           f"\nhex {start}..{start + len(chunk)}:\n{_hexdump(chunk, start)}"]
    if start == 0 and size > len(chunk):
        tail = data[-64:]
        out.append(f"\nlast {len(tail)} B:\n{_hexdump(tail, size - len(tail))}")
    out.append("\ndecoded range:")
    for enc in encodings[:8]:
        try:
            text = chunk.decode(enc, errors="replace")
        except LookupError:
            out.append(f"  {enc}: unknown encoding")
            continue
        bad = text.count("\ufffd") / max(1, len(text))
        shown = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "·", text[:200])
        out.append(f"  {enc} ({bad:.0%} undecodable): {shown!r}")
    strings = re.findall(rb"[\x20-\x7e]{6,}", data[:1 << 20])
    if strings:
        out.append("\nprintable strings (first 30): " + " | ".join(s.decode()[:80] for s in strings[:30]))
    summary = _structure(data, mime)
    if summary:
        out.append("\n" + summary)
    return "\n".join(out)


@mcp.tool()
@friendly
async def server_status() -> str:
    """What runs on the Mac Studio right now: loaded models and whether a video is rendering."""
    async with httpx.AsyncClient(timeout=30) as c:
        run = (await c.get(f"{LSW}/running", headers=AUTH)).json().get("running", [])
        vids = (await c.get(VIDEO_GET, headers=AUTH)).json().get("data", [])
    busy = [v for v in vids if v.get("status") in ("queued", "in_progress")]
    return json.dumps({"loaded_models": [m["model"] for m in run],
                       "rendering": [{"id": v["id"], "progress": v.get("progress"), "phase": v.get("phase")}
                                     for v in busy]})


# ---------------------------------------------------------------------------------------------------- health
@mcp.custom_route("/healthz", methods=["GET"])
async def healthz(request: Request):
    return PlainTextResponse("ok")


# ---------------------------------------------------------------------------------------------------- app
class BearerAuth:
    """ASGI middleware: /mcp needs Authorization: Bearer <one of MCP_TOKENS>; /healthz passes.
    The chat-context headers survive only for CONTEXT_TOKENS (Open WebUI), so other clients cannot read chats."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"].startswith("/mcp"):
            auth = dict(scope["headers"]).get(b"authorization", b"").decode()
            token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
            if not TOKENS or not any(secrets.compare_digest(token, t) for t in TOKENS):
                return await PlainTextResponse("unauthorized", status_code=401,
                                               headers={"WWW-Authenticate": "Bearer"})(scope, receive, send)
            if not any(secrets.compare_digest(token, t) for t in CONTEXT_TOKENS):
                scope = dict(scope, headers=[(k, v) for k, v in scope["headers"] if k.lower() not in CONTEXT_HEADERS])
        await self.app(scope, receive, send)


def build_app():
    hosts = [h for h in os.environ.get("ALLOWED_HOSTS", "llm-mcp:8000,localhost:8000,"
                                       "127.0.0.1:8000").split(",") if h]
    app = mcp.streamable_http_app(
        streamable_http_path="/mcp", host="0.0.0.0", max_request_body_size=64 << 20,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=True, allowed_hosts=hosts,
                                                     allowed_origins=[f"https://{h}" for h in hosts]))
    return BearerAuth(app)


app = build_app()

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), proxy_headers=True,
                forwarded_allow_ips="*")
