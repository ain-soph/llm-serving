"""OpenAI-compatible text-to-speech API for Qwen3-TTS via mlx-audio, meant to run behind llama-swap.

Endpoints:
  GET  /health            readiness probe for llama-swap's checkEndpoint (200 once the model is loaded)
  GET  /v1/models         lists the single model ID
  GET  /v1/audio/voices   preset voices (CustomVoice) and the OpenAI voice names mapped onto them
  POST /v1/audio/speech   JSON: model, input, voice, instructions, response_format (mp3|opus|aac|flac|wav|pcm),
                          speed (0.25-4.0), stream_format (audio|sse); extensions: language (auto|chinese|english|...),
                          ref_audio (base64 or data: URL, any format ffmpeg reads) + ref_text (Base / clone model),
                          seed (int, reproducible sampling), temperature (0 = greedy)

The model variant comes from the checkpoint (config.json tts_model_type):
  custom_voice  `voice` picks one of the 9 preset speakers (OpenAI names are mapped); `instructions` sets
                emotion / style.
  voice_design  `instructions` describes the voice to create (required); `voice` is ignored.
  base          voice cloning: `ref_audio` (3-20 s) with its exact transcript `ref_text`; `voice` is ignored.
CosyVoice3 (Fun-CosyVoice3-0.5B via mlx-audio-plus, run from tts/cosyvoice/.venv with --voices):
  `voice` picks a built-in prompt voice (voices.json; OpenAI names map to the first), or send `ref_audio`
  (<= 30 s) with its exact transcript `ref_text` (zero-shot) or without it (cross-lingual);
  `instructions` switches to instruct mode (dialect, emotion, speed, e.g. "请用四川话表达。"), ref_text is then unused.
Fish Audio Drama 3 (config.json model_type fish_drama3, mlx-audio fish_drama3, 2026-11-23):
  the whole input goes to the model, which splits it itself and keeps the first and previous chunk as history (one
  voice across a long text). `ref_audio` + `ref_text` (both required) clone a voice; otherwise `voice` picks a
  built-in prompt voice from --voices (zh / en), and anything else (OpenAI names, none) lets the model choose a
  voice (it varies with the seed). Delivery cues go inline in the text ("[whispers]", "[laughs]"); `instructions`
  is refused. `<|speaker:N|>` tags (0-15) switch speakers. Native rate 44.1 kHz (wav / mp3 / ...); pcm is
  resampled to 24 kHz as OpenAI defines it.

Audio is streamed as it is generated (chunked transfer; pcm = 24 kHz mono s16le). Long input is split into
paragraphs and sentences of at most --max-chars characters, generated one after another; each piece stops after
50 + 6 x its length frames (12.5 frames/s), which bounds a degenerate generation. Inference runs one request
at a time on a single worker thread that owns the MLX model (MLX streams are per thread). `speed` != 1 is applied
with ffmpeg atempo (pitch kept).

Usage: .venv/bin/python services/tts/server.py --port PORT --model DIR --model-id ID [--cache-mb 512] [--max-chars 220]
       (the CosyVoice3 branch, --voices / --s3tokenizer, is kept for a possible return; the entry is retired: docs/tts.md)"""
import argparse
import base64
import concurrent.futures as cf
import json
import logging
import os
import queue
import re
import socketserver
import struct
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

log = logging.getLogger("tts-server")
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
SR = 24000
MAX_INPUT = 20000
CTYPES = {"mp3": "audio/mpeg", "opus": "audio/ogg", "aac": "audio/aac", "flac": "audio/flac", "wav": "audio/wav",
          "pcm": "audio/pcm"}
ENCODE = {"mp3": ["-f", "mp3", "-b:a", "128k"], "opus": ["-f", "ogg", "-c:a", "libopus", "-b:a", "64k"],
          "aac": ["-f", "adts", "-c:a", "aac", "-b:a", "128k"], "flac": ["-f", "flac"],
          "wav": ["-f", "wav"], "pcm": ["-f", "s16le"]}
# OpenAI voice names -> preset speakers (female: Vivian / Serena, male: Ryan / Aiden / Uncle_Fu)
OPENAI_VOICES = {"alloy": "vivian", "coral": "vivian", "marin": "vivian", "nova": "serena", "shimmer": "serena",
                 "sage": "serena", "echo": "ryan", "ash": "ryan", "verse": "ryan", "fable": "aiden",
                 "ballad": "aiden", "cedar": "aiden", "onyx": "uncle_fu"}
LANGUAGES = {"zh": "chinese", "en": "english", "de": "german", "it": "italian", "pt": "portuguese", "es": "spanish",
             "ja": "japanese", "ko": "korean", "fr": "french", "ru": "russian"}


class BadRequest(Exception):
    pass


def decode_audio(raw: bytes, sr: int = SR) -> np.ndarray:
    """Any container ffmpeg reads -> mono float32 at `sr` (24 kHz by default)."""
    p = subprocess.run([FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-ac", "1",
                        "-ar", str(sr), "-f", "f32le", "pipe:1"], input=raw, capture_output=True)
    if p.returncode != 0 or not p.stdout:
        raise BadRequest("could not decode ref_audio: " + p.stderr.decode(errors="replace")[-300:])
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def atempo(speed: float) -> str:
    """atempo takes 0.5-2.0 per stage; chain stages for 0.25-4.0."""
    stages = []
    while speed > 2.0:
        stages.append(2.0)
        speed /= 2.0
    while speed < 0.5:
        stages.append(0.5)
        speed /= 0.5
    stages.append(speed)
    return ",".join(f"atempo={s:.6f}" for s in stages)


_SENTENCE = re.compile(r"(?<=[。！？!?；;…])|(?<=[.](?=\s))")
_CLAUSE = re.compile(r"(?<=[，,、：:])")


def split_text(text: str, limit: int) -> list[str]:
    """Paragraphs, then sentences packed up to `limit` characters; an over-long sentence is cut at commas."""
    pieces = []
    for para in [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]:
        if len(para) <= limit:
            pieces.append(para)
            continue
        units = []
        for sent in [s for s in _SENTENCE.split(para) if s.strip()]:
            if len(sent) <= limit:
                units.append(sent)
                continue
            for clause in [c for c in _CLAUSE.split(sent) if c.strip()]:
                while len(clause) > limit:
                    units.append(clause[:limit])
                    clause = clause[limit:]
                units.append(clause)
        current = ""
        for unit in units:
            if current and len(current) + len(unit) > limit:
                pieces.append(current.strip())
                current = ""
            current += unit
        if current.strip():
            pieces.append(current.strip())
    return pieces


class Engine:
    def __init__(self, model_dir: str, cache_mb: int, max_chars: int, voices: str = None, s3tokenizer: str = None):
        self.model_dir, self.cache_mb, self.max_chars = model_dir, cache_mb, max_chars
        self.voices_file, self.s3tokenizer = voices, s3tokenizer
        self.ready = False
        self.sr = SR
        self.pool = cf.ThreadPoolExecutor(1)
        self.pool.submit(self._load).result()

    def _load(self):
        import mlx.core as mx

        if self.s3tokenizer:  # CosyVoice3's speech tokenizer is fetched from the Hub unless pointed at a local copy
            import mlx_audio.codec.models.s3tokenizer.model_v3 as s3v3
            s3v3.fetch_from_hub = lambda repo: Path(self.s3tokenizer)
        from mlx_audio.tts.utils import load_model

        mx.set_cache_limit(self.cache_mb << 20)
        t = time.time()
        self.model = load_model(self.model_dir)
        if getattr(self.model.config, "model_type", None) == "cosyvoice3":
            self.kind = "cosyvoice3"
            spec = json.loads(Path(self.voices_file).read_text())
            self.prompts = {name: (decode_audio((Path(self.voices_file).parent / v["wav"]).read_bytes()), v["text"])
                            for name, v in spec.items()}
            self.speakers = list(self.prompts)
        elif getattr(self.model.config, "model_type", None) == "fish_drama3":
            self.kind = "fish_drama3"
            self.sr = int(self.model.sample_rate)
            self.prompts = {}
            if self.voices_file:
                spec = json.loads(Path(self.voices_file).read_text())
                self.prompts = {name: (decode_audio((Path(self.voices_file).parent / v["wav"]).read_bytes(), self.sr),
                                       v["text"]) for name, v in spec.items()}
            self.speakers = list(self.prompts)
        else:
            self.kind = getattr(self.model.config, "tts_model_type", "base")
            self.speakers = sorted(s.lower() for s in (self.model.supported_speakers or []))
        mx.clear_cache()
        log.info("loaded %s (%s) in %.1f s, speakers %s", self.model_dir, self.kind, time.time() - t, self.speakers)
        self.ready = True

    def options(self, req: dict) -> dict:
        text = req.get("input")
        if not isinstance(text, str) or not text.strip():
            raise BadRequest("input is required")
        if len(text) > MAX_INPUT:
            raise BadRequest(f"input is longer than {MAX_INPUT} characters")
        opts = {"pieces": split_text(text, self.max_chars), "instruct": (req.get("instructions") or "").strip() or None}
        if self.kind == "fish_drama3":
            return self._drama_options(req, text, opts)
        lang = str(req.get("language") or "auto").lower()
        opts["language"] = LANGUAGES.get(lang, lang)
        if self.kind == "custom_voice":
            voice = req.get("voice") or "vivian"
            if isinstance(voice, dict):
                voice = voice.get("id") or voice.get("name") or "vivian"
            voice = str(voice).lower()
            voice = OPENAI_VOICES.get(voice, voice)
            if voice not in self.speakers:
                raise BadRequest(f"unknown voice {voice!r}; presets: {', '.join(self.speakers)}; "
                                 f"OpenAI names: {', '.join(OPENAI_VOICES)}")
            opts["voice"] = voice
        elif self.kind == "voice_design":
            if not opts["instruct"]:
                raise BadRequest("this model creates a voice from `instructions` (a description of the voice)")
        elif self.kind == "cosyvoice3":
            if req.get("ref_audio"):
                ref = req["ref_audio"]
                if ref.startswith("data:"):
                    ref = ref.split(",", 1)[1]
                try:
                    audio = decode_audio(base64.b64decode(ref, validate=False))
                except ValueError:
                    raise BadRequest("ref_audio is not valid base64")
                if not 1.0 <= len(audio) / SR <= 30.0:
                    raise BadRequest("ref_audio must be 1-30 s long")
                opts["ref_audio"], opts["ref_text"] = audio, (req.get("ref_text") or "").strip() or None
            else:
                voice = req.get("voice") or self.speakers[0]
                if isinstance(voice, dict):
                    voice = voice.get("id") or voice.get("name") or self.speakers[0]
                voice = str(voice).lower()
                if voice in OPENAI_VOICES:
                    voice = self.speakers[0]
                if voice not in self.prompts:
                    raise BadRequest(f"unknown voice {voice!r}; voices: {', '.join(self.speakers)} "
                                     "(or send ref_audio, optionally with ref_text)")
                opts["ref_audio"], opts["ref_text"] = self.prompts[voice]
            if opts["instruct"]:
                opts["ref_text"] = None
        else:
            ref, ref_text = req.get("ref_audio"), (req.get("ref_text") or "").strip()
            if not ref or not ref_text:
                raise BadRequest("this model clones a voice: send ref_audio (base64 / data URL) and ref_text")
            if ref.startswith("data:"):
                ref = ref.split(",", 1)[1]
            try:
                raw = base64.b64decode(ref, validate=False)
            except ValueError:
                raise BadRequest("ref_audio is not valid base64")
            audio = decode_audio(raw)
            if not 1.0 <= len(audio) / SR <= 60.0:
                raise BadRequest("ref_audio must be 1-60 s long (3-20 s works best)")
            opts["ref_audio"], opts["ref_text"] = audio, ref_text
        speed = float(req.get("speed") or 1.0)
        if not 0.25 <= speed <= 4.0:
            raise BadRequest("speed must be between 0.25 and 4.0")
        opts["speed"] = speed
        if req.get("temperature") is not None:
            opts["temperature"] = float(req["temperature"])
        if req.get("seed") is not None:
            opts["seed"] = int(req["seed"])
        return opts

    def _drama_options(self, req: dict, text: str, opts: dict) -> dict:
        if opts["instruct"]:
            raise BadRequest("Drama 3 takes delivery cues inline in the text, e.g. [whispers], not `instructions`")
        opts["pieces"] = [text.strip()]  # the model chunks long text itself, keeping one voice
        opts["ref_audio"] = opts["ref_text"] = None
        if req.get("ref_audio"):
            ref, ref_text = req["ref_audio"], (req.get("ref_text") or "").strip()
            if not ref_text:
                raise BadRequest("ref_audio needs its exact transcript in ref_text")
            if ref.startswith("data:"):
                ref = ref.split(",", 1)[1]
            try:
                audio = decode_audio(base64.b64decode(ref, validate=False), self.sr)
            except ValueError:
                raise BadRequest("ref_audio is not valid base64")
            if not 1.0 <= len(audio) / self.sr <= 60.0:
                raise BadRequest("ref_audio must be 1-60 s long")
            opts["ref_audio"], opts["ref_text"] = audio, ref_text
        else:
            voice = req.get("voice")
            if isinstance(voice, dict):
                voice = voice.get("id") or voice.get("name")
            voice = str(voice or "").lower()
            if voice in self.prompts:
                opts["ref_audio"], opts["ref_text"] = self.prompts[voice]
            elif voice not in ("", "none", "auto") and voice not in OPENAI_VOICES:
                raise BadRequest(f"unknown voice {voice!r}; built-in: {', '.join(self.speakers) or 'none'}; "
                                 "or send ref_audio + ref_text; OpenAI names and none let the model choose")
        speed = float(req.get("speed") or 1.0)
        if not 0.25 <= speed <= 4.0:
            raise BadRequest("speed must be between 0.25 and 4.0")
        opts["speed"] = speed
        if req.get("temperature") is not None:
            opts["temperature"] = float(req["temperature"])
        if req.get("seed") is not None:
            opts["seed"] = int(req["seed"])
        return opts

    def stream(self, opts: dict, out: "queue.Queue"):
        """Runs on the worker thread: puts float32 chunks, then None (or an Exception)."""
        import mlx.core as mx

        try:
            if "seed" in opts:
                mx.random.seed(opts["seed"])
            kw = {"stream": True, "streaming_interval": 0.32}
            if "temperature" in opts:
                kw["temperature"] = opts["temperature"]
            if self.kind == "fish_drama3":
                kw.pop("max_tokens", None)
                for r in self.model.generate(text=opts["pieces"][0], ref_audio=opts["ref_audio"],
                                             ref_text=opts["ref_text"], **kw):
                    out.put(np.array(r.audio, dtype=np.float32))
                out.put(None)
                return
            for piece in opts["pieces"]:
                # 12.5 codec frames/s; normal speech needs <= 3 frames per CJK character and < 1 per Latin letter.
                # The cap stops a degenerate generation (e.g. a clone prompt whose transcript does not match the
                # audio) after a bounded length instead of the 8192-frame (11 min) default.
                kw["max_tokens"] = 50 + 6 * len(piece)
                if self.kind == "custom_voice":
                    gen = self.model.generate_custom_voice(text=piece, speaker=opts["voice"],
                                                           language=opts["language"], instruct=opts["instruct"], **kw)
                elif self.kind == "voice_design":
                    gen = self.model.generate_voice_design(text=piece, instruct=opts["instruct"],
                                                           language=opts["language"], **kw)
                elif self.kind == "cosyvoice3":
                    # Speech tokens are capped at 20 x the piece's text tokens inside the model (as upstream), and
                    # the piece is split into sentences there; one result per sentence.
                    gen = self.model.generate(text=piece, ref_audio=mx.array(opts["ref_audio"]),
                                              ref_text=opts["ref_text"], instruct_text=opts["instruct"], verbose=False)
                else:
                    gen = self.model.generate(text=piece, ref_audio=mx.array(opts["ref_audio"]),
                                              ref_text=opts["ref_text"], lang_code=opts["language"], **kw)
                for r in gen:
                    out.put(np.array(r.audio, dtype=np.float32))
            out.put(None)
        except Exception as e:
            log.exception("generation failed")
            out.put(e)
        finally:
            mx.clear_cache()


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def server_bind(self):
        # HTTPServer.server_bind calls socket.getfqdn(), which can block ~35 s on this Mac; the name is unused.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


def wav_header(sr: int = SR) -> bytes:
    """Streaming WAV header (sizes unknown: 0xFFFFFFFF, as ffmpeg and OpenAI do)."""
    return (b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
            + b"data" + struct.pack("<I", 0xFFFFFFFF))


def s16(chunk: np.ndarray) -> bytes:
    return (np.clip(chunk, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


class Handler(BaseHTTPRequestHandler):
    server_version = "local-tts-server"
    protocol_version = "HTTP/1.1"

    def _send(self, status, ctype, body):
        if not isinstance(body, (bytes, str)):
            body = json.dumps(body, ensure_ascii=False)
        if isinstance(body, str):
            body = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, message, kind="invalid_request_error"):
        self._send(status, "application/json", {"error": {"message": message, "type": kind}})

    def do_GET(self):
        engine = self.server.engine
        path = self.path.split("?", 1)[0]  # llama-swap routes GETs by ?model=
        if path == "/health":
            self._send(200 if engine.ready else 503, "application/json",
                       {"status": "ok" if engine.ready else "loading"})
        elif path == "/v1/models":
            self._send(200, "application/json", {"object": "list", "data": [
                {"id": self.server.model_id, "object": "model", "owned_by": "local"}]})
        elif path == "/v1/audio/voices":
            mapped = ({k: engine.speakers[0] for k in OPENAI_VOICES} if engine.kind == "cosyvoice3"
                      else {k: "(model's choice)" for k in OPENAI_VOICES} if engine.kind == "fish_drama3"
                      else OPENAI_VOICES if engine.speakers else {})
            self._send(200, "application/json", {"kind": engine.kind, "voices": engine.speakers,
                                                 "openai_voices": mapped})
        else:
            self._error(404, f"unknown path {self.path}")

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if self.path != "/v1/audio/speech":
            return self._error(404, f"unknown path {self.path}")
        try:
            req = json.loads(body or b"{}")
            if not isinstance(req, dict):
                raise BadRequest("body must be a JSON object")
            fmt = str(req.get("response_format") or "mp3").lower()
            if fmt not in CTYPES:
                raise BadRequest(f"response_format must be one of {', '.join(CTYPES)}")
            sse = str(req.get("stream_format") or "audio").lower() == "sse"
            opts = self.server.engine.options(req)
        except (BadRequest, ValueError, TypeError) as e:
            return self._error(400, str(e))
        if not self.server.engine.ready:
            return self._error(503, "model is still loading", "server_error")
        self._speech(opts, fmt, sse)

    def _speech(self, opts, fmt, sse):
        chunks = queue.Queue()
        t0 = time.time()
        self.server.engine.pool.submit(self.server.engine.stream, opts, chunks)
        first = chunks.get()
        if isinstance(first, Exception):
            return self._error(500, f"generation failed: {first}", "server_error")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream" if sse else CTYPES[fmt])
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("X-First-Chunk-Seconds", f"{time.time() - t0:.3f}")
        self.end_headers()

        def write(data: bytes):
            if data:
                self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
                self.wfile.flush()

        def emit(data: bytes):
            if sse:
                event = {"type": "speech.audio.delta", "audio": base64.b64encode(data).decode()}
                write(f"data: {json.dumps(event)}\n\n".encode())
            else:
                write(data)

        sr = self.server.engine.sr
        direct = opts["speed"] == 1.0 and (fmt == "wav" or (fmt == "pcm" and sr == SR))
        encoder = reader = None
        if not direct:
            args = [FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "s16le", "-ar", str(sr),
                    "-ac", "1", "-i", "pipe:0"]
            if opts["speed"] != 1.0:
                args += ["-filter:a", atempo(opts["speed"])]
            if fmt == "pcm" and sr != SR:
                args += ["-ar", str(SR)]  # OpenAI pcm is 24 kHz
            encoder = subprocess.Popen(args + ENCODE[fmt] + ["pipe:1"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

            def pump():
                gone = False
                while data := encoder.stdout.read1(16384):
                    if gone:
                        continue  # keep draining so ffmpeg never blocks
                    try:
                        emit(data)
                    except (BrokenPipeError, ConnectionResetError):
                        gone = True
            reader = threading.Thread(target=pump, daemon=True)
            reader.start()
        elif fmt == "wav":
            emit(wav_header(sr))
        samples, chunk, failed = 0, first, None
        try:
            while chunk is not None:
                if isinstance(chunk, Exception):
                    failed = chunk
                    break
                samples += len(chunk)
                if direct:
                    emit(s16(chunk))
                else:
                    encoder.stdin.write(s16(chunk))
                    encoder.stdin.flush()
                chunk = chunks.get()
        except (BrokenPipeError, ConnectionResetError):
            failed = "client disconnected"
            while chunk is not None and not isinstance(chunk, Exception):
                chunk = chunks.get()  # let the worker finish this request
        finally:
            if encoder is not None:
                encoder.stdin.close()
                reader.join()
                encoder.wait()
        try:
            if sse and failed is None:
                write(f"data: {json.dumps({'type': 'speech.audio.done'})}\n\n".encode())
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        log.info("speech: %d pieces, %.2f s audio in %.2f s%s", len(opts["pieces"]), samples / sr, time.time() - t0,
                 f" FAILED: {failed}" if failed else "")

    def log_message(self, fmt, *args):
        log.info("%s %s", self.address_string(), fmt % args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--cache-mb", type=int, default=512)
    ap.add_argument("--max-chars", type=int, default=220)
    ap.add_argument("--voices", help="CosyVoice3 / Drama 3: JSON of built-in prompt voices {name: {wav, text}}")
    ap.add_argument("--s3tokenizer", help="CosyVoice3: local speech_tokenizer_v3 MLX directory")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", stream=sys.stderr)
    server = Server((a.host, a.port), Handler)
    server.model_id = a.model_id
    server.engine = Engine(a.model, a.cache_mb, a.max_chars, a.voices, a.s3tokenizer)
    log.info("serving %s on %s:%d", a.model_id, a.host, a.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
