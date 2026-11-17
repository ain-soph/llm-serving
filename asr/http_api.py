"""Shared local OpenAI-compatible audio HTTP transport; inference belongs to each engine."""
import email.parser
import email.policy
import json
import logging
import os
import socketserver
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np

log = logging.getLogger("asr-http")
SR = 16000
FORMATS = ("json", "text", "srt", "vtt", "verbose_json")

class BadRequest(Exception):
    pass

def decode_audio(raw: bytes) -> np.ndarray:
    """Any container ffmpeg reads -> 16 kHz mono float32."""
    p = subprocess.run([os.environ.get("FFMPEG", "ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
                        "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"],
                       input=raw, capture_output=True)
    if p.returncode != 0 or not p.stdout:
        raise BadRequest("could not decode the audio file: " + p.stderr.decode(errors="replace")[-300:])
    return np.frombuffer(p.stdout, dtype=np.float32).copy()

def ts(seconds, sep):
    ms = int(round(max(0.0, seconds) * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"

class Server(ThreadingHTTPServer):
    daemon_threads = True

    def server_bind(self):
        # HTTPServer.server_bind calls socket.getfqdn(), which can block ~35 s on this Mac; the name is unused.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]

class Handler(BaseHTTPRequestHandler):
    server_version = "local-asr-server"
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
        if self.path == "/health":
            ready = self.server.engine.ready
            self._send(200 if ready else 503, "application/json", {"status": "ok" if ready else "loading"})
        elif self.path == "/v1/models":
            self._send(200, "application/json", {"object": "list", "data": [
                {"id": self.server.model_id, "object": "model", "owned_by": "local"}]})
        else:
            self._error(404, f"unknown path {self.path}")

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        if self.path != "/v1/audio/transcriptions":
            return self._error(404, f"unknown path {self.path}")
        try:
            self._transcriptions(body)
        except BadRequest as e:
            self._error(400, str(e))
        except Exception as e:
            log.exception("transcription failed")
            self._error(500, f"transcription failed: {e}", "server_error")

    def _transcriptions(self, body):
        if not self.server.engine.ready:
            return self._error(503, "model is still loading", "server_error")
        ctype = self.headers.get("Content-Type", "")
        if not ctype.startswith("multipart/form-data"):
            raise BadRequest("transcriptions require multipart/form-data")
        msg = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
            f"Content-Type: {ctype}\r\n\r\n".encode() + body)
        fields, granularities, audio_raw = {}, [], None
        for part in msg.iter_parts():
            name = part.get_param("name", header="content-disposition")
            if name == "file":
                audio_raw = part.get_payload(decode=True)
            elif name in ("timestamp_granularities[]", "timestamp_granularities"):
                granularities.append(part.get_payload(decode=True).decode().strip())
            elif name:
                fields[name] = part.get_payload(decode=True).decode(errors="replace").strip()
        if not audio_raw:
            raise BadRequest("file is required")
        fmt = (fields.get("response_format") or "json").lower()
        if fmt not in FORMATS:
            raise BadRequest(f"response_format must be one of {', '.join(FORMATS)}")
        language = fields.get("language") or None
        if language and language.lower() == "auto":
            language = None
        audio = decode_audio(audio_raw)
        duration = len(audio) / SR
        if duration < 0.05:
            raise BadRequest("audio is too short")
        timestamps = fmt in ("srt", "vtt", "verbose_json")
        r = self.server.engine.transcribe(audio, language, fields.get("prompt") or "", timestamps)
        ctype_out, payload = self.server.response_builder(r, fmt, duration, "word" in granularities)
        self._send(200, ctype_out, payload)

    def log_message(self, fmt, *args):
        log.info("%s %s", self.address_string(), fmt % args)
