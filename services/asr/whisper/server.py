"""Whisper large-v3 MLX behind the shared local OpenAI audio HTTP API."""
import argparse
import concurrent.futures as cf
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from http_api import SR, Handler, Server, ts

log = logging.getLogger("whisper-mlx-server")


class Engine:
    def __init__(self, path):
        self.ready = False
        self.path = path
        self.pool = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper-mlx")
        self.pool.submit(self._load).result()

    def _load(self):
        import mlx.core as mx
        from mlx_whisper.transcribe import ModelHolder
        mx.set_cache_limit(512 * 1024**2)
        t = time.time()
        model = ModelHolder.get_model(self.path, mx.float16)
        mx.eval(model.parameters())
        self.ready = True
        log.info("loaded %s in %.1fs", self.path, time.time() - t)

    def transcribe(self, audio, language, context, timestamps):
        return self.pool.submit(self._run, audio, language, context, timestamps).result()

    def _run(self, audio, language, context, timestamps):
        import mlx_whisper
        from mlx_whisper.tokenizer import LANGUAGES
        if language:
            language = language.lower()
            language = {v.lower(): k for k, v in LANGUAGES.items()}.get(language, language)
            if language not in LANGUAGES:
                from http_api import BadRequest
                raise BadRequest("unsupported language")
        t = time.time()
        r = mlx_whisper.transcribe(audio, path_or_hf_repo=self.path, language=language,
                                  initial_prompt=context or None, word_timestamps=timestamps,
                                  condition_on_previous_text=False, temperature=0.0, verbose=None)
        log.info("transcribed %.1fs in %.2fs (lang=%s, timestamps=%s)",
                 len(audio) / SR, time.time() - t, r.get("language"), timestamps)
        return r


def build_response(r, fmt, duration, words_wanted):
    if fmt == "text":
        return "text/plain; charset=utf-8", r["text"].strip() + "\n"
    if fmt == "json":
        return "application/json", {"text": r["text"].strip()}
    segments = r.get("segments") or []
    if fmt in ("srt", "vtt"):
        body = "WEBVTT\n\n" if fmt == "vtt" else ""
        sep = "." if fmt == "vtt" else ","
        for i, seg in enumerate(segments, 1):
            if fmt == "srt":
                body += f"{i}\n"
            body += f"{ts(seg['start'], sep)} --> {ts(seg['end'], sep)}\n{seg['text'].strip()}\n\n"
        return "text/vtt; charset=utf-8" if fmt == "vtt" else "text/plain; charset=utf-8", body
    result = {"task": "transcribe", "language": r.get("language"), "duration": round(duration, 3),
              "text": r["text"].strip(), "segments": segments}
    if words_wanted:
        result["words"] = [w for seg in segments for w in seg.get("words", [])]
    return "application/json", result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--model", default=str(Path.home() / "models/whisper-large-v3-mlx"))
    p.add_argument("--model-id", default="whisper-large-v3")
    a = p.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    logging.basicConfig(level=logging.INFO, format="[whisper-mlx] %(message)s")
    server = Server((a.host, a.port), Handler)
    server.model_id = a.model_id
    server.response_builder = build_response
    server.engine = type("Loading", (), {"ready": False})()
    import threading
    threading.Thread(target=lambda: setattr(server, "engine", Engine(a.model)), daemon=True).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
