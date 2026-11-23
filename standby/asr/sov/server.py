"""OpenAI-compatible speech-to-text API for Salesforce SOV-Transcribe-1 via mlx-audio, meant to run behind llama-swap.

Endpoints:
  GET  /health                    readiness probe for llama-swap's checkEndpoint (200 once the model is loaded)
  GET  /v1/models                 lists the single model ID
  POST /v1/audio/transcriptions   multipart: file, model, language, prompt, response_format
                                  (json|text|srt|vtt|verbose_json), timestamp_granularities[] (word|segment),
                                  temperature (accepted, ignored: decoding is greedy)

SOV-Transcribe-1 transcribes English, French, German, Spanish, Italian, Portuguese and Dutch, up to 30 minutes per
request, as one pass over the whole recording (30 s encoder windows, one shared decoder context). It has no language
detection and no timestamps: `language` is passed to the model as a hint, and srt/vtt/verbose_json transcribe ~30 s
chunks (split at low-energy points) one by one and align each with Qwen3-ForcedAligner-0.6B (kept resident) for word
times. Chinese audio is not supported (it comes back translated or garbled; use qwen3-asr-1.7b), `language=zh`
is refused with 400. `prompt` is the vocabulary/context hint ("Contextual vocabulary: ..."), not the instruction.

One worker thread owns the MLX model (MLX streams are per thread); requests queue behind it.

Usage: .venv/bin/python standby/asr/sov/server.py --port PORT --model DIR --aligner DIR [--model-id ID]"""
import argparse
import concurrent.futures as cf
import logging
import math
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from http_api import SR, BadRequest, Handler, Server, ts

log = logging.getLogger("sov-server")

# ISO codes / names the model card lists; the hint is written into the instruction as the English name.
LANGUAGES = {"en": "English", "fr": "French", "de": "German", "es": "Spanish", "it": "Italian", "pt": "Portuguese",
             "nl": "Dutch"}
MAX_SECONDS = 1800


class Engine:
    def __init__(self, model_dir, aligner_dir, cache_mb):
        self.model_dir, self.aligner_dir, self.cache_mb = model_dir, aligner_dir, cache_mb
        self.ready = False
        self.pool = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
        self.pool.submit(self._load).result()

    def _load(self):
        import mlx.core as mx
        from mlx_audio.stt import load
        from mlx_qwen3_asr import ForcedAligner
        mx.set_cache_limit(self.cache_mb * 1024**2)
        t = time.time()
        self.model = load(self.model_dir)
        self.aligner = ForcedAligner(self.aligner_dir, dtype=mx.float16)
        self.aligner._ensure_loaded()  # resident: stable memory, first srt not slow
        self.ready = True
        log.info("loaded %s in %.1fs", self.model_dir, time.time() - t)

    def transcribe(self, audio, language, context, timestamps):
        if len(audio) > MAX_SECONDS * SR:
            raise BadRequest(f"audio longer than {MAX_SECONDS // 60} minutes")
        name = None
        if language:
            name = LANGUAGES.get(language.lower()) or next(
                (v for v in LANGUAGES.values() if v.lower() == language.lower()), None)
            if name is None:
                raise BadRequest("unsupported language; SOV-Transcribe-1 supports " + ", ".join(LANGUAGES))
        return self.pool.submit(self._run, audio, name, context, timestamps).result()

    def _generate(self, audio, language, context):
        # English speech gives ~3.3 tokens/s (6.6 at most on short clips); a budget of 64 + 10/s keeps a runaway
        # decode (unsupported language, e.g. Chinese) from running to 8192 tokens (~70 s) per request.
        budget = min(16384, 64 + math.ceil(10 * len(audio) / SR))
        r = self.model.generate(audio, language=language, hotwords=[context] if context else None, max_tokens=budget)
        if r.finish_reason != "stop":
            raise RuntimeError("decoder token budget exhausted; refusing an incomplete transcript")
        return r

    def _run(self, audio, language, context, timestamps):
        t = time.time()
        if not timestamps:
            r = self._generate(audio, language, context)
            log.info("transcribed %.1fs in %.2fs (prompt %d tok %.0f tok/s, %d generated %.1f tok/s)",
                     len(audio) / SR, time.time() - t, r.prompt_tokens, r.prompt_tps, r.generation_tokens,
                     r.generation_tps)
            return {"text": r.text, "language": language, "words": [], "chunks": []}
        from mlx_qwen3_asr.chunking import split_audio_into_chunks
        texts, words, chunks = [], [], []
        for piece, offset in split_audio_into_chunks(audio, SR):
            text = self._generate(piece, language, context).text
            if not text:
                continue
            texts.append(text)
            end = offset + len(piece) / SR
            chunks.append({"start": offset, "end": end, "text": text})
            try:
                aligned = self.aligner.align(piece, text, language or "English")
            except Exception:
                log.exception("alignment failed for the chunk at %.1fs; using chunk times", offset)
                continue
            words += [{"text": w.text, "start": offset + w.start_time, "end": offset + w.end_time} for w in aligned]
        log.info("transcribed %.1fs in %.2fs with timestamps (%d chunks, %d words)", len(audio) / SR,
                 time.time() - t, len(chunks), len(words))
        return {"text": " ".join(texts), "language": language, "words": words, "chunks": chunks}


def build_response(r, fmt, duration, words_wanted):
    from mlx_qwen3_asr.writers import group_subtitle_segments
    text = r["text"]
    if fmt == "text":
        return "text/plain; charset=utf-8", text + "\n"
    if fmt == "json":
        return "application/json", {"text": text}
    words = r["words"]
    cues = group_subtitle_segments(words, language=r["language"] or "English", text=text) if words else []
    if not cues:  # aligner produced nothing: fall back to chunk-level times
        cues = r["chunks"]
    if fmt == "srt":
        body = "".join(f"{i}\n{ts(c['start'], ',')} --> {ts(c['end'], ',')}\n{c['text'].strip()}\n\n"
                       for i, c in enumerate(cues, 1))
        return "text/plain; charset=utf-8", body
    if fmt == "vtt":
        body = "WEBVTT\n\n" + "".join(f"{ts(c['start'], '.')} --> {ts(c['end'], '.')}\n{c['text'].strip()}\n\n"
                                      for c in cues)
        return "text/vtt; charset=utf-8", body
    code = next((k for k, v in LANGUAGES.items() if v == r["language"]), None)
    out = {"task": "transcribe", "language": code, "duration": round(duration, 3), "text": text,
           "segments": [{"id": i, "seek": 0, "start": round(c["start"], 3), "end": round(c["end"], 3),
                         "text": c["text"], "tokens": [], "temperature": 0.0, "avg_logprob": None,
                         "compression_ratio": None, "no_speech_prob": None} for i, c in enumerate(cues)]}
    if words_wanted:
        out["words"] = [{"word": w["text"], "start": round(w["start"], 3), "end": round(w["end"], 3)} for w in words]
    return "application/json", out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--aligner", required=True)
    ap.add_argument("--model-id", default="sov-transcribe-1")
    ap.add_argument("--cache-mb", type=int, default=1024, help="MLX allocator cache limit")
    a = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    logging.basicConfig(level=logging.INFO, format="[sov] %(message)s")
    server = Server((a.host, a.port), Handler)  # bind first so llama-swap sees the port while loading
    server.model_id = a.model_id
    server.response_builder = build_response
    server.engine = type("E", (), {"ready": False})()
    threading.Thread(target=lambda: setattr(server, "engine", Engine(a.model, a.aligner, a.cache_mb)),
                     daemon=True).start()
    log.info("listening on http://%s:%d (model %s)", a.host, a.port, a.model)
    server.serve_forever()


if __name__ == "__main__":
    main()
