"""OpenAI-compatible speech-to-text API for Qwen3-ASR via mlx-qwen3-asr, meant to run behind llama-swap.

Endpoints:
  GET  /health                    readiness probe for llama-swap's checkEndpoint (200 once the model is loaded)
  GET  /v1/models                 lists the single model ID
  POST /v1/audio/transcriptions   multipart: file, model, language, prompt, response_format
                                  (json|text|srt|vtt|verbose_json), timestamp_granularities[] (word|segment),
                                  temperature (accepted, ignored: decoding is greedy)

Requests block until the transcript is ready; inference runs one request at a time on a single worker thread
that owns the MLX model (MLX state should not be shared across threads). Long audio is split by the library into
~30 s chunks at low-energy points. srt/vtt/verbose_json run the Qwen3-ForcedAligner-0.6B (preloaded and
kept resident) for word timestamps; segments are subtitle-style cues grouped from the words.
`prompt` is passed to Qwen3-ASR as its context (domain terms / hotwords).

--q8: the text decoder's linear layers and lm_head become 8-bit weights with float16 activations (W8A16, Q8_0-style
blocks of 32; services/asr/w8a8/asr_quant.py, converted at load). Decoding reads every weight once per token, so
this is 1.27-1.31x faster end to end with the same error rates (reports/asr.md); the audio encoder stays float16
(W8A8 gave no speed-up there). Used by the qwen3-asr-1.7b-int8 entry; the default entry stays exact.

--backend mlx-audio (2026-11-21): run the same weights on mlx-audio's qwen3_asr / qwen3_forced_aligner through
mlxaudio_engine.py (same chunking, language, token-budget and timestamp semantics).
Default stays mlx-qwen3-asr. --dtype / --q8 apply to mlx-qwen3-asr only
(mlx-audio keeps the stored float16 / 8-bit layers). With mlx-audio:
  --fast     one-pass prefill, pipelined greedy loop, vectorized prompt merge, fast aligner call (4.5-10 % faster
             than mlx-qwen3-asr on every test set; same transcripts as mlx-audio's own loop on 12 of 13 sets)
  --batch N  decode up to N chunks of one long request together (needs --fast; rows grouped by length, per-row end
             token / repetition stop / token budget): 2.3-2.9x on 12-16 min audio at N=8, MLX peak 4.20 -> 4.59 GB

Usage: .venv/bin/python server.py --port PORT [--model DIR] [--aligner DIR] [--model-id ID] [--dtype float16] [--q8]
       [--backend mlx-qwen3-asr|mlx-audio] [--fast] [--batch N]"""
import argparse
import concurrent.futures as cf
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # mlxaudio_engine.py
from http_api import SR, Server, Handler, ts

log = logging.getLogger("qwen3-asr-server")

class Engine:
    """Owns the MLX model; every call runs on one dedicated thread."""

    def __init__(self, model_dir, aligner_dir, dtype, q8=False, backend="mlx-qwen3-asr", fast=False, batch=1):
        self.pool = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
        self.model_dir, self.aligner_dir, self.dtype_name, self.q8 = model_dir, aligner_dir, dtype, q8
        self.backend, self.fast, self.batch = backend, fast, batch
        self.session = self.aligner = self.mlxaudio = None
        self.ready = False
        self.pool.submit(self._load).result()

    def _load(self):
        import mlx.core as mx
        from mlx_qwen3_asr import ForcedAligner, Session
        # Bound reusable temporaries so long timestamp requests do not retain
        # several GiB of allocator cache beside the resident model/aligner.
        mx.set_cache_limit(512 * 1024**2)
        t = time.time()
        if self.backend == "mlx-audio":
            from mlxaudio_engine import MlxAudioQwen3
            self.mlxaudio = MlxAudioQwen3(self.model_dir, self.aligner_dir, fast=self.fast, batch=self.batch)
            self.ready = True
            log.info("loaded %s + %s on mlx-audio (fast=%s, batch=%d) in %.1fs", self.model_dir, self.aligner_dir,
                     self.fast, self.batch, time.time() - t)
            return
        self.session = Session(self.model_dir, dtype=getattr(mx, self.dtype_name))
        if self.q8:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "w8a8"))
            import asr_quant
            counts = asr_quant.convert(self.session.model, asr_quant.plan(asr_quant.QWEN, "dec_q8+dec_q8_head"),
                                       getattr(mx, self.dtype_name))
            log.info("W8A16 text decoder: %s", counts)
        self.aligner = ForcedAligner(self.aligner_dir, dtype=getattr(mx, self.dtype_name))
        self.aligner._ensure_loaded()  # keep it resident so memory is stable and the first srt is not slow
        self.ready = True
        log.info("loaded %s (%s) in %.1fs", self.model_dir, self.dtype_name, time.time() - t)

    def transcribe(self, audio, language, context, timestamps):
        return self.pool.submit(self._run, audio, language, context, timestamps).result()

    def _run(self, audio, language, context, timestamps):
        t = time.time()
        if self.mlxaudio is not None:
            r = self.mlxaudio.transcribe(audio, language=language, context=context, timestamps=timestamps)
        else:
            r = self.session.transcribe(audio, language=language, context=context, return_timestamps=timestamps,
                                        return_chunks=True, forced_aligner=self.aligner if timestamps else None)
        if r.truncated:
            raise RuntimeError("decoder token budget exhausted; refusing an incomplete transcript")
        log.info("transcribed %.1fs audio in %.2fs (lang=%s, timestamps=%s)", len(audio) / SR, time.time() - t,
                 r.language, timestamps)
        return r

def build_response(r, fmt, duration, words_wanted):
    from mlx_qwen3_asr.writers import group_subtitle_segments
    text = r.text
    if fmt == "text":
        return "text/plain; charset=utf-8", text + "\n"
    if fmt == "json":
        return "application/json", {"text": text}
    words = r.segments or []
    cues = group_subtitle_segments(words, language=r.language or "", text=text) if words else []
    if not cues and r.chunks:  # aligner failed or produced nothing: fall back to chunk-level times
        cues = [{"start": c["start"], "end": c["end"], "text": c["text"]} for c in r.chunks if c.get("text")]
    if fmt == "srt":
        body = "".join(f"{i}\n{ts(c['start'], ',')} --> {ts(c['end'], ',')}\n{c['text'].strip()}\n\n"
                       for i, c in enumerate(cues, 1))
        return "text/plain; charset=utf-8", body
    if fmt == "vtt":
        body = "WEBVTT\n\n" + "".join(f"{ts(c['start'], '.')} --> {ts(c['end'], '.')}\n{c['text'].strip()}\n\n"
                                      for c in cues)
        return "text/vtt; charset=utf-8", body
    out = {"task": "transcribe", "language": (r.language or "").lower(), "duration": round(duration, 3),
           "text": text,
           "segments": [{"id": i, "seek": 0, "start": round(c["start"], 3), "end": round(c["end"], 3),
                         "text": c["text"], "tokens": [], "temperature": 0.0, "avg_logprob": None,
                         "compression_ratio": None, "no_speech_prob": None} for i, c in enumerate(cues)]}
    if words_wanted:
        out["words"] = [{"word": w["text"], "start": round(w["start"], 3), "end": round(w["end"], 3)}
                        for w in words]
    return "application/json", out

def main():
    home = Path.home()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default=str(home / "models/qwen3-asr-1.7b"))
    ap.add_argument("--aligner", default=str(home / "models/qwen3-forcedaligner-0.6b"))
    ap.add_argument("--model-id", default="qwen3-asr-1.7b")
    ap.add_argument("--dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    ap.add_argument("--q8", action="store_true", help="8-bit text decoder weights (W8A16)")
    ap.add_argument("--backend", default="mlx-qwen3-asr", choices=("mlx-qwen3-asr", "mlx-audio"))
    ap.add_argument("--fast", action="store_true", help="mlx-audio: our one-pass prefill / pipelined decode path")
    ap.add_argument("--batch", type=int, default=1, help="mlx-audio --fast: chunks of one request decoded together")
    a = ap.parse_args()
    if a.backend != "mlx-audio" and (a.fast or a.batch != 1):
        ap.error("--fast / --batch apply to --backend mlx-audio only")
    if a.batch > 1 and not a.fast:
        ap.error("--batch needs --fast (mlx-audio's own batched loop has no repetition stop)")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # weights are local; never fetch at runtime
    logging.basicConfig(level=logging.INFO, format="[qwen3-asr] %(message)s")
    server = Server((a.host, a.port), Handler)  # bind first so llama-swap sees the port while loading
    server.model_id = a.model_id
    server.response_builder = build_response
    server.engine = type("E", (), {"ready": False})()
    import threading
    threading.Thread(target=lambda: setattr(server, "engine", Engine(a.model, a.aligner, a.dtype, a.q8, a.backend, a.fast,
                                                                          a.batch)),
                     daemon=True).start()
    log.info("listening on http://%s:%d (model %s)", a.host, a.port, a.model)
    server.serve_forever()

if __name__ == "__main__":
    main()
