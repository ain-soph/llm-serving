"""IndexTTS 2.5 speech API, using the repository's TTS HTTP transport."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import BadRequest, FFMPEG, Handler, MAX_INPUT, Server, split_text

LOG = logging.getLogger("indextts")
SAMPLE_RATE = 22050
MAX_AUDIO_SECONDS = 15
MAX_BODY_BYTES = 20 * 1024 * 1024
EMOTIONS = ("happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm")
LANGUAGES = {
    "zh": "zh", "chinese": "zh", "mandarin": "zh", "zh-cn": "zh",
    "en": "en", "english": "en", "ja": "ja", "japanese": "ja",
    "es": "es", "spanish": "es", "ar": "ar", "arabic": "ar",
}


def number(value, name, minimum, maximum):
    if type(value) not in (int, float) or not minimum <= value <= maximum:
        raise BadRequest(f"{name} must be between {minimum} and {maximum}")
    return float(value)


def audio_reference(value):
    if not isinstance(value, str) or not value:
        raise BadRequest("send ref_audio as base64 audio or an audio data URL")
    if value.startswith("data:"):
        prefix, sep, value = value.partition(",")
        if not sep or not prefix.startswith("data:audio/") or not prefix.endswith(";base64"):
            raise BadRequest("ref_audio must be a base64 audio data URL")
    try:
        audio = base64.b64decode(value, validate=True)
        decoded = subprocess.run(
            [FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error",
             "-protocol_whitelist", "pipe", "-i", "pipe:0", "-vn", "-t", str(MAX_AUDIO_SECONDS + 0.1),
             "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"],
            input=audio, capture_output=True, check=True, timeout=30,
        )
    except (ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise BadRequest("could not decode ref_audio") from exc
    samples = np.frombuffer(decoded.stdout, dtype="<f4")
    if not SAMPLE_RATE <= len(samples) <= MAX_AUDIO_SECONDS * SAMPLE_RATE or not np.isfinite(samples).all():
        raise BadRequest(f"ref_audio must be 1–{MAX_AUDIO_SECONDS} seconds of finite audio")
    return samples


def emotion_weights(req):
    emotion = req.get("emotion")
    if emotion is None:
        if req.get("emo_alpha") is not None:
            raise BadRequest("emo_alpha requires an emotion name or vector")
        return None
    if isinstance(emotion, str) and emotion.lower() in EMOTIONS:
        weights = [float(emotion.lower() == name) for name in EMOTIONS]
    elif isinstance(emotion, list) and len(emotion) == len(EMOTIONS):
        weights = [number(value, "emotion weight", 0, 1) for value in emotion]
    else:
        raise BadRequest(f"emotion must be an eight-number vector or one of {', '.join(EMOTIONS)}")
    if sum(weights) > 1.000001:
        raise BadRequest("emotion weights must sum to at most 1")
    alpha = number(req.get("emo_alpha", 0.6), "emo_alpha", 0, 1)
    return [value * alpha for value in weights]


class Engine:
    kind = "indextts-2.5"
    sr = SAMPLE_RATE
    speakers = ()

    def __init__(self, binary, model, bits, max_chars):
        self.command = [str(binary), str(model), str(bits)]
        self.max_chars = max_chars
        self.ready = False
        self.worker = None
        self.pending = bytearray()
        self.pool = ThreadPoolExecutor(max_workers=1)
        try:
            self.start_worker()
        except BaseException:
            self.close()
            raise
        self.ready = True

    def start_worker(self):
        self.pending.clear()
        self.worker = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        startup = self.event(timeout=180)
        if startup.get("kind") != "ready":
            raise RuntimeError("IndexTTS worker did not report readiness")
        LOG.info("model loaded in %.3f s", startup["seconds"])

    def event(self, timeout=120):
        deadline = time.monotonic() + timeout
        fd = self.worker.stdout.fileno()
        while (line_end := self.pending.find(b"\n")) < 0:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
                raise RuntimeError("IndexTTS worker timed out")
            data = os.read(fd, 65536)
            if not data:
                raise RuntimeError(f"IndexTTS worker exited ({self.worker.poll()})")
            self.pending.extend(data)
            if len(self.pending) > MAX_BODY_BYTES:
                raise RuntimeError("IndexTTS worker output exceeded its limit")
        line = self.pending[:line_end]
        del self.pending[:line_end + 1]
        return json.loads(line)

    def stop_worker(self):
        worker, self.worker = self.worker, None
        if worker is not None:
            if worker.poll() is None:
                worker.terminate()
                try:
                    worker.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait()
            worker.stdin.close()
            worker.stdout.close()

    def close(self):
        self.stop_worker()
        self.pool.shutdown(wait=False, cancel_futures=True)

    def options(self, req):
        text = req.get("input")
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_INPUT:
            raise BadRequest(f"input must contain 1–{MAX_INPUT} characters")
        if req.get("instructions") or req.get("emo_text") or req.get("emo_audio_prompt"):
            raise BadRequest("set emotion and emo_alpha for emotion control")
        language = req.get("language") or "auto"
        if not isinstance(language, str):
            raise BadRequest("language must be auto, zh, en, ja, es, or ar")
        language = language.strip().lower()
        if language != "auto" and language not in LANGUAGES:
            raise BadRequest("language must be auto, zh, en, ja, es, or ar")
        seed = req.get("seed")
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise BadRequest("seed must be an unsigned 64-bit integer")
        factor = req.get("duration_factor")
        if factor is not None and req.get("speed") is not None:
            raise BadRequest("send either speed or duration_factor")
        rate = (1.0 / number(factor, "duration_factor", 0.5, 2.0) if factor is not None
                else number(req.get("speed", 1.0), "speed", 0.5, 2.0))
        temperature = req.get("temperature")
        return {
            "pieces": split_text(text, max(1, int(self.max_chars * min(rate, 1)))),
            "reference": audio_reference(req.get("ref_audio")),
            "language": LANGUAGES.get(language),
            "emotion": emotion_weights(req),
            "speechRate": rate,
            "temperature": number(temperature, "temperature", 0, 2) if temperature is not None else None,
            "seed": seed,
            "speed": 1.0,
        }

    def stream(self, opts, out):
        try:
            if self.worker is None:
                self.start_worker()
            reference = opts["reference"]
            pcm16k = resample_poly(reference, 320, 441).astype("<f4", copy=False)
            payload = {name: opts[name] for name in ("pieces", "language", "emotion", "speechRate", "temperature", "seed")}
            payload["reference16k"] = base64.b64encode(pcm16k.tobytes()).decode()
            payload["reference22k"] = base64.b64encode(reference.tobytes()).decode()
            self.worker.stdin.write(json.dumps(payload, ensure_ascii=False).encode() + b"\n")
            self.worker.stdin.flush()
            while True:
                event = self.event()
                kind = event["kind"]
                if kind == "done":
                    LOG.info("generation %.3f s, MLX allocated peak %.0f MiB",
                             event["seconds"], event["peakBytes"] / 2**20)
                    out.put(None)
                    return
                if kind == "error":
                    LOG.error("IndexTTS inference failed: %s", event["message"])
                    out.put(RuntimeError("IndexTTS inference failed"))
                    return
                if kind != "audio":
                    raise RuntimeError("unexpected IndexTTS worker event")
                samples = np.frombuffer(base64.b64decode(event["samples"], validate=True), dtype="<f4")
                if samples.size == 0 or not np.isfinite(samples).all():
                    raise RuntimeError("IndexTTS generated invalid samples")
                out.put(samples)
        except Exception:
            LOG.exception("IndexTTS worker failed")
            self.stop_worker()
            out.put(RuntimeError("IndexTTS worker failed"))


class IndexHandler(Handler):
    timeout = 60

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if not 0 < length <= MAX_BODY_BYTES:
            self.close_connection = True
            return self._error(413, "expected JSON body of at most 20 MiB")
        super().do_POST()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", default="indextts-2.5")
    parser.add_argument("--max-chars", type=int, default=100)
    parser.add_argument("--bits", type=int, choices=(0, 8), default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    with Server(("127.0.0.1", args.port), IndexHandler) as server:
        server.model_id = args.model_id
        engine = Engine(args.worker, args.model, args.bits, args.max_chars)
        server.engine = engine
        try:
            server.serve_forever()
        finally:
            engine.close()


if __name__ == "__main__":
    main()
