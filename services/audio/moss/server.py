"""Serve MOSS audio understanding checkpoints through OpenAI chat completions."""

import argparse
import base64
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import uuid

import mlx.core as mx
from mlx_audio.lm.generate import StreamingDetokenizer
from mlx_audio.stt.models.moss_music.processor import ProcessorOutput
from mlx_audio.utils import base_load_model
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "asr"))
from http_api import BadRequest, Handler as AudioHandler, Server


LOG = logging.getLogger("moss")
AUDIO_SPAN = "<|audio_bos|><|AUDIO|><|audio_eos|>"
AUDIO_SECONDS = 600
SAMPLE_RATE = 16000
MAX_SAMPLES = AUDIO_SECONDS * SAMPLE_RATE
MAX_AUDIO_PARTS = 16
MAX_BODY = 64 * 1024 * 1024
CONTEXT = 16384
MAX_COMPLETION = 8192


def decode_audio(part, remaining_samples):
    if part.get("type") == "input_audio":
        source = part.get("input_audio")
        encoded = source.get("data") if isinstance(source, dict) else None
    else:
        source = part.get("audio_url")
        url = source.get("url") if isinstance(source, dict) else source
        encoded = None
        if isinstance(url, str):
            header, sep, data = url.partition(",")
            if sep and header.startswith("data:audio/") and header.endswith(";base64"):
                encoded = data
    if not isinstance(encoded, str) or not encoded:
        raise BadRequest("audio must be supplied as base64 input_audio or an audio data URL")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise BadRequest("invalid base64 audio") from exc
    try:
        result = subprocess.run(
            [
                os.environ.get("FFMPEG", "ffmpeg"), "-nostdin", "-loglevel", "error",
                "-protocol_whitelist", "pipe", "-threads", "1", "-i", "pipe:0",
                "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
                "-t", str((remaining_samples + SAMPLE_RATE // 10) / SAMPLE_RATE),
                "-f", "f32le", "pipe:1",
            ],
            input=raw, capture_output=True, timeout=30, check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise BadRequest("could not decode audio") from exc
    samples = np.frombuffer(result.stdout, dtype="<f4")
    if len(samples) > remaining_samples:
        raise BadRequest(f"audio duration exceeds {AUDIO_SECONDS} seconds per request")
    if len(samples) < SAMPLE_RATE // 20 or not np.isfinite(samples).all():
        raise BadRequest("audio must contain at least 0.05 seconds of finite samples")
    return samples


def chat_input(messages):
    if not isinstance(messages, list) or not messages:
        raise BadRequest("messages must be a nonempty list")
    rendered, audio = [], []
    remaining_samples = MAX_SAMPLES
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in ("system", "user", "assistant"):
            raise BadRequest("message role must be system, user or assistant")
        role, content = message["role"], message.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if not isinstance(content, list):
            raise BadRequest("message content must be text or a list of text and audio parts")
        parts = []
        for part in content:
            if not isinstance(part, dict):
                raise BadRequest("message parts must be objects")
            kind = part.get("type")
            if kind == "text" and isinstance(part.get("text"), str):
                value = part["text"]
                if "<|AUDIO|>" in value or "<|audio_bos|>" in value or "<|audio_eos|>" in value:
                    raise BadRequest("supply audio as a separate content part")
                parts.append(value)
            elif kind in ("input_audio", "audio_url") and role == "user":
                if len(audio) >= MAX_AUDIO_PARTS:
                    raise BadRequest(f"at most {MAX_AUDIO_PARTS} audio parts are allowed")
                waveform = decode_audio(part, remaining_samples)
                audio.append(waveform)
                remaining_samples -= len(waveform)
                parts.append(AUDIO_SPAN)
            else:
                raise BadRequest("expected text or a user audio part")
        rendered.append({"role": role, "content": "\n".join(parts)})
    if not any(message["role"] == "user" for message in rendered):
        raise BadRequest("a user message is required")
    return rendered, audio


def generation_options(body):
    max_tokens = body.get("max_completion_tokens", body.get("max_tokens", 1024))
    if type(max_tokens) is not int or not 1 <= max_tokens <= MAX_COMPLETION:
        raise BadRequest(f"max_completion_tokens must be between 1 and {MAX_COMPLETION}")
    options = {"max_tokens": max_tokens}
    for key, low, high, default in (("temperature", 0, 2, 0), ("top_p", 0, 1, 1)):
        value = body.get(key, default)
        if type(value) not in (float, int) or not low <= value <= high or not math.isfinite(value):
            raise BadRequest(f"{key} must be between {low} and {high}")
        options[key] = value
    if body.get("top_p") == 0:
        raise BadRequest("top_p must be greater than zero")
    return options


def load_model(path):
    kind = json.loads((path / "config.json").read_text())["model_type"]
    if kind not in ("moss_audio", "moss_music"):
        raise ValueError(f"unsupported MOSS checkpoint type {kind}")
    mx.set_cache_limit(512 * 1024 * 1024)
    return base_load_model(
        path, category="stt", model_remapping={}, model_type="moss_music", strict=True,
    )


def prepare(model, messages, audio, max_tokens):
    processor = model._ensure_processor()
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    if audio:
        processed = processor(text=prompt, audio=audio)
    else:
        ids = mx.array(processor.tokenizer.encode(prompt, add_special_tokens=False), dtype=mx.int32)
        processed = ProcessorOutput(input_ids=ids, audio_input_mask=ids == model.config.audio_token_id)
    if len(processed.input_ids) + max_tokens > CONTEXT:
        raise BadRequest(f"prompt and max_completion_tokens exceed the {CONTEXT}-token context")
    return model._build_prompt_embeddings(processed)


def token_events(model, inputs, options):
    tokenizer = model._ensure_processor().tokenizer
    start, end = (tokenizer.convert_tokens_to_ids(tag) for tag in ("<think>", "</think>"))
    detokenizer = StreamingDetokenizer(tokenizer, skip_special_tokens=True)
    channel = "content"
    reason = "length"
    count = 0
    for token in model._generate_tokens(
        *inputs[:3], **options, top_k=0, min_p=0, repetition_penalty=None,
        repetition_context_size=0, prefill_step_size=1024,
    ):
        if token == model.config.eos_token_id:
            reason = "stop"
            break
        count += 1
        if token in (start, end):
            detokenizer.finalize()
            if segment := detokenizer.last_segment:
                yield {channel: segment}
            detokenizer.reset()
            channel = "reasoning_content" if token == start else "content"
        else:
            detokenizer.add_token(token)
            if segment := detokenizer.last_segment:
                yield {channel: segment}
    detokenizer.finalize()
    if segment := detokenizer.last_segment:
        yield {channel: segment}
    yield reason, count


class Handler(AudioHandler):
    server_version = "local-moss-server"
    timeout = 60

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.close_connection = True
            return self._error(404, "unknown path")
        self.streaming_response = False
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= MAX_BODY:
                self.close_connection = True
                return self._error(413, "expected JSON body of at most 64 MiB")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise BadRequest("expected a JSON object")
            if type(body.get("stream", False)) is not bool:
                raise BadRequest("stream must be a boolean")
            n = body.get("n", 1)
            if type(n) is not int or n != 1:
                raise BadRequest("n must be 1")
            options = generation_options(body)
            with self.server.inference_lock:
                messages, audio = chat_input(body.get("messages"))
                model = self.server.engine.model
                inputs = prepare(model, messages, audio, options["max_tokens"])
                self.complete(body, token_events(model, inputs, options), inputs[3])
        except (BadRequest, ValueError) as exc:
            if not self.streaming_response:
                self._error(400, str(exc))
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            self.close_connection = True
        except Exception:
            LOG.exception("chat completion failed")
            if not self.streaming_response:
                self._error(500, "chat completion failed", "server_error")

    def complete(self, body, events, prompt_tokens):
        base = {
            "id": f"chatcmpl-{uuid.uuid4().hex}", "created": int(time.time()),
            "model": self.server.model_id,
        }
        stream = body.get("stream", False)
        include_usage = isinstance(body.get("stream_options"), dict) and body["stream_options"].get("include_usage")
        fields = {}

        def send_chunk(choices, **extra):
            data = dict(base, object="chat.completion.chunk", choices=choices, **extra)
            self.wfile.write(f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()

        if stream:
            self.streaming_response = True
            self.close_connection = True
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            send_chunk([{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}])

        for event in events:
            if isinstance(event, tuple):
                reason, count = event
            elif stream:
                send_chunk([{"index": 0, "delta": event, "finish_reason": None}])
            else:
                for key, value in event.items():
                    fields.setdefault(key, []).append(value)
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": count,
                 "total_tokens": prompt_tokens + count}
        if stream:
            send_chunk([{"index": 0, "delta": {}, "finish_reason": reason}])
            if include_usage:
                send_chunk([], usage=usage)
            self.wfile.write(b"data: [DONE]\n\n")
            return

        message = {"role": "assistant", "content": ""}
        message.update({key: "".join(pieces) for key, pieces in fields.items()})
        self._send(200, "application/json", dict(
            base, object="chat.completion", usage=usage,
            choices=[{"index": 0, "message": message, "finish_reason": reason}],
        ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    model = load_model(args.model)
    with Server(("127.0.0.1", args.port), Handler) as server:
        server.engine = SimpleNamespace(model=model, ready=True)
        server.model_id = args.model_id
        server.inference_lock = threading.Lock()
        server.serve_forever()


if __name__ == "__main__":
    main()
