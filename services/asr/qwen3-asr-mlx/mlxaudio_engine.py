"""Qwen3-ASR + Qwen3-ForcedAligner on mlx-audio's qwen3_asr / qwen3_forced_aligner (2026-11-21, evaluation of a switch
away from mlx-qwen3-asr). Same request semantics as the mlx-qwen3-asr path in server.py:

- audio split into <= CHUNK_S second chunks at low-energy points (mlx-audio's splitter), each transcribed on its own;
- language: ISO codes / names mapped to Qwen3-ASR's prompt names; auto-detect per chunk, the request's language is the
  first detected one;
- greedy decoding with a per-chunk token budget of clamp(ceil(12 * seconds) + 32, 128, 512) (mlx-qwen3-asr's rule);
  a chunk that exhausts it without an end token marks the result truncated (the server refuses those);
- decoding stops early on a repeated token / short pattern (threshold 20, as mlx-qwen3-asr) and the text gets the
  official Qwen3-ASR repetition clean-up;
- word timestamps: the forced aligner on each chunk's audio and text, shifted by the chunk offset.

The repetition helpers are ported from mlx-qwen3-asr (Apache-2.0, itself ported from the official Qwen3-ASR toolkit).
"""
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np

SR = 16000
CHUNK_S = 30.0
REPETITION_THRESHOLD = 20

_ALIASES = {
    "en": "English", "en-us": "English", "en-gb": "English", "zh": "Chinese", "zh-cn": "Chinese", "zh-tw": "Chinese",
    "cmn": "Chinese", "mandarin": "Chinese", "yue": "Cantonese", "zh-hk": "Cantonese", "ja": "Japanese",
    "ko": "Korean", "de": "German", "fr": "French", "es": "Spanish", "es-419": "Spanish", "ru": "Russian",
    "ar": "Arabic", "hi": "Hindi", "it": "Italian", "pt": "Portuguese", "pt-br": "Portuguese", "tr": "Turkish",
    "nl": "Dutch", "id": "Indonesian", "th": "Thai", "vi": "Vietnamese", "ms": "Malay", "sv": "Swedish",
    "da": "Danish", "fi": "Finnish", "pl": "Polish", "cs": "Czech", "fil": "Filipino", "tl": "Filipino",
    "fa": "Persian", "el": "Greek", "ro": "Romanian", "hu": "Hungarian", "mk": "Macedonian",
}
NO_SPACE = {"chinese", "cantonese", "japanese"}


def canonical_language(language: Optional[str], supported) -> Optional[str]:
    if not language or not str(language).strip():
        return None
    key = str(language).strip().casefold().replace("_", "-")
    names = {s.casefold(): s for s in supported or []}
    return names.get(key) or _ALIASES.get(key) or str(language).strip()


def token_budget(seconds: float, cap_max: int = 512) -> int:
    return min(cap_max, max(128, math.ceil(max(0.0, seconds) * 12.0) + 32))


def detect_repetition(tokens, threshold=REPETITION_THRESHOLD) -> bool:
    if len(tokens) < threshold:
        return False
    count = 1
    for i in range(len(tokens) - 1, 0, -1):
        if tokens[i] != tokens[i - 1]:
            break
        count += 1
        if count >= threshold:
            return True
    for plen in range(2, min(11, len(tokens) // 2 + 1)):
        pattern, repeats, pos = tokens[-plen:], 1, len(tokens) - plen * 2
        while pos >= 0 and tokens[pos:pos + plen] == pattern:
            repeats += 1
            pos -= plen
        if repeats >= max(2, threshold // plen):
            return True
    return False


def fix_repetitions(text: str, threshold: int = REPETITION_THRESHOLD) -> str:
    def char_runs(s):
        out, i, n = [], 0, len(s)
        while i < n:
            j = i + 1
            while j < n and s[j] == s[i]:
                j += 1
            out.append(s[i] if j - i > threshold else s[i:j])
            i = j
        return "".join(out)

    def pattern_runs(s, max_len=20):
        n = len(s)
        if n < threshold * 2:
            return s
        i, out = 0, []
        while i <= n - threshold * 2:
            for k in range(1, max_len + 1):
                if i + k * threshold > n:
                    break
                p = s[i:i + k]
                if all(s[i + r * k:i + r * k + k] == p for r in range(1, threshold)):
                    end = i + threshold * k
                    while end + k <= n and s[end:end + k] == p:
                        end += k
                    return "".join(out) + p + pattern_runs(s[end:], max_len)
            out.append(s[i])
            i += 1
        return "".join(out) + s[i:]

    return pattern_runs(char_runs(text))


@dataclass
class Result:
    text: str
    language: str
    segments: Optional[list] = None          # words: {text, start, end}
    chunks: list = field(default_factory=list)
    truncated: bool = False
    stats: dict = field(default_factory=dict)


class MlxAudioQwen3:
    def __init__(self, model_dir, aligner_dir=None, chunk_s=CHUNK_S, batch=1, profile=False, fast_embed=False,
                 fast=False, cap_max=512):
        from mlx_audio.stt.utils import load_model
        self.model = load_model(Path(model_dir))
        self.aligner = load_model(Path(aligner_dir)) if aligner_dir else None
        self.supported = self.model.config.support_languages or []
        self.chunk_s, self.batch, self.profile = chunk_s, batch, profile
        self.eos = self.model._eos_token_ids()
        self.stats = {}
        self.fast, self.cap_max = fast, cap_max
        self._ids_checked = set()
        if fast_embed or fast:
            self._vectorize_embed_merge()
        if profile:
            self._wrap_profile()

    def _vectorize_embed_merge(self):
        """Optimization candidate: upstream _build_inputs_embeds rebuilds the prompt row by row in Python (one slice per
        token, then mx.stack); this gathers the same rows with one index instead (bit-identical result)."""
        inner = [getattr(self.model, "_model", self.model)]
        if self.aligner is not None:
            inner.append(getattr(self.aligner, "_model", self.aligner))
        for m in inner:
            audio_id = m.config.audio_token_id

            def build(input_ids, audio_features, m=m, audio_id=audio_id):
                emb = m.model.embed_tokens(input_ids)
                af = audio_features.astype(emb.dtype)
                if input_ids.shape[0] != 1:
                    raise ValueError("expects one prompt")
                ids = np.array(input_ids[0])
                pos = np.flatnonzero(ids == audio_id)
                n = min(len(pos), af.shape[0])
                if n == 0:
                    return emb
                index = np.arange(ids.shape[0], dtype=np.int32)
                index[pos[:n]] = ids.shape[0] + np.arange(n, dtype=np.int32)
                rows = mx.concatenate([emb[0], af[:n]], axis=0)
                return mx.take(rows, mx.array(index), axis=0)[None]
            m._build_inputs_embeds = build

    # --- profiling (forces a sync after the encoder; use for the stage split, not for end-to-end times) ---
    def _wrap_profile(self):
        m = getattr(self.model, "_model", self.model)  # mlx-audio's Model forwards to an inner Qwen3ASRModel
        orig_feat, orig_enc, orig_emb = m._preprocess_audio, m.get_audio_features, m._build_inputs_embeds

        def timed(key, fn):
            def wrap(*a, **k):
                t = time.perf_counter()
                out = fn(*a, **k)
                mx.eval([x for x in (out if isinstance(out, tuple) else (out,)) if isinstance(x, mx.array)])
                self.stats[key] = self.stats.get(key, 0.0) + time.perf_counter() - t
                return out
            return wrap
        m._preprocess_audio = timed("features_s", orig_feat)
        m.get_audio_features = timed("encoder_s", orig_enc)
        m._build_inputs_embeds = timed("embed_merge_s", orig_emb)

    # --- fast path (our code on mlx-audio's modules): one prefill pass, pipelined greedy loop, no clear_cache per chunk ---
    def _prompt_ids(self, m, n_audio, build):
        """Token ids with the <|audio_pad|> run expanded in numpy instead of tokenizing N copies of the pad string."""
        audio_id = m.config.audio_token_id
        one = np.array(build(1)).reshape(-1)
        pos = int(np.flatnonzero(one == audio_id)[0])
        ids = np.concatenate([one[:pos], np.full(n_audio, audio_id, dtype=one.dtype), one[pos + 1:]])
        key = (id(m), n_audio > 0)
        if key not in self._ids_checked:  # one-time check against the library's own tokenization
            full = np.array(build(n_audio)).reshape(-1)
            if not np.array_equal(full, ids):
                raise RuntimeError("expanded prompt ids differ from the tokenizer's")
            self._ids_checked.add(key)
        return mx.array(ids[None])

    def _decode_chunk_fast(self, audio, language, context):
        m = getattr(self.model, "_model", self.model)
        cap = token_budget(len(audio) / SR, self.cap_max)
        t0 = time.perf_counter()
        feats, fmask, n_audio = m._preprocess_audio(audio)
        af = m.get_audio_features(feats, fmask)
        ids = self._prompt_ids(m, n_audio, lambda n: m._build_prompt(n, language, context or None))
        x = m._build_inputs_embeds(ids, af)
        cache = m.make_cache()
        head = m.lm_head if m.lm_head is not None else m.model.embed_tokens.as_linear

        def logits_of(h):
            return head(h[:, -1:, :])[:, -1, :]
        y = mx.argmax(logits_of(m.model(inputs_embeds=x, cache=cache)), axis=-1)
        mx.async_eval(y)
        tokens, finish, t_first = [], "length", None
        for _ in range(cap):
            nxt = mx.argmax(logits_of(m.model(inputs_embeds=m.model.embed_tokens(y[:, None]), cache=cache)), axis=-1)
            mx.async_eval(nxt)
            t = int(y.item())
            if t_first is None:
                t_first = time.perf_counter()
            if t in self.eos:
                finish = "stop"
                break
            tokens.append(t)
            if detect_repetition(tokens):
                finish = "repetition"
                break
            y = nxt
        t_end = time.perf_counter()
        if self.profile:
            st = self.stats
            st["first_token_s"] = st.get("first_token_s", 0.0) + ((t_first or t_end) - t0)
            st["decode_s"] = st.get("decode_s", 0.0) + (t_end - (t_first or t_end))
            st["decode_steps"] = st.get("decode_steps", 0) + max(0, len(tokens))
            st["chunks"] = st.get("chunks", 0) + 1
        ntok = len(tokens)
        return m._tokenizer.decode(tokens, skip_special_tokens=True), finish, ntok, cap

    def _align_fast(self, audio, text, language):
        a = getattr(self.aligner, "_model", self.aligner)
        feats, fmask, n_audio = a._preprocess_audio(audio)
        words, text_in = a.aligner_processor.encode_timestamp(text, language)
        ids = self._prompt_ids(a, n_audio, lambda n: a._tokenizer.encode(
            text_in.replace("<|audio_pad|>", "<|audio_pad|>" * n), return_tensors="np", add_special_tokens=False))
        x = a._build_inputs_embeds(ids, a.get_audio_features(feats, fmask))
        out = np.array(mx.argmax(a.lm_head(a.model(inputs_embeds=x, cache=None)), axis=-1)[0])
        ms = out[np.array(ids[0]) == a.config.timestamp_token_id] * a.config.timestamp_segment_time
        return [{"text": str(it["text"]), "start": round(it["start_time"] / 1000.0, 3),
                 "end": round(it["end_time"] / 1000.0, 3)} for it in a.aligner_processor.parse_timestamp(words, ms)]

    def _decode_chunk(self, audio, language, context):
        if self.fast:
            return self._decode_chunk_fast(audio, language, context)
        cap = token_budget(len(audio) / SR, self.cap_max)
        tokens, finish = [], "stop"
        t0 = time.perf_counter()
        t_first = None
        gen = self.model.stream_generate(audio, max_tokens=cap, language=language, system_prompt=context or None)
        for tok, _ in gen:
            if t_first is None:
                t_first = time.perf_counter()
            tokens.append(int(tok))
            if detect_repetition(tokens):
                finish = "repetition"
                break
        else:
            if len(tokens) >= cap:
                finish = "length"
        gen.close()
        t_end = time.perf_counter()
        if self.profile:
            s = self.stats
            s["first_token_s"] = s.get("first_token_s", 0.0) + ((t_first or t_end) - t0)
            s["decode_s"] = s.get("decode_s", 0.0) + (t_end - (t_first or t_end))
            s["decode_steps"] = s.get("decode_steps", 0) + max(0, len(tokens) - 1)
            s["chunks"] = s.get("chunks", 0) + 1
        raw = self.model._tokenizer.decode(tokens, skip_special_tokens=True)
        return raw, finish, len(tokens), cap

    def _parse(self, raw, forced):
        s = fix_repetitions(raw.strip())
        if forced:
            return forced, (s.split("<asr_text>", 1)[1] if "<asr_text>" in s else s).strip()
        if "<asr_text>" in s:
            head, text = s.split("<asr_text>", 1)
            head, text = head.strip(), text.strip()
            lang = head[len("language "):].strip() if head.startswith("language ") else head
            if lang.casefold() == "none" and not text:
                return "unknown", ""
            return canonical_language(lang, self.supported), text
        return "unknown", s.strip()

    def transcribe(self, audio: np.ndarray, language=None, context="", timestamps=False):
        from mlx_audio.stt.models.qwen3_asr.qwen3_asr import split_audio_into_chunks
        forced = canonical_language(language, self.supported)
        pieces = split_audio_into_chunks(audio, sr=SR, chunk_duration=self.chunk_s, min_chunk_duration=1.0)
        total = len(audio) / SR
        ends = [off for _, off in pieces[1:]] + [total]
        batched = self._decode_batched_fast if self.fast else self._decode_batched
        decoded = batched(pieces, forced, context) if self.batch > 1 and len(pieces) > 1 else \
            [self._decode_chunk(a, forced, context) for a, _ in pieces]
        texts, words, chunks, truncated, detected = [], [], [], False, None
        for (chunk_audio, offset), end, (raw, finish, ntok, cap) in zip(pieces, ends, decoded):
            lang, text = self._parse(raw, forced)
            if detected is None and lang != "unknown":
                detected = lang
            truncated |= finish == "length"
            texts.append(text)
            chunks.append({"text": text, "start": float(offset), "end": float(end), "language": lang,
                           "finish_reason": finish, "generated_tokens": ntok, "max_new_tokens": cap})
            if timestamps and self.aligner is not None and text.strip() and lang != "unknown":
                t = time.perf_counter()
                if self.fast:
                    items = self._align_fast(chunk_audio, text, forced or lang)
                else:
                    items = [{"text": it.text, "start": it.start_time, "end": it.end_time}
                             for it in self.aligner.generate(chunk_audio, text, language=forced or lang)]
                for it in items:
                    words.append({"text": it["text"], "start": it["start"] + offset, "end": it["end"] + offset})
                if self.profile:
                    self.stats["align_s"] = self.stats.get("align_s", 0.0) + time.perf_counter() - t
        detected = forced or detected or "unknown"
        joiner = "" if detected.casefold() in NO_SPACE else " "
        text = joiner.join(t for t in (x.strip() for x in texts) if t)
        mx.clear_cache()
        return Result(text=text, language=detected, segments=words if timestamps else None, chunks=chunks,
                      truncated=truncated)

    def _decode_batched_fast(self, pieces, forced, context):
        """Chunks of one request decoded together, `batch` at a time (decode reads every weight once per step, so a
        batch costs about one sequence). Chunks are grouped by length and padded with silence to the group's longest
        (the prompts must have one length); per row: end token, repetition stop and token budget as in the 1-row path."""
        m = getattr(self.model, "_model", self.model)
        head = m.lm_head if m.lm_head is not None else m.model.embed_tokens.as_linear

        def next_tokens(h):
            return mx.argmax(head(h[:, -1:, :])[:, -1, :], axis=-1)
        out = [None] * len(pieces)
        order = sorted(range(len(pieces)), key=lambda i: len(pieces[i][0]))
        for g0 in range(0, len(order), self.batch):
            group = order[g0:g0 + self.batch]
            pad_to = max(len(pieces[i][0]) for i in group)
            rows, caps = [], []
            for i in group:
                a = pieces[i][0]
                caps.append(token_budget(len(a) / SR, self.cap_max))
                a = np.pad(a, (0, pad_to - len(a))) if len(a) < pad_to else a
                feats, fmask, n_audio = m._preprocess_audio(a)
                ids = self._prompt_ids(m, n_audio, lambda n: m._build_prompt(n, forced, context or None))
                rows.append(m._build_inputs_embeds(ids, m.get_audio_features(feats, fmask))[0])
            cache = m.make_cache()
            y = next_tokens(m.model(inputs_embeds=mx.stack(rows), cache=cache))
            mx.async_eval(y)
            toks, done = [[] for _ in group], [None] * len(group)
            for _ in range(max(caps)):
                nxt = next_tokens(m.model(inputs_embeds=m.model.embed_tokens(y[:, None]), cache=cache))
                mx.async_eval(nxt)
                for b, t in enumerate(y.tolist()):
                    if done[b]:
                        continue
                    if t in self.eos:
                        done[b] = "stop"
                        continue
                    toks[b].append(t)
                    if detect_repetition(toks[b]):
                        done[b] = "repetition"
                    elif len(toks[b]) >= caps[b]:
                        done[b] = "length"
                if all(done):
                    break
                y = nxt
            for b, i in enumerate(group):
                out[i] = (m._tokenizer.decode(toks[b], skip_special_tokens=True), done[b] or "length", len(toks[b]),
                          caps[b])
        return out

    def _decode_batched(self, pieces, forced, context):
        """Optimization candidate: mlx-audio's batched chunk decode (pads a group's audio to equal length)."""
        caps = [token_budget(len(a) / SR, self.cap_max) for a, _ in pieces]
        cap = max(caps)
        texts, gen_toks, _, _ = self.model._generate_chunks_batched(
            pieces, max_tokens=cap, sampler=lambda x: mx.argmax(x, axis=-1), logits_processors=None,
            language=forced, system_prompt=context or None, batch_size=self.batch, verbose=False)
        # the batched path decodes with skip_special_tokens and has no per-step repetition stop
        return [(t, "length" if g >= c else "stop", g, c) for t, g, c in zip(texts, gen_toks, caps)]
