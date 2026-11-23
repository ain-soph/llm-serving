#!/bin/sh
# Transcribe one audio file with Whisper large-v3 through the local gateway, i.e. the llama-swap entry
# "whisper-large-v3" (platform/llama-swap/conf.d/asr.yaml -> services/asr/whisper/serve-mlx.sh, weights
# ~/models/whisper-large-v3-q8). llama-swap loads the model on the first request and unloads it after its ttl.
# Until 2026-11-23 this file launched the server itself from the pre-2026-11-18 layout ($LLM_ROOT/asr, the
# whisper-large-v3-mlx weights); the llama-swap entry has replaced it.
#
#   tools/whisper.sh AUDIO [FIELD=VALUE ...]
#   e.g. tools/whisper.sh talk.m4a language=zh response_format=srt
# Fields are sent as multipart form fields of POST /v1/audio/transcriptions; response_format defaults to text
# (json, verbose_json, srt, vtt also work). Env: LLM_GATEWAY (default http://127.0.0.1:8080), WHISPER_MODEL.
set -eu
audio=${1:?usage: whisper.sh AUDIO [FIELD=VALUE ...]}; shift
[ -f "$audio" ] || { echo "whisper.sh: no such file: $audio" >&2; exit 2; }
root=$(cd "$(dirname "$0")/.." && pwd)
key=$(cat "$root/platform/llama-swap/.api_key")
fmt=text
for kv do   # each FIELD=VALUE becomes --form-string FIELD=VALUE
  shift
  case $kv in *=*) ;; *) echo "whisper.sh: expected FIELD=VALUE, got: $kv" >&2; exit 2;; esac
  case $kv in response_format=*) fmt=;; esac
  set -- "$@" --form-string "$kv"
done
# The API key goes to curl on stdin (-K -), not on the command line where ps would show it.
printf 'header = "Authorization: Bearer %s"\n' "$key" |
  curl -sS --fail-with-body -K - "${LLM_GATEWAY:-http://127.0.0.1:8080}/v1/audio/transcriptions" \
    --form-string "model=${WHISPER_MODEL:-whisper-large-v3}" ${fmt:+--form-string "response_format=$fmt"} \
    "$@" -F "file=@$audio"
