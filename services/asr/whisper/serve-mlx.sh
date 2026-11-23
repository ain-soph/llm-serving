#!/bin/zsh
# Whisper large-v3, MLX FP16, shared OpenAI-compatible transcription API.
set -e
port=${1:?usage: serve-mlx.sh PORT}; shift
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
root=${0:A:h:h:h:h}   # this checkout
exec "$root/.venv/bin/python" \
  "$root/services/asr/whisper/server.py" --port "$port" "$@"
