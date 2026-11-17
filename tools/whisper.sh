#!/bin/sh
set -eu
port=${1:?PORT required}; shift
exec "$LLM_PYTHON" "$LLM_ROOT/asr/whisper/server.py" --port "$port" \
  --model "$LLM_MODELS_DIR/whisper-large-v3-mlx" "$@"
