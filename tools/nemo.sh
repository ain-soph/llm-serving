#!/bin/sh
set -eu
port=${1:?PORT required}; shift
: "${LLM_NEMO_EXE:?Nemotron is retired (2026-11-18); rerun tools/configure.py with --nemo-exe to re-enable}"
model_dir="$LLM_MODELS_DIR/nemotron-3.5-transcribe-30b-a4b-gguf"
exec "$LLM_NEMO_EXE" serve --host 127.0.0.1 --port "$port" --no-ui --device metal \
  --asr-model "$model_dir/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-AD-IQ4_NL.gguf" \
  --asr.model.name nemotron-3.5-transcribe \
  --asr.salm.max_sequences 4 --asr.salm.n_ctx_per_sequence 18432 --asr.salm.max_context 73728 \
  --asr.salm.mtp_draft 0 --read-timeout 300 --write-timeout 300 "$@"
