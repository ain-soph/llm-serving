#!/bin/zsh
# Start llama-swap (OpenAI-compatible API on port 8080) in front of llama-server and the H3-Image wrapper.
# Models are defined in config.yaml next to this script, plus one fragment per area in conf.d/*.yaml
# (models and selectors only; a model ID may be defined once). Each request is routed by its "model" field,
# and the capacity router keeps loaded whatever fits in the GPU pool (bin/gpu-pool), unloading by evict_cost.
# Listens on 127.0.0.1:8081 only; the Caddy gateway (~/llm-serving/platform/gateway, com.local.llm-gateway) serves :8080.
# API key is read from .api_key in this directory and passed to llama-swap as LLAMA_API_KEY.
# The previous llama-server router setup is kept in legacy-router/.
# Usage: ./serve.sh [extra llama-swap args...]
set -e
DIR="${0:A:h}"

export LLAMA_API_KEY="$(<"$DIR/.api_key")"
# capacity pool from the current iogpu.wired_limit_mb (config.yaml / conf.d use ${env.LSW_GPU_POOL} and ${env.LSW_GPU_EXCLUSIVE})
eval "$("$HOME/llm-serving/bin/gpu-pool" --env)"
exec llama-swap \
  -config "$DIR/config.yaml" \
  -config-dir "$DIR/conf.d" \
  -listen 127.0.0.1:8081 \
  "$@"
