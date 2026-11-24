# Environment of the llama-swap config: config.yaml and conf.d/*.yaml use ${env.NAME} for every machine-specific
# value, and llama-swap refuses to start (or -validate) when one is not set. Sourced (zsh or POSIX sh) by serve.sh,
# bin/lsw-apply and by hand before a manual check:
#   source ~/llm-serving/platform/llama-swap/env.sh && LLAMA_API_KEY=x llama-swap \
#     -config ~/llm-serving/platform/llama-swap/config.yaml -config-dir ~/llm-serving/platform/llama-swap/conf.d -validate
# LLM_SERVING   this checkout (default ~/llm-serving)      LLM_MODELS        model weights (default ~/models)
# STRATA_RELEASE  the deployed Strata release worktree     LLM_TRUSTED_PROXY  (gateway only)
# HY4_RELEASE   the Strata release that runs hy4-flash (build-hy4/bin/strata-hy4-metal + its server.py)
# H3_VIDEO_SHARE_BASE / H3_VIDEO_FETCH_HOSTS  public base URL of /media links, upload hosts the renderer may fetch
# Machine values live in $LLM_SERVING/.local/serving.env (written by tools/install_serving.py, never committed);
# the Caddy gateway reads the same file with --envfile.
: "${LLM_SERVING:=$HOME/llm-serving}"
if [ -f "$LLM_SERVING/.local/serving.env" ]; then
  set -a
  . "$LLM_SERVING/.local/serving.env"
  set +a
fi
: "${LLM_MODELS:=$HOME/models}"
export LLM_SERVING LLM_MODELS
# capacity pool from the current iogpu.wired_limit_mb (${env.LSW_GPU_POOL}, ${env.LSW_GPU_EXCLUSIVE})
eval "$("$LLM_SERVING/bin/gpu-pool" --env)"
