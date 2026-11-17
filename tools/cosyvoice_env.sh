#!/bin/sh
# Create tts/cosyvoice/.venv for the cosyvoice3 entry (docs/tts.md). mlx-audio-plus is used unmodified at upstream
# commit 4c9ec6a; it and the pinned requirements are installed with --no-deps, which reproduces the validated set.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
env="$root/tts/cosyvoice/.venv"
uv venv --python 3.12 "$env"
uv pip install --python "$env/bin/python" --no-deps -r "$root/tts/cosyvoice/requirements.txt"
uv pip install --python "$env/bin/python" --no-deps \
  "git+https://github.com/DePasqualeOrg/mlx-audio-plus@4c9ec6a8489e790b5ba8964ab1f1d63150476f9f"
"$env/bin/python" -c "import mlx_audio.tts.utils, mlx_audio.codec.models.s3tokenizer.model_v3; print('CosyVoice3 environment ready')"
