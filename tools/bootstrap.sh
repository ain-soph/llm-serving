#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
command -v brew >/dev/null || { echo 'Install Homebrew from https://brew.sh first.' >&2; exit 1; }
brew bundle --file "$root/Brewfile"
cd "$root"
uv sync --locked
printf '%s\n' 'Python environment ready. Next: .venv/bin/python tools/native.py --help'
