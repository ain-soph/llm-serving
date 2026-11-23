#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
command -v brew >/dev/null || { echo 'Install Homebrew from https://brew.sh first.' >&2; exit 1; }
brew bundle --file "$root/Brewfile"
cd "$root"
python_bin="$(brew --prefix python@3.15)/bin/python3.15"
uv sync --locked --python "$python_bin"
# Privacy gate on commits and pushes (tools/git-hooks: privacy_check, .gitignore generated from public-paths.txt).
[ -d .git ] && git config core.hooksPath tools/git-hooks
# Multi-agent rules -> ~/.claude/CLAUDE.md (marker block, never overwrites), the local worker link, and the
# llm-collab bridge (pinned checkouts, launcher, MCP registration; platform/agent-bridge/README.md).
[ -n "${LLM_SERVING_SKIP_AGENTS:-}" ] || .venv/bin/python tools/install_agents.py
# Machine-specific serving files: .local/serving.env, Strata engine JSON, ~/Library/LaunchAgents, API key
# (creates missing files only, loads nothing; values from .local/serving-values.json).
[ -n "${LLM_SERVING_SKIP_SERVING:-}" ] || .venv/bin/python tools/install_serving.py
printf '%s\n' 'Python environment ready. Next: .venv/bin/python tools/native.py --help'
