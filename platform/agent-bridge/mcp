#!/bin/sh
# llm-collab MCP launcher. Installed as <bridge>/mcp by tools/install_agents.py from agents/llm-collab/launcher.sh
# in the llm-serving repository; edit it there, not in the bridge directory. Every path is resolved from this
# file's location at launch time:
#   <bridge>/release           ain-soph/llm-collab checkout (MCP server, pinned in agents/llm-collab/pins.json)
#   <bridge>/runtime/upstream  ain-soph/local-agent-bridge checkout, branch serving (Bridge runtime)
#   <bridge>/llm-collab.json   llm-collab config (relative paths: workspace, state, runtime, production root)
#   <bridge>/llm-collab.env    provider models and efforts; llm-collab.local.env (optional, unmanaged) overrides
set -eu
umask 077
bridge_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd -P)

# Node.js 22+: an explicit LLM_COLLAB_NODE, else the bridge's private runtime, else the node the installer used
# (<bridge>/node link), else Homebrew's or the system's, else one on the caller's PATH.
inherited_node=$(command -v node || true)
node=${LLM_COLLAB_NODE:-}
for candidate in "$bridge_dir/runtime/node/bin/node" "$bridge_dir/node"; do
  if [ -z "$node" ] && [ -x "$candidate" ]; then node=$candidate; fi
done
# MCP clients start servers with a minimal PATH. Use a fixed one so every client launches the same provider CLIs.
PATH="/opt/homebrew/bin:/usr/local/bin:${HOME:-}/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
if [ -z "$node" ]; then node=$(command -v node || printf '%s' "$inherited_node"); fi
if [ -z "$node" ] || [ ! -x "$node" ]; then
  echo "llm-collab: Node.js 22+ not found; install it (brew install node) or set LLM_COLLAB_NODE." >&2
  exit 1
fi
node=$(realpath "$node" 2>/dev/null || printf '%s' "$node")
PATH="$(dirname -- "$node"):$PATH"
export PATH

set -a
. "$bridge_dir/llm-collab.env"
if [ -f "$bridge_dir/llm-collab.local.env" ]; then . "$bridge_dir/llm-collab.local.env"; fi
set +a

# Keep the broker independent of the MCP client's working directory.
cd "$bridge_dir"
exec "$node" "$bridge_dir/release/bin/llm-collab.mjs" --config "$bridge_dir/llm-collab.json"
