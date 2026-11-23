#!/bin/zsh
# Stop llama-server to free unified memory (e.g. before running another large model).
# Unloads the LaunchAgent so it stays stopped until ./start.sh or the next login.
if launchctl print gui/$UID/com.local.llama-server >/dev/null 2>&1; then
  launchctl bootout gui/$UID/com.local.llama-server && echo "LaunchAgent unloaded"
fi
pkill -x llama-swap; pkill -x llama-server && echo "llama-server stopped" || echo "llama-server not running"
