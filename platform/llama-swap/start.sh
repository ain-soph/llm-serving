#!/bin/zsh
# Start llama-server via its LaunchAgent (auto-restarts on crash, starts at login).
PLIST=~/Library/LaunchAgents/com.local.llama-server.plist
if launchctl print gui/$UID/com.local.llama-server >/dev/null 2>&1; then
  launchctl kickstart gui/$UID/com.local.llama-server && echo "llama-server already loaded (kickstarted)"
else
  launchctl bootstrap gui/$UID "$PLIST" && echo "llama-server started"
fi
echo "log: ${0:A:h}/logs/server.log"
