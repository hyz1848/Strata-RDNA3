#!/bin/bash
# Stop the Strata server started by start_strata.sh
systemctl --user stop strata-iq3s 2>/dev/null
pkill -f "serve/server[.]py --engine strata" 2>/dev/null
pkill -f "engine/strata --s[e]rve" 2>/dev/null
sleep 2
if pgrep -f "serve/server[.]py --engine strata" >/dev/null 2>&1; then
  echo "warning: still running"
  exit 1
fi
echo "Strata stopped"
