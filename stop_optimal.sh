#!/bin/sh
# Stop the Strata server started by start_optimal.sh
pkill -f "engine/strata --s[e]rve" 2>/dev/null
pkill -f "serve/server[.]py --engine strata" 2>/dev/null
echo "stopped"
