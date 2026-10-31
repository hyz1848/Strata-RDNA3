#!/bin/bash
# Start Strata (Qwen3.8-Flash-Next IQ3_S) for this machine.
# Config: /opt/Strata-main/strata-iq3_xxs.json  (host 0.0.0.0, port 1234, cors *, parallel 2, ctx 262144)
# Loads ~47 GB into RAM; the PC can be slow for 1-2 minutes during load. Do not close.
set -u
cd /opt/Strata-main || exit 1
PORT="${PORT:-1234}"
LOG=/opt/Strata-main/run.log

# stop any previous instance (self-match-safe patterns)
systemctl --user stop strata-iq3s 2>/dev/null
pkill -f "serve/server[.]py --engine strata" 2>/dev/null
pkill -f "engine/strata --s[e]rve" 2>/dev/null
sleep 3

# start detached, no browser
nohup .venv/bin/python serve/server.py --engine strata --config strata-iq3_xxs.json --port "$PORT" > "$LOG" 2>&1 &
PID=$!
echo "Strata starting (pid $PID). First load ~1 min - the PC may be slow, please wait."

for _ in $(seq 1 60); do
  sleep 3
  if curl -s --max-time 3 "http://127.0.0.1:$PORT/v1/status" 2>/dev/null | grep -q '"loaded": true'; then
    IP=$(hostname -I | awk '{print $1}')
    echo "ready:"
    echo "  OpenAI :  http://$IP:$PORT/v1"
    echo "  Anthropic: http://$IP:$PORT/v1/messages"
    echo "  model  :  qwen3.8-27b   (api key: anything)"
    exit 0
  fi
done
echo "not ready yet after ~3 min; check $LOG"
exit 1
