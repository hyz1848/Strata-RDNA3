#!/bin/sh
cd "/opt/Strata-main"
exec "/opt/Strata-main/.venv/bin/python" "/opt/Strata-main/serve/server.py" "--engine" "strata" "--config" "/opt/Strata-main/strata-iq3_xxs.json" "--port" "8080" "--open"
