#!/bin/sh
# UD-Q4_K_XL (Unsloth dynamic 4-bit) on port 1236 - test config, does not touch the Swift pack
cd "/opt/Strata-main" || exit 1

export STRATA_WMMA_GEMM=1
export STRATA_PREFILL_HELP=1
export STRATA_QFUSE=0
export STRATA_PF_GEMM=1
export STRATA_PF_SWITCH_MIN_T=4096
export STRATA_SPLIT_OWN=1
export STRATA_RESIDENT_HEADROOM_GIB=1

exec "/opt/Strata-main/.venv/bin/python" "/opt/Strata-main/serve/server.py" \
  "--engine" "strata" \
  "--config" "/opt/Strata-main/strata-ud-q4_k_xl.json" \
  "--port" "1236"
