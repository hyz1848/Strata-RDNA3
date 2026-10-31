#!/bin/sh
# Strata optimal start - 2x RX 7900 XTX (gfx1100), Swift-1.5 IQ3_XXS, 256K, 2-way concurrency
# Measured on this box: layer-split decode ~WMMA_GEMM helps 4K/8K, full pinned arena, --batch 2.
cd "/opt/Strata-main" || exit 1

# --- speed switches (see docs/AMD_HIP.md, docs/MULTI_GPU.md) ---
export STRATA_WMMA_GEMM=0      # A/B: 写手说与 PF_GEMM 冲突
export STRATA_PREFILL_HELP=1   # idle card computes a share of one-chunk prompts (layer split)
export STRATA_QFUSE=0          # keep off with parallel:2 (issue #1356)
export STRATA_PF_GEMM=1          # #1389 AMD prompt-path levers
export STRATA_PF_SWITCH_MIN_T=4096
export STRATA_SPLIT_OWN=1        # each card keeps its own prompt buffers: prefill +6..17% (measured)
export STRATA_STAGE_TRIM=1     # community 4x7900XT A/B: 2.7x prefill
export STRATA_PF_FUSED=1        # community 4x7900XT A/B
export STRATA_SPLIT_MTP_BATCH=1  # 0.1.42: split MTP batch, 32K prompt +7.5% (draft K/V rounds differently)
export STRATA_SPLIT_RING=384     # 0.1.42: 104K prompt +2.3%, file reads 98->34 GB

export STRATA_ROUTE_TAIL_SKIP=7   # decode: skip the verify window's expert tail (default on CUDA/gfx12)

export STRATA_HIP_PROMPT_F16=1
export STRATA_IO_THREADS=32
export STRATA_SPLIT_OVERLAP=1

exec "/opt/Strata-main/.venv/bin/python" "/opt/Strata-main/serve/server.py" \
  "--engine" "strata" \
  "--config" "/opt/Strata-main/strata-swift-iq3s-huihui.json" \
  "--port" "1234"
