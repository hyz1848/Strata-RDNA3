# 0.2.20 / engine 0.1.66 merge (2x RX 7900 XTX, DavidAU IQ3_S)

Base: upstream Niko1221/Strata v0.2.20 (engine 0.1.66), merged onto the 0.1.65 fork merge
(`merge-0219`). The fork's tensor-parallel generate/prefill is NOT in this tree: the
upstream prefill/generate/verify are used, plus the fork's GR kernels and r8 images.

## What is kept from the fork

- `engine/src/kernels/cuda/fused_gr.cu`, `rdna_gr.cu` and `fused_gr.hpp` (fork pair: the
  upstream pair crashes inside a verify-window capture with the fork env switches:
  `fused_gr_read_multi rdna: ... during capture`).
- `engine/src/core/pinned.cu` (PinnedArena Adopt ctor, used by arena_shm).
- `include/strata/core/native_head.hpp`: `rows()`/`row0()` accessors.
- `core/layer.cpp`: `qsa_state_init` keeps the 7th `host_owner` parameter (ignored).

## Patches on top of 0.1.66 (`b36ed2977`)

Wide batch-MTP windows (a slot's token + its drafts fill one window):

- `generate.cpp` `batch_max_t()`: serial batch-MTP returns
  `min(batch * (mtp_max_t or spec), kVerifyMaxT)`.
- `generate.cpp` `hrows` (hand-off buffer rows): same bound.
- `generate.cpp` `batch_step`: `win_cap = kVerifyMaxT`; `n_act` and
  `per_cap = win_cap / n_act - 1`; a slot's rows = `min(n_drafts, per_cap, room)`.
- `generate.cpp` `set_max_drafts` for the serial slot drafters:
  `min((mtp_max_t or spec) - 1, kVerifyMaxT / batch - 1)`.
- The draft call feeds `feed = min(cap, nrow)` rows and returns `n_drafts`.

## Measured (4x / 8x 16K prompts, 512 output tokens, engine batch stats)

| config | rows/window | ms/window | decode aggregate |
|---|---|---|---|
| 4 slots, 1 draft | 7.3 | 44.8 | 163 |
| 4 slots, 2 drafts, 256k tuned | 9.7 | 57.0 | 170 |
| 8 slots, 1 draft (wide, 16 rows) | 14.4 | 75.9 | 190 avg, 211 at 16 rows |
| old 0.1.59+fork (monitor, running=4) | ~8.9 | ~45 | 194 |

Single request 16K: prefill 3263-3292 tok/s, decode 109-120 tok/s. 32K prefill 4200 tok/s.

## What does not work

- `STRATA_BATCH_MTP_FUSED=1`: single GPU only (`stages.empty()`), no effect on a split.
- `STRATA_PIPED_MTP=1` with `--batch-groups 2`: the piped groups carry no slot drafts and
  the chain variant dies with `verify batch: timed out at layer 31`.
- `--adapt-every 0`: CPU-routed experts go 1.8% -> 20%, window 76 -> 115 ms. Keep adapt on.
- `--spec-min-p 0` without enough draft room: no change beyond the per-slot cap.

## The remaining 4-concurrency gap

The window is `wait 21-27 ms (both stages, sum = the two stages' compute) + commit 3-7 +
emit 2-5`. 4 slots cannot pass 16 rows, so rows/(wait + rows*cost) plateaus at ~170-178.
Upstream states it explicitly: "Batch windows never use the second verifiers" (a layer
split's second parity, `ver_b`, is solo-only). Getting 4 concurrent to 194 needs batch
window double-buffering: the A window runs on stage 1 while the B window's stage-0 work
starts, with `init_deferred_commit` / `batch_launch_rows` / `batch_commit_rows` over the
two parities (solo has this; batch does not). The 8-slot config already reaches 211 at 16
rows and is the working answer meanwhile.

## TP is the missing core (owner's direction)

0.1.66 ships `src/core/tp.cpp` and `include/strata/core/tp.hpp`, but nothing wires
them: `generate.cpp` has no `--tensor-parallel`, no `tp_on`, and `verify.cpp` never
builds a `TpLink`; `verify_kernels` is the upstream one (no `tp_exchange`). The fork's
TP (ours branch) is the 2x decode the old engine had.

Port order (ours -> 0.1.66):

1. `verify_kernels.{hpp,cu}`: `tp_window_begin`, `tp_exchange`, `tp_exchange_blocks`,
   `tp_wait_epoch` (ours has them; 0.1.66 does not).
2. `verify.cpp/hpp`: `TpLink`, the per-layer exchange points, head split, `set_tp_head_split`,
   `tp_gpu_only_`, `tp_other_res`, the batch row tables for two-card rows.
3. `generate.cpp`: `--tensor-parallel` parse, `tp_on = ... && o.batch == 0`, session/dev
   setup for the pair, `run_tp`/TP prefill, the head split, the DP/TP helpers ours calls.
4. `prefill`: `gdn_recurrence_tp`, `tp_y16_pack`, `tp_gdn_merge`, `help_remap`, the TP
   chunk loop (`ours` prefill/kernels.cu has them; the merged tree's prefill.hpp still
   declares them).
5. Acceptance: `--tensor-parallel 2` (batch 0), solo 1K/6.5K decode 134-150, 4 concurrent
   with `"parallel": 1` (queued) aggregate ~76, or TP beside batch if PR#15's route is
   taken; the target is decode aggregate >= 194 at 4 concurrent.
