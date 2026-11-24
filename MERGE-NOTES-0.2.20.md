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

### Torch-2 status of the TP port (do not redo)

- `baba3bba3` tp kernels; `e55d737f8` 0.1.66 Verifier = fork TP verify (builds);
  `c5d034156` `--tensor-parallel [1|2]` parsed.
- Remaining: generate/prefill run wiring. 0.1.66 replaced ours' `ss`/`wt`/`stages[0]`
  locals with `GpuStage { dev, wt, head, ss, ver, ver_b, sp, ... }` (generate.cpp:1714),
  and the decode loop constructs a local `Verifier ver` at :13150 (serve path :24397).
  Map ours -> 0.1.66:
  - ours `ss` (main session) -> the text stage's `GpuStage::ss`; ours `wt` -> `GpuStage::wt`
  - ours `stages[0]->ss/wt/head/stream` -> the second `GpuStage`
  - ours `Prefill sp` -> each stage's `sp` (Prefill::set_stage already splits layers)
  - ours' TP setup block (generate.cpp ours 7058-7300: tp_geometry/session_init per
    card, PleRun twice, TpWeights, head half, TpLink, set_tp/set_tp_head_split) inserts
    after stage sessions init in 0.1.66 (after generate.cpp:7672 session_init) and before
    the expert caches take VRAM.
  - ours' handover/loop hooks: 12427 set_tp, 12431 set_tp_head_split, 12467 tp_to,
    12514 tp_state_move, plus the prefill TP branch 18360/18518; port into the 0.1.66
    loop's window call sites (search `ver.run(` / `batch_launch`).
  - prefill: ours' run_tp/tp_state_move in src/prefill/prefill.cpp -> 0.1.66 Prefill.
- Then the old fork TP acceptance: `--tensor-parallel 2`, solo decode 134-150 at 1K/6.5K.

### TP runtime status (2026-11-24)

- `4be1c0b54` sessions+weights+verifier pair; `f...` tp_to hooks; builds and STARTS:
  `strata serve ... --tensor-parallel 1 --draft mtp` (needs `STRATA_MTP_KEEP=1` so the
  drafter survives the memory decision, and `STRATA_STAGE_TRIM=0` so both arenas hold
  every layer).
- It then HANGS in the first window: `strata verify: timed out at layer 0` (both card
  verifiers entered run; rank 0 drives the pair as in the fork).  Next: compare
  `Verifier::run`'s TP prologue against ours' generate-side requirements (`tp_pre_join`,
  the PLE flag at verify.cpp:1631, tp_pages flag init) - the pair is stuck on the first
  exchange.  A/B: STRATA_TP_INBOX=vram, and print tp_k/tp_pub at the first layer.

### TP debug 2026-11-24 (later): progress

- fix: the pair's verifiers must run EVERY layer (`ver.set_stage(0, g.n_layers, nullptr, nullptr)`
  before init); rank 0 had kept [0, 24) and the two cards' exchange counts diverged.  After the fix
  both ranks finish all 96 exchanges of the 7-token prompt with the same pub total (1152), so the
  layer loop and the exchange protocol are correct.
- The window then still times out: the stall is after the exchanges, inside `serve_rings` or
  `finish_head` (verify.cpp run_tp: launch -> serve_rings(first) -> serve_rings(second) -> sync ->
  finish_head).  Next: trace each card's serve_rings entry/exit (STRATA_TP_TRACE) and the PLE
  flag word (`h_ple_flag_`, written try_run_tp: `tp_epoch_ + 1`) to see which ring/hand-off spins.
- Repro config: /opt/strata-merge/test-tp1.json (`--draft mtp`, drafting=mtp, parallel 1,
  STRATA_MTP_KEEP=1, STRATA_STAGE_TRIM=0, max-context 65536), start with /tmp/start_tp1.sh.

### TP hang root-cause direction (2026-11-24, last)

The first window deadlocks before the GPU rings layer 0 (`serve_rings` waits `*seq`,
nothing advances; prompt read then cancels after 35 s).  Both graphs are captured and
launched, both ranks run the same exchange sequence at capture.  A/B `STRATA_TP_INBOX=vram`
changes nothing.

This points at the compute kernels, not the wiring: the fork's `kernels.cu` had TP hooks
(the out-projection partial sums that publish at each layer and the expert owner's send /
the other card's receive) and the merged tree took 0.1.66's `kernels.cu`, which has no TP
code.  The exchange kernels enqueued by verify run, but the layer's compute never reaches
the doorbell.  Next: diff `ours:src/kernels/cuda/kernels.cu` against the tree's, port the
TP hooks (grep `tp_` / `tp_xchg` partners: the dense out-proj, the MoE owner post, the
shared-expert split), then rerun `test-tp1.json`.

### TP: the window now runs (2026-11-24, latest)

- Big fix: `Verifier::init` reads `cudaGetDevice()`; the peer was init'ed while CUDA0 was
  current, so both graphs lived on card 0.  Wrap each init in `OnDevice(tp_dev[r])` (and the
  per-rank set_tp loop too).
- After that the 7-token window COMPLETES: both serve_rings, finish_head, picks.  The
  process then dies with SIGSEGV (exit -11) right after `finish_head` returns and before the
  generate-side post-run hook: inside run_tp's tail (`VDBG`, `progress_at("decode")`,
  `progress_beat`) or in a host thread (watchdog / release_gpu_waits) racing it.
- Next: run the serve path under gdb (`gdb -batch -ex run -ex bt --args <engine args>
  --serve`) on the 7-token request, or bisect the three lines of run_tp's tail.  Config:
  test-tp1.json (drafting mtp, parallel 1, STRATA_MTP_KEEP=1, STRATA_STAGE_TRIM=0,
  STRATA_TP_TRACE=1), /tmp/start_tp1.sh.

### TP: runs end-to-end, numerics wrong (2026-11-24, latest)

`2be8697ce` + the handover-hook fix: a request now completes through the TP pair
(`GEN` returns DONE, ~169 ms for the 16-token window) but the tokens are GARBAGE
("， the best part of noniss..."), including the verify-window prompt path with no
Prefill.  So the wiring is right and the *compute* is wrong.

Checks for next session:
- First-token A/B: same 2-token prompt, TP on vs off (--tensor-parallel 0) - the first
  pick differs, so it is in the window, not in the state handover.
- Only `qsa_select.cu` and `verify_kernels.cu` carry `tp_` in `ours`; the merged tree
  has ours' verify_kernels (all TP kernels) and ours' qsa_select (59 tp_ refs).  The
  dense/MoE numbers come from verify.cpp + those kernels; compare the tree's
  `verify.cpp` TP call sites (kVerifyNarrowT paths, `tp_shexp_split`, `tp_gpu_only_`,
  the out-proj half width) against `ours` one by one - the likely drift is a call that
  0.1.66 renamed/reshaped.

### TP numerics A/B (2026-11-24, last)

Same 2-token prompt, max_tokens 1, temp 0:
- `--tensor-parallel 0` (everything else identical): `We`, `We` - deterministic.
- `--tensor-parallel 1`: `!` then `\n` (two identical requests, different picks) - the TP
  window is nondeterministic, so something is read before it is written.
- `STRATA_TP_INBOX=vram` gives the same nondeterminism (so it is not the pinned-page path).
- config pairs: /opt/strata-merge/test-tp1.json / test-tp0.json, /tmp/start_tp1.sh.

Next: the expert path (pool plan vs the pair's own caches), `tp_shexp_split`, and the PLE
copy (`tp_ple_ready`); also `STRATA_TP_HEAD_CHECK=1` (compare the head card's pick with
the full head) to see whether the residual rows are already wrong before the head.
