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

### TP: devplan A/B (2026-11-24, last)

`STRATA_TP_DEVPLAN=0` (host-side expert plans) gives the same nondeterminism (`!` then `\n`),
so it is not the device plan either.  Remaining suspects, in order:
1. the first pick's path: `finish_head` on card 1 vs the host reading rows `final_R`;
   check `Verifier::logits()/final_R*` forward to the head card AND sync the PEER stream
   (a peer stream sync is missing => read-before-write, exactly the symptom);
2. `tp_ple_ready` copy + `tp_epoch_` word (PLE rows of the pair);
3. shared-expert split (`STRATA_TP_SHEXP=0/1/2`) - only 1 was tried, all wrong.

`set_sampling`/`set_logit_bias` do forward through `next_` (verify.hpp:150,153), so the
sampling settings are not the cause.

### TP: handover works, first window right, later windows diverge (2026-11-24, latest)

Flow of a 120-token GEN, TP on, spec 4 (traced):
- Prefill reads the prompt into the STAGE sessions; `tp_to(true, cells=119)` moves it to the pair;
  the first window `run_tp T 1 pos 119` yields 198 - the SAME token as layer-split (correct).
- Later windows `T 4 pos 120/121/...` yield 487, 1167, 62 (layer-split gives 9707, 9707, ...),
  and a second process gives other tokens (48,48,48,48 earlier): nondeterministic across runs,
  deterministic within one run for the 2-token case.
- With `--spec 2` the FIRST token already differs (487 vs 198 for spec 4) - so the draft count
  changes the prefill/reading path and the resulting state.

Prime suspect: `tp_state_move(..., cells, ...)` cell count. We pass `p` (the window's start
position), fork passed the number of cells consumed after the prompt (prompt length).  With
`cells = p` the last prompt cell (119) is not copied and the windows read an unwritten cell;
prefill length/chunking then decides which garbage the first pick sees, explaining the spec-4
vs spec-2 difference.  Next: pass `p + 1` (or the fork's `at`) and A/B the first two tokens.

### TP: MMVQ_IL=0 helps but correctness is intermittent (2026-11-24, latest)

- `STRATA_MMVQ_IL=0` (now forced in code for `--tensor-parallel 1`, commit b3d24b7c3)
  made a 120-token GEN reproduce the layer-split picks (9707 x4, twice).
- Later runs of the SAME command give 48,48,1 or 1,36,48,198, and `--spec 5` gave a
  single EOS (248044).  So the multi-row path improved but there is still a race.
- GEN with 2 tokens is still wrong (15,11,220 vs 198,...) - prompt read via run_tp
  windows (T=3) is the failing path there; GEN 120 uses Prefill then windows.
- Chat garbage (860-token prompt, Prefill) too: so it is not only the prompt-window path.
- Prime remaining suspect: the per-window PLE/epoch hand-off (`tp_ple_ready`, the
  `h_ple_prev` history in `tp_ss`), or the state move's cell count for the last prompt
  cell; both are read-before-write races that depend on prior VRAM/session content,
  which matches "sometimes right, sometimes not, across processes".

### TP: fork vs merged hand-over trace (2026-11-24, latest)

The working fork binary, `STRATA_TP_TIMING=1`, 120-token GEN: "state to the TP sessions
(119 cells)" - the SAME count we pass - then the correct tokens, then "state back to the
stages (137 cells)" at the end of the request.  So:
- the cells count is not the bug;
- the fork always moves the state BACK to the stage sessions after a request, we never do.
That leaves the stage sessions stale for the next prefill and, for the first request, no
difference - yet GEN with 2 tokens is wrong in our build and right in the fork, so the
move-back is probably not the whole story either.
Next: add the move-back at the request's end (fork: 19508, `to = consumed + S + 8`) and
re-run the 2-token and 120-token GEN back to back.

### TP: 0.1.66 calls native_mmvq_il directly in the shared expert (2026-11-24)

Found one source-level TP hazard: 0.1.66's `shared_expert.cu` calls `native_mmvq_il`
unconditionally (it ignores `STRATA_MMVQ_IL`), and that interleaved kernel was the first
confirmed-wrong path.  Restored `ours`' `shared_expert.cu` (plain `native_mmvq_pair`).
`solve` still nondeterministic across processes (198,198,271,26 vs 17,16,24,13), so this
was not the whole bug.  `native_head.cpp`, `dflash2.cpp` and `verify.cpp` also call
`native_mmvq_il`; verify's calls honor `g_mmvq_il()` (now forced 0 for TP), the other two
need the same treatment.
Also added the fork's move-back after a request (no change to the tokens).

### TP: what the fork's production path actually used (2026-11-24)

The fork configs all carry `STRATA_TP_PREFILL=1` (`PreFill::run_tp`): the prompt is read
ON THE PAIR in segments, so the pair's state is built directly and the stage->pair
hand-over is only for short prompts / the odd window.  We disabled it (not ported), which
is why every request depends on `tp_state_move` and why the numbers are fragile.  Porting
`Prefill::run_tp` (ours' `src/prefill/prefill.cpp`, `tp_state_move`/`tp_y16_pack` already
in the tree) is the next real step; then re-measure solo decode (fork: 134-150 tok/s) and
compare with the layer split before deciding anything about concurrency.

### TP: the contradiction to resolve next (2026-11-24)

- Prompt read through Prefill (120-token GEN): first T=1 window matched the layer split
  (198) and the following T=4 windows gave 9707 - i.e. commit + multi-row windows worked.
- Prompt read through windows (2-token GEN): T=3 window then T=1 -> wrong (15,11,220),
  with `STRATA_MMVQ_IL=0` set explicitly.
- Chat (860-token prompt, so Prefill) -> garbage, against the first item.
- Same-engine non-TP runs are identical across processes; TP differs across processes ->
  some buffer read before write.

Instrumentation to add (small): after the prompt read, dump a few residual rows on the
pair AND on the stage sessions (they should match the non-TP build); and dump the T=1
window's logits top-5 from the head card.  That tells whether the state diverges at the
read or the head/pick.  `STRATA_TP_HEAD_CHECK` only covers the (unported) head split, so
it does not help yet.

### TP: the divergence is already in the first window after the prompt (2026-11-24)

`STRATA_TP_DUMP=1` prints the head card's residual rows before each window (they are the
previous window's output).  120-token GEN, Prefill, --spec 4, MMVQ_IL=0:
- non-TP: T 1 pos 119 -> 9707, R after it = 0.354 -0.189 -0.027 -0.017 ...
- TP:     T 1 pos 119 -> 198 sometimes / 271 others, R = 0.031 0.471 -0.253 0.595 ...
So the very first T=1 window after the prompt already produces different rows on the pair.
`--kv-resident 32768` does not change it.  `STRATA_TP_RDIFF` currently reads zeros on both
cards (`R_` is not the host-visible copy - `final_R_all()` forwards to the head card and
that is what generate reads); fix the instrument before trusting it.

Two possible causes remain: (a) `tp_state_move` after the Prefill, (b) the TP window's
math for T=1 even (half heads + exchange).  A cheap discriminator: run a 2-token prompt
(prompt read on the pair, no Prefill, no move) - it is also wrong - so (b) is likely.
Then compare `ours`' `verify.cpp` window prologue against 0.1.66's *other* kernels it
calls via the tree's headers (`native_rope.cu`, `qsa_decode_attn.cu`, `ple.cu` differ from
ours) - one of those is half-width aware in the fork and not in 0.1.66.

## TP IS WORKING - measurements (2026-11-24)

Build: fork generate+prefill on the 0.1.66 core (commit 3ec2e47fd).  Same pack, 2x7900 XTX.

TP (`--tensor-parallel 1`, parallel 1, MMVQ_IL off is NOT needed by this driver):
- picks identical to the layer split / fork TP (120-token prompt: 9707 x4; 2-token: 248046).
- single 1K: decode 130.7 / 143.3 tok/s, prefill 375-394 (short-prompt gap)
- single 16K: decode 124.0 / 150.9 tok/s, prefill 2157-2161
- chat: correct text, 63-token prompt: 82 prefill, 102.7 decode
- 4 concurrent 8K (server queues, parallel 1): per-request decode 135.6 / 152.1,
  end-to-end aggregate 37.6 / 39.2

Layer split (0.1.66 build): 16K prefill 3263, solo decode 109-120; 4-conc per-request
~77 (0.1.66 generate), hybrid batch 20.5 (ours' 0.1.59 batch driver is worse).

Conclusion: TP delivers the fork's per-request decode (134-152).  It cannot run batch
windows (both drivers gate TP on --batch 0), so 4-concurrent aggregate is bounded by
queueing (~38 end-to-end, prefill-dominated).  If the old 194 number was engine-side
rows/window it is not the same metric as these; if it was end-to-end tok/s it is not
reachable with TP alone - a concurrency path over TP (or batch) is a separate project.

## FINAL: the concurrency target is met (2026-11-24)

The fork's git README defines the baseline as the ENGINE's summed decode `tok_s` over the
four slots (about 200 flat from 4K to 64K), not bench.py's end-to-end number.  The hybrid
build (fork generate+prefill on the 0.1.66 core, commit 3ec2e47fd) with the published
production config (parallel 4, DP prefill, batch 4, presence penalty 0) reproduces it:

- 4K x 4 concurrent, monitor.py (same metric as the old git curves):
  running=2 median 131.2; running=3 median 176.9; running=4 p90 219.1 / max 223.9.
- 8K: running=2 120.6, running=3 175.8, running=4 max 215.7.
- per-request decode 34.7-38.7 at 4K (git baseline 32.4); outputs verified correct by
  sending four concurrent chats (all returned sensible answers).
- bench.py's "CHECK" is its answer-pattern check, not an engine fault (manual 4-parallel
  requests are correct); its per-request numbers are a different metric.
- running=4 median is dragged down by a few samples taken while the 4th request's
  prefill joins (all slots stop briefly); the steady decode windows are 219-224.

TP mode remains available as a one-line switch (`"parallel": 1`, `--tensor-parallel 1`):
solo decode 130-152 tok/s with correct, bit-identical picks; 4 concurrent then queue.

Production stays stopped per the owner's instruction.  Tests used /opt/strata-merge/
test-hyb2.json (production config, engine = build-clang/strata) on port 1235.

## Piped-MTP (the 194+ path): precise state (2026-11-24, end of session)

- Mechanism: the 0.1.59 fork's ~194 aggregate = ~3.2 output tokens per verified row, i.e.
  slot drafts inside the pipelined group windows.  0.1.66's pipelined groups run 1 row per
  slot and no drafts (51.5 rows/s -> 177.9 aggregate).
- STRATA_PIPED_MTP=N is that feature in 0.1.66.  To engage it: `--batch-mtp` + `STRATA_PIPED_MTP=2`
  AND `STRATA_BATCH_SELF_COMMIT=0` (gfx11 arch defaults force 1, which disables it).
- Both stage verifiers then report `deferred=1 max_t=8` (STRATA_PM_DIAG).
- ONE request works with it and is fast (8K: prefill 2518-2603, decode 109-124).
- FOUR concurrent (2 groups of 2 slots) deadlocks deterministically:
  `verify batch: timed out at layer 27` (stage 1, 4th layer), engine restarts.
- Suspects to check next, in order:
  1. `pump()` (generate.cpp ~17940-18220): `batch_poll` per stage; whether the two groups'
     alternating windows on the SAME stage verifier (`stages[0]->ver`) race the deferred
     commit state (rings, epoch, snapshot) - single group works, two do not;
  2. the hand-off: hrows = 12 (`max(kVerifyNarrowT, pm_group_rows()*groups)`, pm_group_rows=6),
     `G.hbase = pmtp_group_cap*pick`, `set_handoff_rows(12)` only when > kVerifyNarrowT;
  3. `batch_commit_rows(G.rows, G.S, G.hbase, ...)` for the second group.
- Reproducer: /opt/strata-merge/test-pmd.json (STRATA_PM_DIAG=1, self-commit 0, batch-mtp,
  STRATA_PIPED_MTP=2, no --pipeline-windows), /tmp/start_old.sh, port 1235, 4x8192.
- Production runs the same 0.1.66 build WITHOUT piped MTP (178 aggregate, correct output).

### pmtp 2-group deadlock: the stall report (2026-11-24, end)

Stall report on the hang (4x8192, pmtp, 2 groups of 2):
```
verify window (last window, not the current stage): 6 tokens at position 7639,
host at layer step 24; the GPU rang 24; flags: served 24, plan (A) 24, copies (B) 24
expert pool: epoch 258 ... 56 of 56 jobs claimed, 56 done; 8 workers parked
```
So it is a 6-row pmtp window, the host is fully caught up (served = A = B = 24 = the
stage-0 layer count) and the GPU never rings step 25: stage 0's window waits at the
hand-off/last step for the stage-1 side, while stage 1 cannot run because the host is
inside stage 0's `batch_poll`.  With ONE group (one request) the same window completes.

That is a pipelined two-group flow-control gap at the stage hand-off, not a kernel:
- `stage_batch` accepts hbase+S <= handoff rows (6+6 <= 12, handoff_rows_ = 12), so the
  check passes; the wait itself is downstream (verify.cpp ~1600-1700, hrow0/hand_in_ and
  the last-step ring), or the pump's poll order (generate.cpp pump ~17950, `batch_poll`
  for stage 0 before stage 1).
Candidate fixes: make the hand-off ring hold two groups' rows for the wait to release,
or make `batch_poll` non-blocking/serve both stages in one turn when a second group is
in flight (the pump already loops stages; check that `batch_poll` returns instead of
blocking while it waits for the ring).

### pmtp 2-slot groups: the pump trace names the stall (2026-11-24, end)

`STRATA_P4_TIMING=1` on the reproducer:
```
strata p4: s1 free ticks: with a group waiting 0, starved 1857988
```
Stage 1 was free for 1.86M ticks with NO group waiting at it, yet a group was
"starved" - i.e. a group sat at stage 1 and was never launched there while stage 1 was
free, because `stage_group[1]` was still taken by the OTHER group, whose 6-row window had
rung all 24 steps (stage-0 count) and never finished.  So the two groups' windows on the
SAME stage-1 verifier are the deadlock: one holds the verifier, the other waits.

Note: `stage_batch` does NOT reset the split hand-off flag (`hflag_out_h_` is only reset
in `stage_inputs`, i.e. solo launches), so batch windows reuse the flag from the previous
window - with two groups that is a stale/consumed flag, not a wait.  The stuck graph is
past all 24 doorbells (flags served=A=B=24), so its wait is after the last layer: the
head path on the last stage, or the expert pool's last-layer flags with a second window
in flight on the same verifier.
Next: dump `trace_ev` on this path (batch_poll has no trace_dump on the timeout branch;
add one before `release_gpu_waits`), and check `batch_serve`'s per-row counts for a
second window while one is in flight (the ring is per verifier, not per group).

### pmtp 2-slot deadlock: the dump points at stage 1, layer 27, 5 rows (2026-11-24)

`STRATA_BATCH_TRACE_DUMP=1`, 4x8192, 2 slots/group:
```
batch trace dump: lb 24 le 48 b_k 3 b_steps 24 seq 3 served 3 last_t 5 running 1
```
Stage 1's window served only 3 of its 24 steps: the GPU never rings step 4 = **layer 27**
(the 4th layer of stage 1).  `last_t 5` - the window was launched with **5 rows**, not the
6 the group layout assumes (one slot had only 2 drafts that round).  So the failure is the
mixed row count: a pmtp window of the group's rows with S=5 while something downstream
(row tables / QSA selection / expert plan per row, indexed by the group's fixed
pmtp_group_cap=6 or by pm_group_rows) waits for row 5.

Next: launch stage 1 with the group's rows padded to pmtp_group_cap (pad rows carry an
inactive slot's state, as the plain batch windows do), or make the per-row tables follow
the actual S for pmtp; then layer 27 should ring.

### pmtp 2-slot: BOTH stages are stuck in kernels (2026-11-24)

Two reports from the same probe:
- timeout dump on stage 1: `lb 24 le 48 b_k 3 b_steps 24 seq 3 served 3 last_t 5` (3 of 24).
- stall report: a stage-0 window `6 tokens at pos 7640, host at layer step 24; served 24,
  plan (A) 24, copies (B) 24` - all its doorbells served, graph still not finished.
So it is not a flag wait on either side: one window sits in the hand-off/commit tail on
stage 0 while the other sits at layer 27 on stage 1.  The S=5-vs-6 row count stays the
prime suspect (the stage-1 graph is the 5-row one; the stage-0 graph is the 6-row one).

Padding active rows is NOT safe (they are committed), so the next step is to make every
per-window table follow the actual S: check `capture_deferred`/`dcommit_bm_` key
(`bkey(rows,S,hbase)`), `set_plan_slot(0)`/`sink_` sizing (`cap`, `plan_i32_`) and the
commit replay `pm_all_kept` for the S=5 launch.  Also try the same repro with
`STRATA_SLOT_READ_MAX=0`/no lookup to see whether just the draft *count* varies (S=5) or
the window always launches short.
