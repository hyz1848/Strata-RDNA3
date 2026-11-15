# Strata-rdna3 merge notes (2026-11-16)

Source: https://github.com/nicejunjie/Strata-rdna3 (fork of Niko1221/Strata, RDNA3/gfx1100 tuned).

## What was merged

- Base: our 0.1.59 tree (`/opt/strata-0213`, deployed engine 0.1.59) committed on the fork's
  merge base `8183c6b6` (upstream v0.2.4.1, 2026-10-24), then `git merge origin/main` (fork HEAD).
- Fork changes since that base: 114 files, ~19.6k insertions (49 new files).
- 29 conflicted files, ~150 hunks.

## Why the fork is faster (measured on the same DavidAU IQ3_S model)

| case | 0.1.59 (layer split) | fork / merged | gain |
|---|---|---|---|
| 1K prefill | 527 tok/s | 1,253-1,339 | 2.4x |
| 6.5K prefill | 1,953 tok/s | 2,995-3,029 | 1.55x |
| 1K decode | 88 tok/s | 146-162 | 1.7x |
| 6.5K decode | 84 tok/s | 134-137 | 1.6x |

Causes, in order of impact:
1. `--tensor-parallel 2`: decode runs every layer on both cards (attention split by head,
   experts split by id, one exchange per layer over PCIe). Replaces the pipelined layer split.
2. WMMA matrix-core prompt attention (`STRATA_HIP_WMMA`, `STRATA_SELECT_WMMA`, rocWMMA).
3. K-quant MMQ prompt experts (`STRATA_PREFILL_MMQ`, `STRATA_MMQ_KQUANTS`), FP16 pair kernels
   (`STRATA_PF_F16`), fused router/combine (`STRATA_ROUTE_PUBLISH`, `STRATA_COMBINE_CPU`).
4. MTP draft vocab subset (`--mtp-draft-vocab`, 60k ids instead of the full table) and
   `--mtp-q4 all`, `--mtp-window 4096`.
5. TP VRAM aliasing (`STRATA_TP_ALIAS`, `_SHARE_DP`, `_ALIAS_MAP=rows`): ~1.6 GiB/card back to
   the expert cache (94-98% hit rate observed).
6. `STRATA_GFX1100_DEFAULTS` table in `arch_defaults.cpp` turns all of the above on by default.

## Merge decisions (the risky part)

- Taken from the fork wholesale: `src/prefill/prefill.cpp`, `src/program/generate.cpp`,
  `src/core/verify.cpp`, `include/strata/core/verify.hpp`, `src/prefill/kernels.cu` (GDN
  recurrence), `src/prefill/moe_mmq.cu`. Both sides had rewritten these; a hunk-level merge
  was not safe. The fork's generate.cpp already carries `STRATA_SPEC_CUM`.
- Merged by hand: `arch_defaults.{hpp,cpp}`, `expert_source.cpp`, `mtp.{hpp,cpp}` (both
  `DraftResult` and `between`), `peer_experts.cpp`, `native_head.cpp`, `verify_kernels.cu`
  (template `NT` + `partial`), `qsa_prompt_attn.cu` (ours' gate path + fork's WMMA launch),
  `shared_expert.cu`, `native_bf16.cu` (fork's aux rows), `elementwise.cu`, `native_moe.cu`.
- Headers: `verify.hpp` kept ours' 2-D `[sel_long_][T]` graph tables + the fork's `long_`
  twins (`ex()`/`lau()` now index both).
- CMake: our refactored `cmake/strata_gpu_engine.cmake` + fork sources (`tp.cpp`,
  `arena_shm.cpp`, `sdma_copy.cpp`, `kfdq.cpp`, `rdna_mmvq.cu`, `rdna_gr.cu`,
  `megakernel_gdn.cu`), HSA link, `--export-dynamic-symbol=ioctl`, rocWMMA include dir.
- Our `serve/server.py` kept (max_tokens fuse) + the fork's `EngineError` and
  `_raise_if_fatal_gpu` both kept.

## Build

```
R=/opt/Strata-main/.venv/lib/python3.12/site-packages/_rocm_sdk_devel
export HIP_PLATFORM=amd HIP_COMPILER=clang HIP_RUNTIME=rocclr ROCM_PATH=$R HIP_PATH=$R
export PATH=$R/bin:$R/llvm/bin:/opt/Strata-main/.venv/bin:$PATH
cmake -S . -B build-merge -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_PREFILL_MMQ=ON -DSTRATA_MMQ_KQUANTS=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_HIP_COMPILER=$R/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT=$R \
  "-DCMAKE_HIP_FLAGS=-isystem $R/include --rocm-path=$R --rocm-device-lib-path=$R/lib/llvm/amdgcn/bitcode" \
  -DSTRATA_ROCWMMA_INCLUDE_DIR=/opt/rocWMMA/library/include -DCMAKE_PREFIX_PATH=$R
cmake --build build-merge -j12
```

Notes:
- `-isystem $R/include` is required; without it HIP picks `/usr/include/hip` and fails on
  `__ocml_exp10_f32`.
- rocWMMA headers: `/opt/rocWMMA` (commit 1ab208f4, tag rocm-7.1.1).
- Runtime needs `LD_LIBRARY_PATH=$R/lib` (libhipblaslt).

## VRAM

`--vram-reserve-mib 1024` leaves ~964 MiB free per card with the 84.8G DavidAU pack.
512 left ~100-450 MiB and is not safe for production.

## Risks

- The fork's base is upstream 0.1.49; our 0.1.50-0.1.59 fixes are lost in the files taken
  wholesale (notably verify.cpp features: STRATA_GRAPH_PRIO, STRATA_SELECT_LONG,
  STRATA_LAUNCH_PROBE, the ProposalBorrow cleanup; prefill.cpp: STRATA_KVA, PLE compact,
  layer-major prefill).
- The merged tree builds and runs, but is not yet soak-tested with vision, batch and 512k.

## Rollback

Production still runs `/opt/Strata-main/engine/strata` (0.1.59). The merged binary is
`/opt/strata-rdna3-fork/build-merge/strata`; switching is a config `exe` change.

## Production switch (2026-11-16)

- Unit `strata-engine.service` now runs
  `--config /opt/strata-rdna3-fork/strata-davidau-merged.json` (backup:
  `~/.config/systemd/user/strata-engine.service.bak-20261116`).
- Config: merged engine, TP 2, layer split 24, 512k context (yarn 2.0), vision,
  reserve 1024 (940/876 MiB free), `"parallel": 1`, env `STRATA_REFILL_SDMA=0`.
- Verified: facts 10/10 (2 misses were formatting: H2O/3e8), vision reads the
  test image correctly, 1K/6.5K decode 135-138 tok/s, prefill 936/2,858 tok/s.

### `"parallel": 1` is required for TP

`serve/server.py` appends `--batch N` when `"parallel": N > 1`. TP is only taken
when `o.batch == 0` (`generate.cpp`: `tp_on = ... && o.batch == 0`), so
`"parallel": 4` silently turns TP off (decode falls back to 84 tok/s). With
`"parallel": 1` concurrent requests are queued and each runs at full speed.

### Community findings (repo issues/PRs, 2026-11-16)

- #14: the gfx1100 TP defaults set `STRATA_REFILL_SDMA=1`; on ROCm 10.x the SDMA
  engine is unavailable (`engine -1`), each refill tranche waits 30 s and the 60 s
  watchdog aborts the engine. We saw `engine -1` in our log. Fixed by
  `STRATA_REFILL_SDMA=0` (refill falls back to hipMemcpyAsync).
- #16: TP watchdog can fire on ~200K-token prompts (60 s without progress); the
  supervisor restarts the engine. A 512k context can hit this.
- #17: kfdq idle-power issue only when `GPU_MAX_HW_QUEUES=8`; we do not set it.
- PR #15 (open, unmerged): `STRATA_PW_WITH_BATCH=1` keeps the solo decode
  pipeline while batch slots are enabled: solo +43-62%, two-client aggregate
  unchanged. This is the path to fast solo AND true concurrency, but it is a
  16k-line unmerged PR; evaluate separately.

### Concurrency measurements (merged engine, DavidAU IQ3_S)

| mode | solo decode | 4 concurrent total |
|---|---|---|
| TP 2, no batch (`parallel: 1`) | 135-146 tok/s | 4.7 s, queued (76 tok/s aggregate) |
| layer split + `--batch 4` | ~44 tok/s | 3.6 s (89 tok/s aggregate) |

The batch path in the fork base (0.1.49) lacks upstream 0.1.50-0.1.59 batch
optimizations, so it is 3x slower solo. PR #15 targets exactly this.

## Second pass: 0.1.59 preserved in verify.cpp (2026-11-16, evening)

The first merge replaced `verify.cpp`, `prefill.cpp` and `generate.cpp` with the fork's
copies wholesale. That was wrong: it dropped 0.1.59 code. Corrected so far:

- `verify.cpp`: real 3-way merge. Kept 0.1.59's `capture()` / `capture_all()` / `pl_launch()`
  (STRATA_SELECT_LONG two-variant graphs, STRATA_GRAPH_PRIO node priorities, launch probe,
  `instantiate_evicting` flags, 2-D `[sel_long_][T]` tables + the fork's `long_` twins) and
  the fork's `record_window()` (TP/DP/peer/megakernel). Header `verify.hpp` is the merged one
  (both table sets, `ex()`/`lau()` indexing `sel_long_` and `long_`).
  Parity: verify_parity, verify_batch_parity, spec_verify_parity, gr_parity, gdn_parity,
  arch_defaults_test all PASS; engine decode 135-150 tok/s at 1K/6.5K.
- `generate.cpp`: still the fork's copy, with one API fix (`AdaptGate::Tick` is
  0.1.59's `{counter, layer}` in `verify.hpp`; the fork's copy used `{gate, layer}`).
- `prefill.cpp`: still the fork's copy.

Still to do (per-function, not hunk-level): the fork rewrote the decode loop in
`generate.cpp` (TP/DP/peer) and the prompt loop in `prefill.cpp` (TP/DP/F16/refill) where
0.1.59 has KVA, layer-major prefill, PLE-compact and DFlash2 wiring. Those features are all
opt-in and unused by production, but they must be reconstructed function by function to be
preserved. Current production behaviour is unaffected.

## generate.cpp / prefill.cpp: why a hunk merge is not valid

Attempted the 3-way merge of `generate.cpp` (24 conflicts) with per-hunk choices. It does
not compile, and the failures show why:

- The two versions restructured the same decode loop. Ours (0.1.59) declares state in the
  conflicted blocks that later (auto-merged) code uses: `df2_solo`, `df2_vocab_path`,
  `df2_deferred`, `dfr_n`/`dfr_pos`, `arm_prompt_helpers` + `paths`. Taking the fork's side
  of those hunks leaves dangling references; taking both duplicates the loop structure.
- APIs differ: ours' `Prefill::init(..., prefill_head_alias)` does not exist in the merged
  `prefill.hpp` (the fork's `init(..., res_of_stage)` won), and ours' code calls
  `Prefill::counter` / `set_spec_sampling` on `sp`, which the fork's Prefill does not have.
- `verify.cpp` worked because its conflicts were separable: `capture()`/`pl_launch()` vs
  `record_window()`. `generate.cpp`/`prefill.cpp` are one interleaved loop.

Correct approach (what remains): reconstruct the two loops function by function, starting
from 0.1.59's `run()`/prompt loop and porting the fork's TP/DP blocks into it (or the
reverse), with the missing Prefill/Verifier APIs added back as needed. This is a dedicated
task; the current tree keeps the fork's loops and 0.1.59's headers, which is consistent and
passes the parity suite.

Current tree state (commit efbbe2d5 + notes): builds clean, verify.cpp merged, production
running on it.
