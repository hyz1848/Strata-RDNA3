# Strata-RDNA3

RDNA3 (Radeon RX 7900 XTX, gfx1100) optimizations merged onto the current Strata engine, with
batch concurrency that raises aggregate throughput above a single stream.

This build runs Qwen3.8-Flash-Next IQ3_S with MTP speculative decoding, vision, a 512k context
and four concurrent request slots on two RX 7900 XTX cards.

## What this build adds

1. **The RDNA3 work, on current Strata code.** The gfx1100 optimizations from the Strata-rdna3
   fork (tensor-parallel and data-parallel prompt paths, fused int8 expert kernels, MMQ prompt
   tiles, hipBLASLt tuning tables, expert-cache policies, MTP slot drafters, wide verify windows
   of 9-16 rows) are merged into the current Strata engine source (0.1.59) instead of living in a
   fork that trails the engine. The tree builds, passes the engine's parity tests, and serves.

2. **Concurrency that pays.** Four batch slots plus the solo path for a lone request. A single
   request keeps MTP and decodes at 120-146 tok/s. Four concurrent requests decode at about
   **190 tok/s aggregate** (decode phase, +40-50% over one stream) while each request still runs
   at 30-48 tok/s. A request that arrives while others decode joins a batch slot instead of
   queueing behind them.

## Credits

This project is a merge, not a rewrite. It would not exist without:

- **[Strata](https://github.com/Niko1221/Strata)** by Niko1221 and contributors - the engine,
  its kernels, the batching and speculative-decoding design. MIT licensed.
- **[Strata-rdna3](https://github.com/nicejunjie/Strata-rdna3)** by nicejunjie and its
  contributors (cennac, Vop3p and others) - the gfx1100 tuning this build is based on:
  hipBLASLt tables, MMQ prompt tiles, the TP/DP prompt paths, expert-cache policies, the batch
  solo-path work and the wide-window kernels.
- **rocWMMA**, **ggml / llama.cpp** kernels, and **AMD ROCm** for the lower layers.

If you use this work, please credit Strata and the Strata-rdna3 fork first.

## Hardware and software

| | |
|---|---|
| GPUs | 2x AMD Radeon RX 7900 XTX 24 GB (gfx1100), PCIe 4.0 x8 each, no xGMI |
| CPU / RAM | Ryzen 9 9900X, 121 GB DDR5 |
| OS / ROCm | Ubuntu, kernel 7.0, ROCm SDK 10.3 (venv), in-kernel amdgpu |
| Engine | Strata 0.1.59 + the RDNA3 merge, HIP backend, gfx1100 |
| Model | DavidAU Qwen3.8-Flash-Next-Abliterated-Turbo-50x-Reasoning IQ3_S (GSQ-RCO), 84 GB with its 26.8 GiB PLE table |
| Context | 512k tokens, YaRN x2.0, int8 KV, 32k KV resident |
| Concurrency | 4 batch slots (`--batch 4`), MTP (`--spec 4`), vision |

## Features

- Layer split across two cards (24/24 layers), with the DP prompt path for long prompts.
- MTP speculative decoding: solo requests decode through the MTP head; batch slots use the
  batch-MTP path when enabled.
- Four concurrent request slots; a lone request automatically uses the solo path.
- 512k context with YaRN x2.0, int8 KV cache, conversation cache (8 slots).
- Vision (image input) through a separate vision encoder.
- Fused int8 expert kernels, MMQ prompt tiles, hipBLASLt tuning for gfx1100.
- Wide verify windows (9-16 rows) for wide batch windows.

## Configuration

Key engine flags used in production:

```
--layer-split 24 --batch 4
--prefill 18432
--spec 4 --spec-min-p 0.5 --mtp /path/to/mtp/rt --mtp-q4 all --mtp-window 4096
--lookup-chain 3
--kv int8 --kv-resident 32768
--max-context 524288 --rope-scaling yarn --rope-scale 2.0
--conversation-cache-mib 8192 --conversation-cache-slots 8
--expert-cache auto --ple-io direct --ple-row-cache 1048576
--pool-workers 8 --vram-reserve-mib 1024
--vision
```

Environment that matters on gfx1100:

```
STRATA_HIPBLASLT_TUNING=tools/hip/gfx1100-hipblaslt-100500.txt
STRATA_PREFILL_MMQ_TILE_ROWS=32
STRATA_PEER_TRANSPORT=host
GPU_MAX_HW_QUEUES=8
STRATA_REFILL_SDMA=0
```

Sampling: temperature 0, thinking off, `presence_penalty 0` (a nonzero penalty forces serial
decode and lowers MTP acceptance).

## Performance

Measured with `tools/bench.py` on the configuration above. Each prompt is a synthetic document
of N tokens with three 5-character codes at the start, middle and end; the model must return the
codes and then write an English guide (`prose`) or a Python module (`code`). A run counts only
when all three codes come back with exactly 256 output tokens. Prompt lengths are approximate
(about 4.8 characters per token); the tables show the server's actual prompt tokens. Median of
3 runs (2 for >= 64K).

### Single request

A lone request uses one slot and the solo MTP path.

| target | actual prompt | prefill tok/s | decode tok/s (prose) | decode tok/s (code) |
|---:|---:|---:|---:|---:|
| 1K | 859 | 640 | 127.2 | 146.2 |
| 4K | 3,244 | 1,772 | 128.5 | 138.2 |
| 16K | 12,777 | 2,757 | 121.6 | 141.9 |
| 32K | 25,486 | 3,478 | 122.0 | 135.9 |
| 64K | 50,908 | 4,746 | 123.7 | 144.2 |
| 128K | 101,745 | 5,383 | 120.8 | 139.4 |

Prefill rises with prompt length as fixed costs amortize. Decode is flat at 120-146 tok/s;
code drafts are accepted more often than prose (the MTP head).

### Four concurrent requests

Four requests launched together, each with its own codes and nonce. `aggregate (end-to-end)` is
4 x 256 output tokens divided by the wall time, prefill included. `per-request decode` is the
median of each request's decode rate. Median of 2 bursts.

| target | actual prompt | wall | aggregate end-to-end tok/s | per-request decode tok/s |
|---:|---:|---:|---:|---:|
| 1K | 860 | 10.6 s | 97.3 | 36.7 |
| 4K | 3,245 | 12.3 s | 83.0 | 33.1 |
| 16K | 12,777 | 23.0 s | 44.6 | 22.3 |
| 32K | 25,485 | 34.8 s | 29.5 | 16.5 |
| 64K | 50,908 | 48.8 s | 20.9 | 12.5 |

The decode-phase aggregate (the engine's `tok_s` over the four slots, excluding prefill) is about
**190 tok/s** at 4K, against 120-146 tok/s for one request: concurrency adds throughput instead
of replacing it. Per-request latency falls because the four slots share the two cards; the
end-to-end aggregate is lower than the decode aggregate when the four prefills serialize.

## Build

Requirements: ROCm SDK (HIP), CMake, Ninja, rocWMMA headers, and the model pack.

```sh
R=/path/to/rocm-sdk
export HIP_PLATFORM=amd HIP_COMPILER=clang HIP_RUNTIME=rocclr ROCM_PATH=$R HIP_PATH=$R
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_HIP_COMPILER=$R/llvm/bin/clang++ \
  -DSTRATA_ENABLE_HIP=ON \
  -DSTRATA_PREFILL_MMQ=ON -DSTRATA_MMQ_KQUANTS=ON \
  -DSTRATA_ROCWMMA_INCLUDE_DIR=/path/to/rocWMMA/library/include \
  -DCMAKE_PREFIX_PATH=$R
cmake --build build -j12
```

The engine binary is `build/strata`.

## Run

```sh
LD_LIBRARY_PATH=$R/lib ./build/strata --serve \
  --pack /path/to/pack --native /path/to/model-00001-of-00002.gguf \
  --ple-gguf /path/to/model-00002-of-00002.gguf \
  --mtp /path/to/mtp/rt --spec 4 \
  --layer-split 24 --batch 4 --prefill 18432 \
  --kv int8 --kv-resident 32768 --max-context 524288 \
  --rope-scaling yarn --rope-scale 2.0
```

Or through the Strata server, which adds the OpenAI-compatible API on port 1234:

```sh
python serve/server.py --engine strata --config strata.json --port 1234
```

## Repository layout

| Path | Contents |
|---|---|
| `src/core/` | verifier, prefill, MTP, expert sources, caches |
| `src/kernels/` | HIP/CUDA kernels (fused GR, router, MMQ, wide windows) |
| `src/program/` | `generate.cpp`, `verify.cpp`, server wiring |
| `src/prefill/` | prompt path, compatibility layer |
| `include/` | public headers |
| `rdna3/` | gfx1100 configs, tuning tables, MTP draft vocab, bench harness |
| `tools/` | `bench.py`, `monitor.py`, `iq_test.py` |
| `BENCH.md` | full benchmark notes and methodology |

## License

MIT, following Strata. See [LICENSE](LICENSE). Strata is Copyright its authors; the RDNA3
optimizations are from the Strata-rdna3 fork and its contributors.
