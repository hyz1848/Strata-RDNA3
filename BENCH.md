# Benchmark: DavidAU Qwen3.8-Flash-Next IQ3_S on 2x RX 7900 XTX

Measured 2026-11-16 on the production engine of this tree (`/opt/strata-rdna3-fork`,
engine 0.1.59 + the RDNA3 fork's kernels, merged). The numbers are for **this** model, not the
stock Qwen3.8-Flash-Next IQ3_S the upstream/fork tables use, and the packs differ (see Caveats).

## Setup

| | |
|---|---|
| GPUs | 2x AMD Radeon RX 7900 XTX 24 GB (gfx1100), PCIe 4.0 x8 each, no xGMI |
| CPU / RAM | Ryzen 9 9900X, 121 GB DDR5 |
| OS / ROCm | Ubuntu, kernel 7.0, ROCm SDK 10.3.0a20261101 (venv), in-kernel amdgpu |
| Model | DavidAU Qwen3.8-Flash-Next-Abliterated-Turbo-50x-Reasoning IQ3_S (GSQ-RCO), 84 GB with its 26.8 GiB PLE table |
| Engine | layer split 24, batch 4, `--prefill 18432`, `--spec 4`, kv int8, 512k context (yarn 2.0), vision, `--vram-reserve-mib 1024` |
| Sampling | temperature 0, thinking off, 256 output tokens, prompt cache defeated by a unique nonce |

Method (`tools/bench.py`, in the spirit of Strata-rdna3's `rdna3/bench.py`): each prompt is N
tokens of synthetic warehouse records with three 5-character codes at the start, middle and end;
the model must return the codes, then write an English guide (`prose`) or a Python module
(`code`). A request counts only if it returns all three codes with exactly 256 output tokens.
Prompt lengths are approximate (4 chars/token); the tables show the server's actual prompt tokens.

## 1. Single request

One request at a time (the engine still has 4 batch slots; a lone request uses one).
Median of 3 runs (2 for >= 64K).

| target | actual prompt | prefill tok/s | decode tok/s (prose) | decode tok/s (code) |
|---:|---:|---:|---:|---:|
| 1K | 859 | 640 | 127.2 | 146.2 |
| 4K | 3,244 | 1,772 | 128.5 | 138.2 |
| 16K | 12,777 | 2,757 | 121.6 | 141.9 |
| 32K | 25,486 | 3,478 | 122.0 | 135.9 |
| 64K | 50,908 | 4,746 | 123.7 | 144.2 |
| 128K | 101,745 | 5,383 | 120.8 | 139.4 |

Prefill rises with length (fixed overheads amortize); decode is flat at 121-146 tok/s and code
drafts are accepted more often than prose (the MTP head), as the fork reports.

## 2. Four concurrent requests

Four requests launched together, same lengths, each with its own codes and nonce. `aggregate`
is end-to-end (all 4 x 256 output tokens / wall time, prefill included); `per-request decode` is
the median of each request's decode rate. Median of 2 bursts.

| target | actual prompt | wall | aggregate tok/s | per-request decode tok/s | codes returned |
|---:|---:|---:|---:|---:|---|
| 1K | 860 | 10.6 s | 97.3 | 36.7 | 2/2 bursts |
| 4K | 3,245 | 12.3 s | 83.0 | 33.1 | 1/2 (prose 1 miss) |
| 16K | 12,777 | 23.0 s | 44.6 | 22.3 | 2/2 |
| 32K | 25,485 | 34.8 s | 29.5 | 16.5 | 2/2 |
| 64K | 50,908 | 48.8 s | 20.9 | 12.5 | 1/2 (prose 1 miss) |

Concurrency trades per-request latency for aggregate throughput: at 1K, four clients together
decode at ~97 tok/s end-to-end (a single client gets 127-146); at 64K the prefill of four 50K
prompts dominates the wall time, so the end-to-end aggregate falls to ~21 tok/s while each
request still decodes at ~12.5 tok/s.

## Caveats

- **The fork's environment matters more than the model.** The merge had dropped the fork's
  80-key environment. Restoring it (notably `STRATA_HIPBLASLT_TUNING` for gfx1100,
  `STRATA_PREFILL_MMQ_TILE_ROWS=32`, `STRATA_PEER_TRANSPORT=host`, `GPU_MAX_HW_QUEUES=8`) gave
  +45% decode (76-93 -> 121-146) and +10-27% prefill at 4K-64K. The tables above are with it.
- **Model A/B (stock vs DavidAU)**: the stock ISTA-DASLab IQ3_S was downloaded and packed
  (`experts.bin`, 46.84 GiB) and measured on the same engine and env: 32K prefill 4,969,
  64K 6,040 (TP), decode 136-170. That is not faster than DavidAU (32K 3,478 batch / ~4,300 TP,
  decode 122-144), so the fork's 6,760 at 32K is not explained by the model; the remaining
  ~25% is engine/benchmark conditions we have not reproduced.
- **TP option**: with `--tensor-parallel 2` (solo only, `"parallel": 1`) the same model does
  28K prefill at ~4,286 tok/s and 134-150 tok/s decode, but requests queue instead of decoding
  together. Production uses batch 4 for true concurrency.
- **`presence_penalty`**: the model card's default 1.5 forces serial decode and lowers MTP
  acceptance; this benchmark and production set it to 0 (the fork's production does too).
- **`STRATA_REFILL_SDMA=0`**: the gfx1100 TP defaults enable an SDMA refill that does not land
  on ROCm 10.x (`engine -1`, 30 s waits, watchdog) - see the fork's issue #14.
- Code-return misses are the model writing the guide before listing a code within 256 tokens;
  the `code` kind returned all three codes in every burst.

## Reproduce

```bash
python3 tools/bench.py --port 1234 --mode single \
  --lengths 1024,4096,16384,32768,65536,131072 --out bench_single.json
python3 tools/bench.py --port 1234 --mode concurrent --concurrency 4 \
  --lengths 1024,4096,16384,32768,65536 --out bench_concurrent.json
```
