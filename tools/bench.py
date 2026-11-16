#!/usr/bin/env python3
"""Benchmark a running Strata server, in the spirit of Strata-rdna3's rdna3/bench.py.

Two modes:
  single      one request at a time: prefill (TTFT) and decode tokens/s per prompt length
  concurrent  N requests at once: wall time, aggregate tokens/s, per-request decode

Each prompt is N tokens of synthetic warehouse records with three hidden codes at the start,
middle and end; the model must return the codes, then write an English guide or a Python module.
Prompts are unique per request (a nonce), so the conversation/prompt cache does not help.

    python3 tools/bench.py --port 1234 --mode single --lengths 1024,4096,16384,32768,65536,131072
    python3 tools/bench.py --port 1234 --mode concurrent --concurrency 4 --lengths 1024,4096,16384,32768,65536

Prompt lengths are approximate (4 chars/token for the English filler); the table reports the
actual prompt tokens from the server's usage.
"""
import argparse
import http.client
import json
import random
import statistics
import string
import threading
import time

KEY_DEFAULT = "29306136ed566e2ca92839fdbd90e337"
FILLER = ("The warehouse log records item codes, quantities, the receiving dock, the auditor's notes and the "
          "shelf positions; each line was checked twice before it was signed. ")
# measured on this model's tokenizer: the filler is ~4.77 chars per token (a 1,024-token prompt was 4,096 chars)
CHARS_PER_TOKEN = 4.766


def code(rng):
    return "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(5))


def make_prompt(target, kind, rng):
    codes = [code(rng) for _ in range(3)]
    nonce = "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(8))
    chars = int(target * CHARS_PER_TOKEN)
    body = (FILLER * (chars // len(FILLER) + 2))[:chars]
    a, b = len(body) // 10, len(body) // 2
    body = body[:a] + f" [{codes[0]}] " + body[a:b] + f" [{codes[1]}] " + body[b:] + f" [{codes[2]}] "
    task = ("write a short inventory-auditing guide." if kind == "prose"
            else "write a short Python inventory-auditing module.")
    q = (f"Session {nonce}. " + body + "\n\nReturn the three hidden codes you saw (they look like ABC12), then " + task)
    return q, codes


def ask(port, key, prompt, maxtok, stream):
    body = json.dumps({"model": "qwen3.8-next-flash", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": maxtok, "temperature": 0, "stream": stream,
                       "stream_options": {"include_usage": True} if stream else None,
                       "chat_template_kwargs": {"enable_thinking": False}})
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=3600)
    t0 = time.time()
    c.request("POST", "/v1/chat/completions", body,
              {"Content-Type": "application/json", "Authorization": "Bearer " + key})
    r = c.getresponse()
    tf = None
    usage = None
    text = ""
    if stream:
        for line in r:
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            d = line[6:]
            if d == "[DONE]":
                break
            try:
                o = json.loads(d)
            except Exception:
                continue
            if o.get("usage"):
                usage = o["usage"]
            delta = o.get("choices", [{}])[0].get("delta", {}).get("content")
            if delta and tf is None:
                tf = time.time()
            if delta:
                text += delta
    else:
        d = json.loads(r.read())
        usage = d.get("usage")
        text = d["choices"][0]["message"].get("content") or ""
        tf = time.time()
    t1 = time.time()
    return {"usage": usage, "ttft": (tf - t0) if tf else 0.0, "total": t1 - t0, "text": text}


def one_case(port, key, target, kind, maxtok, rng):
    q, codes = make_prompt(target, kind, rng)
    r = ask(port, key, q, maxtok, True)
    u = r["usage"] or {}
    p, g = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
    prefill = p / r["ttft"] if r["ttft"] > 0 else 0.0
    decode = (g - 1) / (r["total"] - r["ttft"]) if r["ttft"] > 0 and g > 1 else 0.0
    found = all(c in r["text"] for c in codes)
    return {"prompt": p, "gen": g, "prefill": prefill, "decode": decode, "codes_ok": found}


def run_single(port, key, lengths, kinds, repeats, maxtok, out):
    rng = random.Random(1234)
    rows = []
    for target in lengths:
        for kind in kinds:
            reps = repeats if target < 65536 else max(1, repeats - 1)
            vals = [one_case(port, key, target, kind, maxtok, rng) for _ in range(reps)]
            ok = all(v["codes_ok"] and v["gen"] == maxtok for v in vals)
            row = {"target": target, "kind": kind, "prompt": statistics.median(v["prompt"] for v in vals),
                   "prefill": statistics.median(v["prefill"] for v in vals),
                   "decode": statistics.median(v["decode"] for v in vals), "ok": ok, "reps": reps}
            rows.append(row)
            print(f"single {target:>7} {kind:<5} prompt={row['prompt']:>7.0f} prefill={row['prefill']:>7.0f} tok/s "
                  f"decode={row['decode']:>6.1f} tok/s {'ok' if ok else 'CHECK'}", flush=True)
    if out:
        json.dump({"mode": "single", "rows": rows}, open(out, "w"), indent=1)
    return rows


def run_concurrent(port, key, lengths, kinds, concurrency, rounds, maxtok, out):
    rng = random.Random(5678)
    rows = []
    for target in lengths:
        for kind in kinds:
            walls, aggs, per, prompts, oks, pre_agg = [], [], [], [], [], []
            for _ in range(rounds):
                qs = [make_prompt(target, kind, rng) for _ in range(concurrency)]
                res = [None] * concurrency
                def worker(i):
                    q, codes = qs[i]
                    r = ask(port, key, q, maxtok, True)
                    res[i] = (r, codes)
                th = [threading.Thread(target=worker, args=(i,)) for i in range(concurrency)]
                t0 = time.time()
                [x.start() for x in th]
                [x.join() for x in th]
                wall = time.time() - t0
                ttfts = [r["ttft"] for r, _ in res]
                ends = [r["ttft"] + r["total"] - r["ttft"] if False else (r["ttft"] + (r["total"] - r["ttft"])) for r, _ in res]
                # end time of each request measured from the burst's t0: ttft + (total - ttft) == total
                ends = [r["total"] for r, _ in res]
                max_ttft, max_end = max(ttfts), max(ends)
                sum_prompt = sum((r["usage"] or {}).get("prompt_tokens", 0) for r, _ in res)
                sum_gen = sum((r["usage"] or {}).get("completion_tokens", 0) for r, _ in res)
                walls.append(wall)
                pre_agg.append(sum_prompt / max(max_ttft - 0.0, 1e-6))
                aggs.append(sum_gen / max(wall, 1e-6))   # end-to-end: all output tokens over the burst's wall time
                prompts.append(statistics.median((r["usage"] or {}).get("prompt_tokens", 0) for r, _ in res))
                per.append(statistics.median(
                    ((r["usage"] or {}).get("completion_tokens", 1) - 1) / max(r["total"] - r["ttft"], 1e-6)
                    for r, _ in res))
                oks.append(all(all(c in r["text"] for c in codes) and (r["usage"] or {}).get("completion_tokens") == maxtok
                               for r, codes in res))
            row = {"target": target, "kind": kind, "prompt": statistics.median(prompts),
                   "wall": statistics.median(walls), "prefill_aggregate": statistics.median(pre_agg),
                   "aggregate": statistics.median(aggs),
                   "per_request": statistics.median(per), "ok": all(oks), "rounds": rounds}
            rows.append(row)
            print(f"{concurrency}x {target:>7} {kind:<5} prompt={row['prompt']:>7.0f} wall={row['wall']:>6.1f}s "
                  f"aggregate={row['aggregate']:>6.1f} tok/s (end-to-end) per-request-decode={row['per_request']:>5.1f} tok/s "
                  f"{'ok' if row['ok'] else 'CHECK'}", flush=True)
    if out:
        json.dump({"mode": "concurrent", "concurrency": concurrency, "rows": rows}, open(out, "w"), indent=1)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1234)
    ap.add_argument("--key", default=KEY_DEFAULT)
    ap.add_argument("--mode", choices=["single", "concurrent"], default="single")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--lengths", default="1024,4096,16384,32768,65536")
    ap.add_argument("--kinds", default="prose,code")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    lengths = [int(x) for x in a.lengths.split(",")]
    kinds = [k.strip() for k in a.kinds.split(",")]
    if a.mode == "single":
        run_single(a.port, a.key, lengths, kinds, a.repeats, a.max_tokens, a.out)
    else:
        run_concurrent(a.port, a.key, lengths, kinds, a.concurrency, a.rounds, a.max_tokens, a.out)


if __name__ == "__main__":
    main()
