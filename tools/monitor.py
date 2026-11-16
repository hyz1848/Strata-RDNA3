#!/usr/bin/env python3
"""Record every request the Strata server reports on /metrics, for the Monitor-style stats.

    python3 tools/monitor.py --port 1234 --out /tmp/monitor.jsonl [--seconds 0]

Polls /metrics?requests=all every 2 s and appends each new request record (by id) as one JSON
line, plus a final summary when it stops (Ctrl-C or --seconds).
"""
import argparse
import http.client
import json
import time

KEY_DEFAULT = "29306136ed566e2ca92839fdbd90e337"


def get(port, key, path):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    c.request("GET", path, headers={"Authorization": "Bearer " + key})
    return json.loads(c.getresponse().read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1234)
    ap.add_argument("--key", default=KEY_DEFAULT)
    ap.add_argument("--out", default="/tmp/monitor.jsonl")
    ap.add_argument("--seconds", type=float, default=0, help="0: until Ctrl-C")
    a = ap.parse_args()
    seen = {}
    out = open(a.out, "a", encoding="utf-8")
    t_end = time.time() + a.seconds if a.seconds > 0 else None
    print("monitoring", a.port, "->", a.out, flush=True)
    try:
        while True:
            try:
                d = get(a.port, a.key, "/metrics?requests=all")
            except Exception as e:
                print("poll error:", e, flush=True)
                time.sleep(2)
                continue
            for r in d.get("requests") or []:
                rid = r.get("id")
                if rid and rid not in seen:
                    seen[rid] = r
                    out.write(json.dumps({"kind": "request", **r}, ensure_ascii=False) + "\n")
                    out.flush()
                    p = r.get("prompt_tokens") or 0
                    g = r.get("output_tokens") or 0
                    pms = r.get("prompt_ms") or 0
                    print(f"req {rid} prompt={p} gen={g} prefill={p/(pms/1000) if pms else 0:.0f} tok/s "
                          f"decode={r.get('decode_tok_s') or 0:.1f} tok/s hit={r.get('hit_rate') or 0:.3f} "
                          f"drafts={r.get('drafts_accepted') or 0}/{r.get('drafts_offered') or 0} "
                          f"finish={r.get('finish')}", flush=True)
            live = d.get("live") or {}
            out.write(json.dumps({"kind": "live", "t": time.time(), "state": live.get("state"),
                                  "running": live.get("running"), "queued": live.get("queued"),
                                  "tok_s": live.get("tok_s"), "prefill_tok_s": live.get("prefill_tok_s_mean")},
                                 ensure_ascii=False) + "\n")
            out.flush()
            if t_end and time.time() >= t_end:
                break
            time.sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        out.close()
    rs = [r for r in seen.values()]
    if rs:
        import statistics as st
        pf = [r["prompt_tokens"] / (r["prompt_ms"] / 1000) for r in rs if r.get("prompt_ms") and r.get("prompt_tokens")]
        de = [r["decode_tok_s"] for r in rs if r.get("decode_tok_s")]
        dr = [(r.get("drafts_accepted") or 0) / (r.get("drafts_offered") or 1) for r in rs if r.get("drafts_offered")]
        print(f"\n{len(rs)} requests | prefill med {st.median(pf):.0f} tok/s | decode med {st.median(de):.1f} tok/s "
              f"| draft accept {100*st.mean(dr):.1f}% | output {sum(r.get('output_tokens') or 0 for r in rs)} tokens")


if __name__ == "__main__":
    main()
