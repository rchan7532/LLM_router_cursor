"""Latency probe: TTFT and tokens/sec for the fleet's coding-capable models.

Measures streaming time-to-first-token (TTFT) and output tokens/sec on one
small synthetic coding prompt, two runs per model, best run reported (network
jitter on the HK->Frankfurt leg dominates the median, best-run is the honest
"how fast can this model be" number).

Usage: python tests/latency_probe.py
Reads .env from the project root. Sends only the synthetic prompt below —
no user content. Costs a fraction of a cent per model.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (model id, which key, what it is)
CANDIDATES = [
    ("glm-5.3-flash", "DEFAULT", "current favorite, the slow case"),
    ("qwen3.8-flash", "BASIC", "cheap flash on basic key"),
    ("deepseek-v4.1-flash", "BASIC", "code-leaning flash"),
    ("qwen3-vl-flash", "BASIC", "vision flash (bulk anchor)"),
    ("kimi/kimi-k2.7-code-highspeed", "DEFAULT", "kimi coding, highspeed serving"),
    ("kimi-k2.7-code", "DEFAULT", "premium coding baseline"),
]

PROMPT = (
    "Write a Python function parse_iso_dates(s: str) that extracts all ISO-8601 "
    "date substrings from a string with a regex, returns them as a list of "
    "datetime.date, skipping invalid ones. Include a 4-line usage example. "
    "Code only, no prose."
)

MAX_TOKENS = 250


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    with open(os.path.join(ROOT, ".env"), encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip()
    return env


def probe_once(model: str, key: str, base: str) -> dict:
    """One streaming completion. Returns {ttft_s, total_s, out_tokens, ok}."""
    body = json.dumps({
        "model": model,
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "messages": [{"role": "user", "content": PROMPT}],
    }).encode()
    request = urllib.request.Request(
        f"{base}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            # Cloudflare on toai.hk 403s the default python UA (error 1010).
            "User-Agent": "litellm/1.101.0",
        },
        method="POST",
    )
    start = time.perf_counter()
    ttft = None
    out_tokens = 0
    with urllib.request.urlopen(request, timeout=90) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except ValueError:
                continue
            if ttft is None:
                ttft = time.perf_counter() - start
            choices = chunk.get("choices") or [{}]
            delta = choices[0].get("delta") or {}
            if delta.get("content"):
                out_tokens += 1  # ~1 token per streamed chunk
    total = time.perf_counter() - start
    return {
        "ttft_s": round(ttft, 3) if ttft else None,
        "total_s": round(total, 3),
        "out_tokens": out_tokens,
        "tok_s": round(out_tokens / (total - ttft), 1) if ttft and total > ttft and out_tokens else None,
        "ok": out_tokens > 5,
    }


def main() -> int:
    env = load_env()
    base = env.get("ROUTER_KEY_DEFAULT_BASE", "")
    print(f"latency probe via {base} — streaming, {MAX_TOKENS} max tokens, best of 2\n")
    print(f"{'model':34} {'ttft':>7} {'tok/s':>8} {'total':>7}  note")
    rows = []
    for model, which, note in CANDIDATES:
        key = env.get(f"ROUTER_KEY_{which}", "")
        best = None
        error = None
        for _ in range(2):
            try:
                result = probe_once(model, key, base)
                if result["ok"] and (best is None or result["ttft_s"] < best["ttft_s"]):
                    best = result
            except Exception as error_:  # noqa: BLE001 - probe must survive
                error = str(error_)[:60]
        if best:
            rows.append((model, note, best))
            print(f"{model:34} {best['ttft_s']:>6.2f}s {best['tok_s'] or 0:>7.1f} {best['total_s']:>6.2f}s  {note}")
        else:
            print(f"{model:34} {'—':>7} {'—':>8} {'—':>7}  UNREACHABLE: {error}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
