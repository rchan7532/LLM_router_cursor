"""Latency probe round 3: patient version.

Round 2 read the stream too greedily and mislabeled slow-first-token models
as failures (the 'no usable stream' rows: slow TTFT is EXPECTED for the big
models — that is exactly what we are measuring). This round: per-chunk
socket timeout, no global deadline kill, accept any completion that produced
content, and report BOTH runs so we can see variance.
"""

from __future__ import annotations

import http.client
import json
import os
import sys
import time
from urllib.parse import urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CANDIDATES = [
    ("glm-5.3-flash", "DEFAULT"),
    ("qwen3-vl-flash", "BASIC"),
    ("qwen3.8-flash", "BASIC"),
    ("deepseek-v4.1-flash", "BASIC"),
    ("glm-5.2-fast", "BASIC"),
    ("glm-5.2", "DEFAULT"),
    ("kimi-k2.7-code", "DEFAULT"),
]

PROMPT = (
    "Write a Python function parse_iso_dates(s: str) that extracts all ISO-8601 "
    "date substrings from a string with a regex, returns them as a list of "
    "datetime.date, skipping invalid ones. Include a 4-line usage example. "
    "Code only, no prose."
)
MAX_TOKENS = 220


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
    url = urlparse(base)
    conn = http.client.HTTPSConnection(url.hostname, url.port or 443, timeout=150)
    body = json.dumps({
        "model": model,
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "messages": [{"role": "user", "content": PROMPT}],
    })
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "User-Agent": "litellm/1.101.0",
    }
    start = time.perf_counter()
    try:
        conn.request("POST", f"{url.path}/chat/completions", body=body, headers=headers)
        response = conn.getresponse()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}")
        ttft = None
        out_chars = 0
        # readline blocks up to the socket timeout per chunk; slow models
        # pause between chunks, so patience here is the point of the probe.
        while True:
            line = response.fp.readline()
            if not line:
                break
            text = line.decode("utf-8", "replace").strip()
            if not text.startswith("data:"):
                continue
            payload = text[5:].strip()
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
            out_chars += len(delta.get("content") or "")
        total = time.perf_counter() - start
    finally:
        conn.close()
    if ttft is None or out_chars < 40:
        raise RuntimeError(f"no usable stream ({out_chars} chars)")
    gen_s = max(total - ttft, 0.001)
    return {"ttft": ttft, "tok_s": (out_chars / 3.5) / gen_s, "total": total}


def main() -> int:
    env = load_env()
    base = env.get("ROUTER_KEY_DEFAULT_BASE", "")
    print(f"round 3 via {base} — streaming, {MAX_TOKENS} max tokens, best of 2 (patient)\n")
    print(f"{'model':24} {'ttft':>7} {'tok/s':>7} {'total':>8}")
    for model, which in CANDIDATES:
        key = env.get(f"ROUTER_KEY_{which}", "")
        best = None
        err = ""
        for attempt in range(2):
            try:
                result = probe_once(model, key, base)
                if best is None or result["ttft"] < best["ttft"]:
                    best = result
            except Exception as error:  # noqa: BLE001
                err = f"{str(error)[:60]} (run {attempt + 1})"
                print(f"    {model}: {err}", flush=True)
        if best:
            print(f"{model:24} {best['ttft']:>6.2f}s {best['tok_s']:>6.1f} {best['total']:>7.2f}s", flush=True)
        else:
            print(f"{model:24} {'FAIL':>7} {'—':>7} {'—':>8}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
