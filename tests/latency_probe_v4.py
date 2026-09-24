"""Latency probe v4: correct tok/s via usage tokens, cap runaway thinkers.

tok/s in the previous round divided chunk-characters by a ~zero-length
generation window (chunky streaming = bogus huge numbers). This version
reads usage.completion_tokens from the final chunk and caps run time at
45 s (a >45 s single completion is disqualified for interactive use
anyway — that is a finding, not a measurement problem).
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
MAX_TOKENS = 300
RUN_CAP_S = 45.0


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    with open(os.path.join(ROOT, ".env"), encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip()
    return env


class Runaway(Exception):
    pass


def probe_once(model: str, key: str, base: str) -> dict:
    url = urlparse(base)
    conn = http.client.HTTPSConnection(url.hostname, url.port or 443, timeout=200)
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
    ttft = content_start = None
    answer_chars = 0
    completion_tokens = None
    try:
        conn.request("POST", f"{url.path}/chat/completions", body=body, headers=headers)
        response = conn.getresponse()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}")
        while True:
            if time.perf_counter() - start > RUN_CAP_S:
                raise Runaway()
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
            now = time.perf_counter()
            if ttft is None:
                ttft = now - start
            choices = chunk.get("choices") or [{}]
            delta = choices[0].get("delta") or {}
            if delta.get("content"):
                if content_start is None:
                    content_start = now
                answer_chars += len(delta["content"])
            usage = chunk.get("usage") or {}
            if usage.get("completion_tokens"):
                completion_tokens = int(usage["completion_tokens"])
        total = time.perf_counter() - start
    finally:
        conn.close()
    if ttft is None or answer_chars < 40:
        raise RuntimeError(f"no answer content ({answer_chars} chars)")
    if completion_tokens is None:
        completion_tokens = max(1, answer_chars // 4)
    gen_s = max(total - (content_start or start), 0.001)
    return {
        "ttft": ttft,
        "tok_s": completion_tokens / gen_s,
        "total": total,
        "out_tokens": completion_tokens,
    }


def main() -> int:
    env = load_env()
    base = env.get("ROUTER_KEY_DEFAULT_BASE", "")
    print(f"probe v4 via {base} — 300 max tokens, best-of-2 by total, {RUN_CAP_S:.0f}s cap\n")
    print(f"{'model':22} {'ttft':>6} {'tok/s':>6} {'total':>7} {'out tok':>8}")
    for model, which in CANDIDATES:
        key = env.get(f"ROUTER_KEY_{which}", "")
        best = None
        note = ""
        for _ in range(2):
            try:
                result = probe_once(model, key, base)
                if best is None or result["total"] < best["total"]:
                    best = result
            except Runaway:
                note = f"> {RUN_CAP_S:.0f}s (runaway)"
            except Exception as error:  # noqa: BLE001
                note = str(error)[:40]
        if best:
            print(f"{model:22} {best['ttft']:>5.2f}s {best['tok_s']:>5.1f} "
                  f"{best['total']:>6.1f}s {best['out_tokens']:>8}", flush=True)
        else:
            print(f"{model:22} {'FAIL':>6} {'—':>6} {'—':>7} {'—':>8}  {note}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
