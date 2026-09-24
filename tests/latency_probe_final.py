"""Latency probe FINAL: separates reasoning time from answer time.

Round-3 diagnosis: glm-5.3-flash / deepseek-v4.1-flash / glm-5.2 / kimi all
emit `reasoning_content` chunks before `content`. Earlier rounds counted only
`content`, so a thinking model looked like a dead stream. This probe reports:

  ttft      time to first chunk of ANY kind (proxy + queue + model startup)
  think_s   time spent streaming reasoning_content (invisible work)
  tok/s     answer tokens/sec once content starts flowing
  total     end-to-end wall clock

This is the number that matches what you feel in Cursor: an agent turn waits
for the FULL completion, so total = ttft + think + generate.
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
    ("glm-5.3-flash", "DEFAULT", "current favorite"),
    ("qwen3-vl-flash", "BASIC", "no thinking, fast"),
    ("qwen3.8-flash", "BASIC", "light thinker"),
    ("deepseek-v4.1-flash", "BASIC", "thinker"),
    ("glm-5.2-fast", "BASIC", "new: fast glm-5.2"),
    ("glm-5.2", "DEFAULT", "new: full glm-5.2"),
    ("kimi-k2.7-code", "DEFAULT", "premium coder"),
]

PROMPT = (
    "Write a Python function parse_iso_dates(s: str) that extracts all ISO-8601 "
    "date substrings from a string with a regex, returns them as a list of "
    "datetime.date, skipping invalid ones. Include a 4-line usage example. "
    "Code only, no prose."
)
MAX_TOKENS = 300


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
    conn = http.client.HTTPSConnection(url.hostname, url.port or 443, timeout=240)
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
    ttft = think_s = content_start = None
    think_chars = answer_chars = 0
    try:
        conn.request("POST", f"{url.path}/chat/completions", body=body, headers=headers)
        response = conn.getresponse()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}")
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
            now = time.perf_counter()
            if ttft is None:
                ttft = now - start
            choices = chunk.get("choices") or [{}]
            delta = choices[0].get("delta") or {}
            reasoning = delta.get("reasoning_content") or ""
            content = delta.get("content") or ""
            if reasoning:
                think_chars += len(reasoning)
            if content:
                if content_start is None:
                    content_start = now
                answer_chars += len(content)
        total = time.perf_counter() - start
    finally:
        conn.close()
    if ttft is None or (think_chars + answer_chars) < 40:
        raise RuntimeError(f"no usable stream ({think_chars}+{answer_chars} chars)")
    if content_start is None:
        # burned all tokens on reasoning; treat end as answer start
        content_start = start + ttft + (think_chars / max(think_chars, 1)) * (total - ttft)
    think_s = (content_start - start) - ttft
    gen_s = max(total - content_start, 0.001)
    return {
        "ttft": ttft,
        "think": max(think_s, 0.0),
        "tok_s": (answer_chars / 3.5) / gen_s if answer_chars else 0.0,
        "total": total,
    }


def main() -> int:
    env = load_env()
    base = env.get("ROUTER_KEY_DEFAULT_BASE", "")
    print(f"final probe via {base} — 300 max tokens, best of 2\n")
    print(f"{'model':22} {'ttft':>6} {'think':>7} {'tok/s':>7} {'total':>7}")
    for model, which, note in CANDIDATES:
        key = env.get(f"ROUTER_KEY_{which}", "")
        best = None
        for _ in range(2):
            try:
                result = probe_once(model, key, base)
                if best is None or result["total"] < best["total"]:
                    best = result
            except Exception:  # noqa: BLE001 - probe must survive every model
                pass
        if best:
            print(f"{model:22} {best['ttft']:>5.2f}s {best['think']:>6.1f}s "
                  f"{best['tok_s']:>6.1f} {best['total']:>6.1f}s  {note}", flush=True)
        else:
            print(f"{model:22} {'FAIL':>6} {'—':>7} {'—':>7} {'—':>7}  {note}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
