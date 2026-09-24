"""Diagnose the 'no usable stream' models: is it a non-stream response,
an error body, or empty content deltas? Prints the first raw SSE lines."""

from __future__ import annotations

import http.client
import json
import os
import sys
from urllib.parse import urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHECKS = [
    ("glm-5.3-flash", "DEFAULT", True),
    ("glm-5.3-flash", "DEFAULT", False),   # non-streaming control
    ("deepseek-v4.1-flash", "BASIC", True),
    ("glm-5.2-fast", "BASIC", True),
    ("kimi-k2.7-code", "DEFAULT", True),
]

PROMPT = "Reply with the word ok."
MAX_TOKENS = 10


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    with open(os.path.join(ROOT, ".env"), encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip()
    return env


def main() -> int:
    env = load_env()
    base = urlparse(env.get("ROUTER_KEY_DEFAULT_BASE", ""))
    for model, which, stream in CHECKS:
        key = env.get(f"ROUTER_KEY_{which}", "")
        conn = http.client.HTTPSConnection(base.hostname, base.port or 443, timeout=120)
        body = json.dumps({
            "model": model,
            "max_tokens": MAX_TOKENS,
            "stream": stream,
            "messages": [{"role": "user", "content": PROMPT}],
        })
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
            "User-Agent": "litellm/1.101.0",
        }
        print(f"\n=== {model} [{which}] stream={stream}")
        try:
            conn.request("POST", f"{base.path}/chat/completions", body=body, headers=headers)
            response = conn.getresponse()
            print(f"HTTP {response.status} {response.getheader('Content-Type')}")
            raw = response.read(2000).decode("utf-8", "replace")
            print(raw[:1200])
        except Exception as error:  # noqa: BLE001
            print(f"EXC {error}")
        finally:
            conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
