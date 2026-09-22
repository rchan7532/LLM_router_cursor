"""One-shot connectivity probe for candidate upstream models.

Sends a 1-token completion per model with a short timeout and reports the HTTP
outcome. Distinguishes: 200/4xx-token-billing = routable, 503 = listed but
currently unavailable, timeout/0 = provider silent or network path broken.

Usage: python tests/probe_models.py
Reads .env from the project root. Read-only: sends no prompt content beyond the
literal "hi" with max_tokens=1.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CANDIDATES = [
    # (model id as the provider serves it, which key should serve it)
    ("kimi-k2.7-code", "DEFAULT"),
    ("kimi/kimi-k2.7-code", "DEFAULT"),
    ("kimi/kimi-k2.7-code-highspeed", "DEFAULT"),
    ("kimi-k2.7-code-highspeed", "DEFAULT"),
    ("kimi-k2.6", "DEFAULT"),
    ("deepseek-v4.1-flash", "BASIC"),
    ("deepseek-v4-pro", "DEFAULT"),
    ("claude-haiku-4-5", "DEFAULT"),
    ("claude-sonnet-4-5", "DEFAULT"),
    ("glm-5.2-fast", "BASIC"),
    ("glm-5.2", "DEFAULT"),
    ("glm-4.6v", "DEFAULT"),
    ("qwen3.8-flash", "BASIC"),
    ("qwen3.8-2.4t-a95b", "BASIC"),
    ("qwen3-vl-flash", "BASIC"),
    # excluded-by-policy checks, to confirm the exclusion is price not availability
    ("kimi-k3", "DEFAULT"),
    ("claude-fable-5", "DEFAULT"),
    ("claude-opus-5", "DEFAULT"),
]


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    with open(os.path.join(ROOT, ".env"), encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip()
    return env


def probe(model: str, key: str, base: str) -> str:
    body = json.dumps({
        "model": model,
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "hi"}],
    }).encode()
    request = urllib.request.Request(
        f"{base}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            # toai.hk sits behind Cloudflare, which 403s (error 1010) the
            # default python-urllib UA. httpx/curl/openai-style UAs pass.
            "User-Agent": "litellm/1.101.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return f"HTTP {response.status}"
    except urllib.error.HTTPError as error:
        return f"HTTP {error.code}"
    except Exception as error:  # noqa: BLE001 - probe must survive any failure
        reason = str(error.reason) if hasattr(error, "reason") else error
        return f"UNREACHABLE ({reason})"


def main() -> int:
    env = load_env()
    base = env.get("ROUTER_KEY_DEFAULT_BASE", "")
    if not base:
        print("ROUTER_KEY_DEFAULT_BASE missing")
        return 1
    print(f"probing {len(CANDIDATES)} models via {base} (1-token completions)\n")
    for model, which in CANDIDATES:
        key = env.get(f"ROUTER_KEY_{which}", "")
        result = probe(model, key, base)
        print(f"{model:34} [{which:7}] {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
