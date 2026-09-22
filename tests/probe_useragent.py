"""One-shot: which User-Agents does toai.hk's Cloudflare allow?

litellm's httpx client sends its own UA; if Cloudflare blocks it, every router
request 403s. Identifies the boundary so the config can pin a working UA.

Usage: python tests/probe_useragent.py
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    with open(os.path.join(ROOT, ".env"), encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip()
    return env


AGENTS = [
    ("python-urllib (python default)", "Python-urllib/3.14"),
    ("httpx (litellm's client)", "python-httpx/0.27.0"),
    ("openai-python (litellm inner)", "AsyncOpenAI/Python 1.99.1"),
    ("litellm version string", "litellm/1.101.0"),
    ("curl", "curl/8.0.1"),
    ("no UA at all", None),
    ("browser", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"),
]


def main() -> int:
    env = load_env()
    base = env["ROUTER_KEY_DEFAULT_BASE"]
    key = env["ROUTER_KEY_DEFAULT"]
    body = json.dumps({
        "model": "kimi-k2.7-code",
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "hi"}],
    }).encode()
    for label, ua in AGENTS:
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        if ua:
            headers["User-Agent"] = ua
        request = urllib.request.Request(base + "/chat/completions", data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                print(f"{label:34} -> HTTP {response.status}")
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:24]
            print(f"{label:34} -> HTTP {error.code} {detail}")
        except Exception as error:  # noqa: BLE001
            print(f"{label:34} -> {type(error).__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
