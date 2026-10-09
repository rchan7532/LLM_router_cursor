"""Probe which malformed tool-call shapes the upstream validator rejects.

POSTs tiny crafted chat payloads through the local proxy (direct model name,
bypassing the policy) and prints status + first line of body per variant.

Variants:
  clean      assistant tool_calls + matching tool message     -> expect 200
  dup_ids    two parallel tool_calls sharing id "call_0"      -> suspect 400
  orphan     tool message with id no assistant ever declared   -> suspect 400
  interleave assistant tool_calls, USER, then tool message     -> suspect 400
  unanswered assistant tool_calls with NO tool message         -> suspect 400
"""

import json
import os
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:4000/v1/chat/completions"
KEY = os.environ.get("LITELLM_MASTER_KEY", "")
MODELS = [m for m in os.environ.get("PROBE_MODELS", "glm-5.3-flash").split(",") if m]


def call(name, messages, model):
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": 8,
    }).encode()
    req = urllib.request.Request(
        BASE, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {KEY}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            print(f"{model:22} {name:12} {resp.status}")
    except urllib.error.HTTPError as exc:
        text = exc.read().decode(errors="replace")[:150].replace("\n", " ")
        print(f"{model:22} {name:12} {exc.code}  {text}")
    except Exception as exc:  # noqa: BLE001 - probe must never die mid-matrix
        print(f"{model:22} {name:12} ERR  {type(exc).__name__}: {exc}")


def tool_call(cid, fn):
    return {"id": cid, "type": "function",
            "function": {"name": fn, "arguments": "{}"}}


def tool_msg(cid, content="ok"):
    return {"role": "tool", "tool_call_id": cid, "content": content}


USER = {"role": "user", "content": "hi"}

VARIANTS = {
    "clean": [   # control: well-formed tool exchange
        USER,
        {"role": "assistant", "content": None,
         "tool_calls": [tool_call("call_0", "f1")]},
        tool_msg("call_0"),
        {"role": "user", "content": "go"},
    ],
    "dup_ids": [  # two parallel tool_calls sharing id "call_0"
        USER,
        {"role": "assistant", "content": None,
         "tool_calls": [tool_call("call_0", "f1"), tool_call("call_0", "f2")]},
        tool_msg("call_0", "r1"),
        tool_msg("call_0", "r2"),
        {"role": "user", "content": "go"},
    ],
    "orphan": [   # tool message with id no assistant ever declared
        USER,
        tool_msg("call_99", "lost result"),
        {"role": "user", "content": "go"},
    ],
    "interleave": [  # assistant tool_calls, USER, then tool message
        USER,
        {"role": "assistant", "content": None,
         "tool_calls": [tool_call("call_0", "f1")]},
        {"role": "user", "content": "quick question"},
        tool_msg("call_0"),
        {"role": "user", "content": "go"},
    ],
    "unanswered": [  # assistant tool_calls with NO tool message
        USER,
        {"role": "assistant", "content": None,
         "tool_calls": [tool_call("call_0", "f1")]},
        {"role": "user", "content": "go"},
    ],
    "dup_consumed": [  # 1 declaration, 2 tool msgs same id
        USER,
        {"role": "assistant", "content": None,
         "tool_calls": [tool_call("call_0", "f1")]},
        tool_msg("call_0", "r1"),
        tool_msg("call_0", "r2"),
        {"role": "user", "content": "go"},
    ],
    "late_decl": [  # tool msg BEFORE its declaration
        USER,
        tool_msg("call_0"),
        {"role": "assistant", "content": None,
         "tool_calls": [tool_call("call_0", "f1")]},
        {"role": "user", "content": "go"},
    ],
    "no_id": [  # tool msg missing tool_call_id entirely
        USER,
        {"role": "tool", "content": "lost result"},
        {"role": "user", "content": "go"},
    ],
}

for model in MODELS:
    for name, messages in VARIANTS.items():
        call(name, messages, model)
