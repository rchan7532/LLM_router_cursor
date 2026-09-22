"""
Phase 2 extraction client: turns a batch of newest-human asks into
sanitized distillations (key points, notable phrases, true task kind).

Review M2: this call must NEVER ride the proxy. If the learner routed an
extraction through cursor-auto, the request itself would be classified and
logged as a decision, and a batch of user asks would become the newest
human message for classification - the router would observe its own
monitoring. It therefore calls the upstream provider DIRECTLY with the
default key's env credentials (ROUTER_KEY_DEFAULT / ROUTER_KEY_DEFAULT_BASE,
already present in the container), outside litellm entirely.

One extraction call per batch (~20 asks, ~1 flash call per 20 turns).
Timeout 30 s, one retry, then the batch is dropped: losing a batch loses
only learning (plan C1). Payloads are never interpolated into exception
messages (review M4).

Output schema the model must produce:

    {"items": [{"key_points": [<=3 strings, <=400 chars],
                "phrases": [<=3 strings, <=60 chars],
                "kind": one of the BASE_BAR task kinds}]}

Client-side validation drops non-conforming items entirely (review m5);
the distillations are mechanically sanitized again downstream in
learning.append_observation before anything reaches disk.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

try:
    from learning import KINDS
except ImportError:  # pragma: no cover - split for tests
    KINDS = frozenset()  # type: ignore[assignment]

EXTRACT_MODEL = os.environ.get("POLICY_EXTRACT_MODEL", "glm-5.3-flash")
EXTRACT_BASE_URL = os.environ.get("ROUTER_KEY_DEFAULT_BASE", "").rstrip("/")
EXTRACT_API_KEY = os.environ.get("ROUTER_KEY_DEFAULT", "")
EXTRACT_TIMEOUT_S = 30.0
MAX_KEY_POINTS = 3
MAX_PHRASES = 3

_PROMPT_TEMPLATE = """You distill user requests for a routing analytics log. For EACH numbered request below, return one JSON object with:
- "key_points": at most 3 short strings stating what the user wants done. Strip every credential, token, password, API key or secret you see; replace it with [REDACTED]. Never quote file contents.
- "phrases": at most 3 short distinctive phrases from the request (the user's own wording that could identify this kind of task later). Same credential rule.
- "kind": one of {kinds} - the task kind that fits BEST.

Return STRICT JSON: {{"items": [...]}} with exactly one object per request, in order. No markdown, no commentary.

Requests:
{asks}"""


def build_prompt(asks: list[str]) -> str:
    """The extraction prompt. Numbered asks go into the user turn; the
    schema and the credential-stripping instruction live in the template."""
    numbered = "\n".join(f"{index + 1}. {ask}" for index, ask in enumerate(asks))
    return _PROMPT_TEMPLATE.format(kinds=", ".join(sorted(KINDS)), asks=numbered)


def _coerce_items(payload: Any) -> list[dict[str, Any]]:
    """Extract the item list from a model response. Accepts the strict
    shape {"items": [...]} and tolerates a bare JSON array; markdown code
    fences are stripped. Anything else yields no items."""
    if isinstance(payload, str):
        text = payload.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
        try:
            payload = json.loads(text)
        except ValueError:
            return []
    if isinstance(payload, dict):
        items = payload.get("items")
    elif isinstance(payload, list):
        items = payload
    else:
        return []
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def parse_response(body: bytes | str) -> list[dict[str, Any]]:
    """Parse and validate the chat-completions response. Non-conforming
    items are dropped entirely (review m5); kind must be a real task kind."""
    if isinstance(body, bytes):
        body = body.decode("utf-8", "replace")
    try:
        envelope = json.loads(body)
    except ValueError:
        return []
    if not isinstance(envelope, dict):
        return []
    choices = envelope.get("choices")
    if not isinstance(choices, list) or not choices:
        return []
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    out: list[dict[str, Any]] = []
    for item in _coerce_items(content):
        kind = item.get("kind")
        points = item.get("key_points")
        phrases = item.get("phrases")
        if not isinstance(kind, str) or kind not in KINDS:
            continue
        if not isinstance(points, list) or not isinstance(phrases, list):
            continue
        clean_points = [p for p in points if isinstance(p, str)]
        clean_phrases = [p for p in phrases if isinstance(p, str)]
        if len(clean_points) > MAX_KEY_POINTS or len(clean_phrases) > MAX_PHRASES:
            # Dropped, not truncated: a count violation is a signal the item
            # cannot be trusted (review m5's "drop entirely").
            continue
        out.append({"kind": kind, "key_points": clean_points, "phrases": clean_phrases})
    return out


def _post_chat(payload: dict[str, Any], timeout: float) -> bytes:
    """One synchronous POST to the upstream provider's chat-completions.
    Error bodies are summarized, never echoed (review M4)."""
    request = urllib.request.Request(
        f"{EXTRACT_BASE_URL}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {EXTRACT_API_KEY}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"extraction HTTP {error.code}") from None
    except Exception as error:  # URLError, timeouts, socket errors
        raise RuntimeError(f"extraction transport failure") from None


async def extract(asks: list[str]) -> list[dict[str, Any]]:
    """Extract distillations for a batch of asks, DIRECTLY from the
    upstream provider - never through this proxy (review M2).

    Timeout 30 s per attempt, one retry, then the batch is dropped with an
    empty result (C1: losing a batch loses only learning). The asyncio hop
    exists so the supervised learner task can await this from the proxy's
    event loop without blocking it."""
    if not asks or not EXTRACT_BASE_URL or not EXTRACT_API_KEY:
        return []
    payload = {
        "model": EXTRACT_MODEL,
        "messages": [{"role": "user", "content": build_prompt(asks)}],
        "temperature": 0,
        "max_tokens": min(4000, 200 * len(asks) + 200),
    }
    for attempt in range(2):  # one attempt + one retry
        try:
            body = await _post_chat_async(payload, EXTRACT_TIMEOUT_S)
            return parse_response(body)
        except Exception:
            if attempt == 1:
                return []  # batch dropped; counted by the caller
            time.sleep(0.5)
    return []


async def _post_chat_async(payload: dict[str, Any], timeout: float) -> bytes:
    """The blocking HTTP call in a worker thread."""
    import asyncio
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _post_chat, payload, timeout)
