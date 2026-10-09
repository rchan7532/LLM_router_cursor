"""
Pre-call canonicalization of tool-call message arrays.

Registered via `litellm_settings.callbacks` in litellm-config.yaml. The proxy
calls `async_pre_call_hook` per request and uses the returned dict as the
request data, so this hook repairs `messages` before any upstream validator
sees it.

Why this exists (2026-01-09, request c39ae142): long agentic sessions fail
with 400 `tool_call_id does not match a preceding assistant tool call` on
every model. Cause is client-side history compaction: it drops the assistant
message declaring `tool_calls` while keeping the tool results (or the
reverse), and can duplicate tool_call ids across turns. The router cannot
fix the client, so it repairs the payload.

Live probe matrix (deploy/probe-toolcalls.py, 2026-10-09) pinned what each
fleet validator rejects:

    orphan       tool msg whose id no assistant declares   glm/kimi/deepseek 400
    no_id        tool msg with no tool_call_id             glm/kimi/deepseek 400
    unanswered   tool_calls with no result                 qmax/kimi/deepseek 400
    interleave   user msg between assistant and its result kimi/deepseek 400
    late_decl    tool msg BEFORE its declaration           kimi/deepseek 400
    dup_ids      two tool_calls share one id               kimi/deepseek 400
    dup_consumed one call, two results with its id         deepseek 400

The strictest contract (kimi/deepseek, standard OpenAI): every assistant
tool_calls message is IMMEDIATELY followed by exactly one tool result per
call, ids unique, nothing else in between. Repair normalizes every payload
to that contract, so all models accept it.

Repair rules:

    1. tool_call entries and tool results pair 1:1 by id, positionally when
       an id repeats
    2. duplicate or missing ids get fresh unique ones; the paired result's
       tool_call_id is rewritten to match
    3. each assistant tool_calls block is emitted followed IMMEDIATELY by
       its results in call order (moves results past intervening messages)
    4. a tool result with no pair gets a synthetic stub declaration right
       before it (stub name `unknown_tool`, empty arguments)
    5. a call with no result gets a synthetic placeholder result right after
       its declaration (explicit `[tool result unavailable: ...]` marker)

Fabrication discipline (review C5 spirit): the synthetic stub can never look
like a real tool call (always `unknown_tool` / `{}`) and the placeholder can
never look like real output (always the explicit marker), so a repaired
transcript cannot smuggle fake evidence into the model's context. The
repair log records counts only - never message text.

Kill-switch: POLICY_TOOL_REPAIR=0 (or false/no/off) disables the hook. Every
path is fail-open: any internal error forwards the request untouched, so a
proxy with this module routes byte-identically to one without it.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from typing import Any, Mapping

try:
    from litellm.integrations.custom_logger import CustomLogger
except ImportError:  # pragma: no cover - litellm always present in the proxy
    CustomLogger = object  # type: ignore[assignment,misc]

STUB_TOOL_NAME = "unknown_tool"
PLACEHOLDER_RESULT = "[tool result unavailable: truncated before delivery]"
ID_PREFIX = "call_repair_"

REPAIR_LOG_PATH = os.environ.get(
    "POLICY_REPAIR_LOG",
    os.path.join(
        os.environ.get("POLICY_STATE_DIR")
        or ("/app/logs" if os.path.isdir("/app/logs") else tempfile.gettempdir()),
        "tool_repair.jsonl",
    ),
)
REPAIR_MAX_BYTES = int(os.environ.get("POLICY_LOG_MAX_BYTES", "5000000"))

# In-RAM counters so /health style readers see repairs without re-parsing the
# log. Counts only - no message text (C5).
REPAIRS: dict[str, int] = {"declared": 0, "answered": 0, "renamed": 0,
                           "moved": 0, "requests": 0}

_ZERO = {"declared": 0, "answered": 0, "renamed": 0, "moved": 0}


def _enabled() -> bool:
    return os.environ.get("POLICY_TOOL_REPAIR", "1").lower() not in {"0", "false", "no", "off"}


def _fresh_id(used: set[str]) -> str:
    index = 1
    while f"{ID_PREFIX}{index}" in used:
        index += 1
    cid = f"{ID_PREFIX}{index}"
    used.add(cid)
    return cid


def _synthetic_declaration(cid: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": cid,
                "type": "function",
                "function": {"name": STUB_TOOL_NAME, "arguments": "{}"},
            }
        ],
    }


def _synthetic_result(cid: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": cid, "content": PLACEHOLDER_RESULT}


def _result_message(message: Mapping[str, Any], cid: str) -> dict[str, Any]:
    """The tool result with its tool_call_id forced to `cid` (copy-on-write)."""
    if message.get("tool_call_id") == cid:
        return dict(message)
    patched = dict(message)
    patched["tool_call_id"] = cid
    return patched


def repair_messages(messages: Any) -> tuple[Any, dict[str, int]]:
    """Return (messages, counts). counts keys: declared, answered, renamed,
    moved. A no-op returns the SAME list object with zero counts. Never
    raises: anything that is not a list of mappings comes back untouched."""
    if not isinstance(messages, list):
        return messages, dict(_ZERO)
    try:
        used: set[str] = set()
        entries: list[dict[str, Any]] = []   # assistant tool_call slots
        results: list[dict[str, Any]] = []   # tool result rows
        for index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                continue
            role = message.get("role")
            if role == "assistant":
                tool_calls = message.get("tool_calls")
                if not isinstance(tool_calls, list):
                    continue
                for position, entry in enumerate(tool_calls):
                    orig = entry.get("id") if isinstance(entry, Mapping) else None
                    if isinstance(orig, str) and orig:
                        used.add(orig)
                    entries.append({"mi": index, "ti": position, "orig": orig,
                                    "canon": None, "result": None})
            elif role == "tool":
                orig = message.get("tool_call_id")
                if isinstance(orig, str) and orig:
                    used.add(orig)
                results.append({"mi": index, "msg": message, "orig": orig,
                                "canon": None, "entry": None})

        # Pair 1:1 per id, positionally when the id repeats.
        entries_by_id: dict[Any, list[dict[str, Any]]] = {}
        for entry in entries:
            entries_by_id.setdefault(entry["orig"], []).append(entry)
        results_by_id: dict[Any, list[dict[str, Any]]] = {}
        for result in results:
            results_by_id.setdefault(result["orig"], []).append(result)
        for orig, group in entries_by_id.items():
            for entry, result in zip(group, results_by_id.get(orig, []) if orig else []):
                entry["result"] = result
                result["entry"] = entry

        # Canonical ids: a repeated id keeps its first occurrence, every
        # other slot gets a fresh unique one. An orphan result keeps its own
        # id for the stub declaration when nothing else claims it.
        renamed = 0
        claimed: set[str] = set()
        for orig, group in entries_by_id.items():
            for entry in group:
                if isinstance(orig, str) and orig and orig not in claimed:
                    entry["canon"] = orig
                else:
                    entry["canon"] = _fresh_id(used)
                claimed.add(entry["canon"])
                if entry["orig"] != entry["canon"]:
                    renamed += 1
        for result in results:
            if result["entry"] is not None:
                result["canon"] = result["entry"]["canon"]
            elif isinstance(result["orig"], str) and result["orig"] and result["orig"] not in claimed:
                result["canon"] = result["orig"]
                claimed.add(result["canon"])
            else:
                result["canon"] = _fresh_id(used)
                claimed.add(result["canon"])
            if result["orig"] != result["canon"]:
                renamed += 1

        # Rebuild: assistant blocks emit their results immediately after, in
        # call order; paired results leave their original position; unpaired
        # results keep their position but get a stub declaration in front.
        entries_by_msg: dict[int, list[dict[str, Any]]] = {}
        for entry in entries:
            entries_by_msg.setdefault(entry["mi"], []).append(entry)
        result_by_msg: dict[int, dict[str, Any]] = {r["mi"]: r for r in results}

        final: list[Any] = []
        declared = answered = 0
        for index, message in enumerate(messages):
            result = result_by_msg.get(index)
            if result is not None and result["entry"] is not None:
                continue  # re-emitted with its declaring assistant block
            if result is not None:
                declared += 1
                final.append(_synthetic_declaration(result["canon"]))
                final.append(_result_message(result["msg"], result["canon"]))
                continue
            group = entries_by_msg.get(index)
            if group is not None:
                ordered = sorted(group, key=lambda e: e["ti"])
                patched = dict(message)
                patched_calls = []
                for entry in ordered:
                    call = message["tool_calls"][entry["ti"]]
                    if isinstance(call, Mapping) and call.get("id") != entry["canon"]:
                        call = dict(call)
                        call["id"] = entry["canon"]
                    patched_calls.append(call)
                patched["tool_calls"] = patched_calls
                final.append(patched)
                for entry in ordered:
                    if entry["result"] is not None:
                        final.append(_result_message(entry["result"]["msg"], entry["canon"]))
                    else:
                        answered += 1
                        final.append(_synthetic_result(entry["canon"]))
                continue
            final.append(message)

        if declared == 0 and answered == 0 and renamed == 0 and final == messages:
            return messages, dict(_ZERO)
        counts = {"declared": declared, "answered": answered,
                  "renamed": renamed, "moved": int(final != messages)}
        return final, counts
    except Exception:  # noqa: BLE001 - repair must never fail a request
        return messages, dict(_ZERO)


def record_repair(counts: Mapping[str, int], now: float | None = None) -> None:
    """Append one {ts, declared, answered, renamed, moved} line and bump the
    RAM counters. Counts only - never message text (C5). Failures swallowed."""
    now = time.time() if now is None else now
    if not isinstance(counts, Mapping):
        counts = {}
    record = {"ts": now}
    for key in ("declared", "answered", "renamed", "moved"):
        value = int(counts.get(key, 0) or 0)
        record[key] = value
        REPAIRS[key] += value
    REPAIRS["requests"] += 1
    try:
        directory = os.path.dirname(REPAIR_LOG_PATH)
        if directory:
            os.makedirs(directory, exist_ok=True)
        if os.path.exists(REPAIR_LOG_PATH) and os.path.getsize(REPAIR_LOG_PATH) > REPAIR_MAX_BYTES:
            os.replace(REPAIR_LOG_PATH, REPAIR_LOG_PATH + ".1")
        with open(REPAIR_LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    except Exception:  # noqa: BLE001 - logging must never affect a request
        pass


def snapshot() -> dict[str, int]:
    """A copy of the repair counters (counts only)."""
    return dict(REPAIRS)


class ToolRepairHook(CustomLogger if CustomLogger is not object else object):  # type: ignore[misc,valid-type]
    """litellm CustomLogger whose only job is repairing `data["messages"]`
    before the request leaves the proxy. Disabled or broken it forwards the
    request untouched."""

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):  # noqa: ANN001, D102
        try:
            if not _enabled() or not isinstance(data, dict):
                return None
            messages = data.get("messages")
            repaired, counts = repair_messages(messages)
            if repaired is messages:
                return None
            record_repair(counts)
            patched = dict(data)
            patched["messages"] = repaired
            return patched
        except Exception:  # noqa: BLE001 - forward the request untouched
            return None


# The config must name THIS instance (same convention as failure_hook).
repair_hook = ToolRepairHook()
