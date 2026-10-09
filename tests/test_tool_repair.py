"""Tests for tool_repair.py: pre-call canonicalization of tool-call arrays.

Run from the llm-router directory:

    python -m pytest tests/test_tool_repair.py -v
    # or, without pytest installed:
    python tests/test_tool_repair.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Test isolation: keep the run out of any production repair log.
_REPAIR_TEST_STATE_DIR = tempfile.mkdtemp(prefix="repair-tests-")
os.environ["POLICY_STATE_DIR"] = _REPAIR_TEST_STATE_DIR
os.environ["POLICY_REPAIR_LOG"] = os.path.join(_REPAIR_TEST_STATE_DIR, "tool_repair.jsonl")

import tool_repair  # noqa: E402


def tool_call(cid, fn="f1"):
    entry = {"type": "function", "function": {"name": fn, "arguments": "{}"}}
    if cid is not None:
        entry["id"] = cid
    return entry


def assistant(calls):
    return {"role": "assistant", "content": None, "tool_calls": calls}


def tool_msg(cid, content="ok"):
    message = {"role": "tool", "content": content}
    if cid is not None:
        message["tool_call_id"] = cid
    return message


USER = {"role": "user", "content": "hi"}
END = {"role": "user", "content": "go"}


def repair(messages):
    repaired, counts = tool_repair.repair_messages(messages)
    return repaired, counts


def roles(messages):
    return [m.get("role") for m in messages]


# ---------------------------------------------------------------------------
# no-op paths: clean traffic must come back as the SAME object
# ---------------------------------------------------------------------------

def test_clean_untouched():
    messages = [USER, assistant([tool_call("call_0")]), tool_msg("call_0"), END]
    repaired, counts = repair(messages)
    assert repaired is messages
    assert counts == {"declared": 0, "answered": 0, "renamed": 0, "moved": 0}


def test_clean_parallel_calls_untouched():
    messages = [
        USER,
        assistant([tool_call("call_0"), tool_call("call_1")]),
        tool_msg("call_0", "r0"),
        tool_msg("call_1", "r1"),
        END,
    ]
    repaired, counts = repair(messages)
    assert repaired is messages
    assert sum(counts.values()) == 0


def test_non_list_passthrough():
    for junk in (None, "text", 42, {"role": "user"}):
        repaired, counts = repair(junk)
        assert repaired is junk
        assert sum(counts.values()) == 0


def test_non_mapping_entries_kept():
    messages = ["junk", USER, 123]
    repaired, counts = repair(messages)
    assert repaired is messages


# ---------------------------------------------------------------------------
# rule 4: orphan tool result (no declaration anywhere) -> stub declaration
# ---------------------------------------------------------------------------

def test_orphan_gets_stub_declaration():
    messages = [USER, tool_msg("call_99", "lost result"), END]
    repaired, counts = repair(messages)
    assert counts["declared"] == 1
    stub = repaired[1]
    assert stub["role"] == "assistant"
    call = stub["tool_calls"][0]
    assert call["id"] == "call_99"
    assert call["function"]["name"] == tool_repair.STUB_TOOL_NAME
    assert call["function"]["arguments"] == "{}"
    # the real result keeps its content and follows the stub immediately
    assert repaired[2] == tool_msg("call_99", "lost result")
    assert roles(repaired) == ["user", "assistant", "tool", "user"]


def test_no_id_result_gets_generated_id_and_stub():
    messages = [USER, tool_msg(None, "lost result"), END]
    repaired, counts = repair(messages)
    assert counts["declared"] == 1
    call = repaired[1]["tool_calls"][0]
    assert call["id"].startswith(tool_repair.ID_PREFIX)
    assert repaired[2]["tool_call_id"] == call["id"]
    assert repaired[2]["content"] == "lost result"


def test_orphan_content_preserved_verbatim():
    payload = "CANARY-tool-output-payload"
    messages = [USER, tool_msg("call_99", payload), END]
    repaired, _ = repair(messages)
    assert payload in json.dumps(repaired)


# ---------------------------------------------------------------------------
# rule 5: unanswered tool_calls -> placeholder result after declaration
# ---------------------------------------------------------------------------

def test_unanswered_call_gets_placeholder():
    messages = [USER, assistant([tool_call("call_0")]), END]
    repaired, counts = repair(messages)
    assert counts["answered"] == 1
    assert repaired[1] == assistant([tool_call("call_0")])
    placeholder = repaired[2]
    assert placeholder == tool_repair._synthetic_result("call_0")
    assert placeholder["content"] == tool_repair.PLACEHOLDER_RESULT
    assert roles(repaired) == ["user", "assistant", "tool", "user"]


def test_partially_answered_call():
    messages = [
        USER,
        assistant([tool_call("call_0"), tool_call("call_1")]),
        tool_msg("call_0", "r0"),
        END,
    ]
    repaired, counts = repair(messages)
    assert counts["answered"] == 1
    # call_0's real result stays adjacent; call_1 gets the placeholder after it
    assert repaired[2] == tool_msg("call_0", "r0")
    assert repaired[3] == tool_repair._synthetic_result("call_1")


# ---------------------------------------------------------------------------
# rules 2+3: duplicates, adjacency, interleave, late declaration
# ---------------------------------------------------------------------------

def test_duplicate_ids_renamed_and_paired():
    messages = [
        USER,
        assistant([tool_call("call_0", "f1"), tool_call("call_0", "f2")]),
        tool_msg("call_0", "r1"),
        tool_msg("call_0", "r2"),
        END,
    ]
    repaired, counts = repair(messages)
    assert counts["renamed"] == 2
    calls = repaired[1]["tool_calls"]
    assert calls[0]["id"] == "call_0"
    assert calls[1]["id"] != "call_0"
    # results immediately follow the block, in call order, ids matching
    assert repaired[2]["tool_call_id"] == "call_0"
    assert repaired[2]["content"] == "r1"
    assert repaired[3]["tool_call_id"] == calls[1]["id"]
    assert repaired[3]["content"] == "r2"


def test_duplicate_results_one_call():
    messages = [
        USER,
        assistant([tool_call("call_0")]),
        tool_msg("call_0", "r1"),
        tool_msg("call_0", "r2"),
        END,
    ]
    repaired, counts = repair(messages)
    # second result has no second call: stub declaration, fresh id
    assert counts["declared"] == 1
    assert counts["renamed"] == 1
    assert repaired[2] == tool_msg("call_0", "r1")
    stub_call = repaired[3]["tool_calls"][0]
    assert stub_call["id"] != "call_0"
    assert repaired[4]["tool_call_id"] == stub_call["id"]
    assert repaired[4]["content"] == "r2"


def test_interleaved_user_moves_after_result():
    messages = [
        USER,
        assistant([tool_call("call_0")]),
        {"role": "user", "content": "quick question"},
        tool_msg("call_0"),
        END,
    ]
    repaired, counts = repair(messages)
    assert counts["moved"] == 1
    assert roles(repaired) == ["user", "assistant", "tool", "user", "user"]


def test_late_declaration_result_moves_up():
    messages = [
        USER,
        tool_msg("call_0"),
        assistant([tool_call("call_0")]),
        END,
    ]
    repaired, counts = repair(messages)
    assert counts["moved"] == 1
    assert roles(repaired) == ["user", "assistant", "tool", "user"]


# ---------------------------------------------------------------------------
# rule 3: complex real-world compaction damage, end state is strict-clean
# ---------------------------------------------------------------------------

def test_compaction_damage_end_to_end():
    """The production shape: declarations dropped, results orphaned, one
    call unanswered. After repair the array satisfies the strictest
    contract (kimi/deepseek): every tool_calls block immediately followed by
    exactly one result per call, ids unique."""
    messages = [
        {"role": "system", "content": "You are Cursor."},
        USER,
        assistant([tool_call("call_0", "read_file"), tool_call("call_1", "grep")]),
        # compaction dropped call_1's result; call_0's survived:
        tool_msg("call_0", "file contents"),
        tool_msg("call_99", "orphaned"),       # declaration lost entirely
        {"role": "user", "content": "continue"},
        assistant([tool_call("call_2", "write")]),  # its result was lost
        END,
    ]
    repaired, counts = repair(messages)
    assert counts["declared"] == 1
    assert counts["answered"] == 2   # call_1 and call_2 both lost their results

    # strict contract walk: ids unique, each block immediately followed by
    # exactly one result per call
    seen_ids = set()
    index = 0
    while index < len(repaired):
        message = repaired[index]
        if message.get("role") == "assistant" and message.get("tool_calls"):
            ids = []
            for call in message["tool_calls"]:
                cid = call["id"]
                assert cid not in seen_ids
                seen_ids.add(cid)
                ids.append(cid)
            for offset, cid in enumerate(ids, start=1):
                result = repaired[index + offset]
                assert result.get("role") == "tool"
                assert result.get("tool_call_id") == cid
            index += len(ids) + 1
        else:
            assert message.get("role") != "tool", f"orphan tool result at {index}"
            index += 1

    # real content survives every repair
    blob = json.dumps(repaired)
    assert "file contents" in blob
    assert "orphaned" in blob


def test_repair_is_idempotent():
    messages = [
        USER,
        tool_msg("call_99", "lost"),
        assistant([tool_call("call_0"), tool_call("call_0")]),
        tool_msg("call_0", "r1"),
        END,
    ]
    once, _ = repair(messages)
    twice, counts = repair(once)
    assert twice is once
    assert sum(counts.values()) == 0


# ---------------------------------------------------------------------------
# logging: counts only, never message text (C5)
# ---------------------------------------------------------------------------

def test_log_records_counts_only():
    with tempfile.TemporaryDirectory() as tmp:
        tool_repair.REPAIR_LOG_PATH = os.path.join(tmp, "tool_repair.jsonl")
        tool_repair.record_repair({"declared": 1, "answered": 2}, now=1000.0)
        with open(tool_repair.REPAIR_LOG_PATH, encoding="utf-8") as handle:
            record = json.loads(handle.readline())
        assert record == {"ts": 1000.0, "declared": 1, "answered": 2,
                          "renamed": 0, "moved": 0}


def test_record_repair_never_raises():
    with tempfile.TemporaryDirectory() as tmp:
        tool_repair.REPAIR_LOG_PATH = os.path.join(tmp, "nested", "tool_repair.jsonl")
        tool_repair.record_repair({})          # missing keys fine
        tool_repair.record_repair(None)        # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# the hook itself: fail-open, kill-switch, patched data returned
# ---------------------------------------------------------------------------

def test_hook_patches_messages():
    hook = tool_repair.ToolRepairHook()
    data = {"model": "cursor-auto",
            "messages": [USER, tool_msg("call_99", "lost"), END]}
    result = asyncio.run(hook.async_pre_call_hook(None, None, data, "acompletion"))
    assert isinstance(result, dict)
    assert result["messages"] is not data["messages"]
    assert result["messages"][1]["role"] == "assistant"
    assert result["model"] == "cursor-auto"


def test_hook_noop_on_clean():
    hook = tool_repair.ToolRepairHook()
    messages = [USER, assistant([tool_call("call_0")]), tool_msg("call_0"), END]
    data = {"messages": messages}
    result = asyncio.run(hook.async_pre_call_hook(None, None, data, "acompletion"))
    assert result is None


def test_hook_kill_switch():
    hook = tool_repair.ToolRepairHook()
    os.environ["POLICY_TOOL_REPAIR"] = "0"
    try:
        data = {"messages": [USER, tool_msg("call_99", "lost"), END]}
        result = asyncio.run(hook.async_pre_call_hook(None, None, data, "acompletion"))
        assert result is None
    finally:
        os.environ["POLICY_TOOL_REPAIR"] = "1"


def test_hook_never_raises():
    hook = tool_repair.ToolRepairHook()
    for junk in (None, {}, {"messages": "text"}, {"messages": None}, 42):
        result = asyncio.run(hook.async_pre_call_hook(None, None, junk, "acompletion"))
        assert result is None


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print(f"PASS {name}")
            except AssertionError as error:
                failures += 1
                print(f"FAIL {name}: {error}")
            except Exception as error:  # noqa: BLE001
                failures += 1
                print(f"ERROR {name}: {type(error).__name__}: {error}")
    print("---")
    print("all green" if failures == 0 else f"{failures} failing")
    sys.exit(1 if failures else 0)
