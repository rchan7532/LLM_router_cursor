"""Tests for extractor.py: prompt construction, response parsing, and the
retry/drop policy - all with a stubbed HTTP layer, no network.

    python tests/test_extractor.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import extractor  # noqa: E402
import learning  # noqa: E402


def _envelope(content: str) -> bytes:
    return json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------

def test_prompt_lists_kinds_and_asks():
    prompt = extractor.build_prompt(["fix the parser bug", "design the sync layer"])
    for kind in ("debug", "design", "code_gen", "factual"):
        assert kind in prompt, kind
    assert "1. fix the parser bug" in prompt
    assert "2. design the sync layer" in prompt
    assert "[REDACTED]" in prompt or "credential" in prompt.lower()


def test_prompt_names_only_base_bar_kinds():
    # review claim 14/m5: kind validation must use BASE_BAR keys, not
    # KIND_KEYWORDS ('writing' is unreachable via keywords but is a kind).
    prompt = extractor.build_prompt(["something"])
    assert "writing" in prompt
    assert extractor.KINDS == learning.KINDS


# --------------------------------------------------------------------------
# Response parsing (mocked HTTP bodies)
# --------------------------------------------------------------------------

def test_valid_batch_parses():
    content = json.dumps({"items": [
        {"kind": "debug", "key_points": ["parser fails"], "phrases": ["sort this out"]},
        {"kind": "design", "key_points": ["sync layer"], "phrases": []},
    ]})
    items = extractor.parse_response(_envelope(content))
    assert len(items) == 2
    assert items[0]["kind"] == "debug"
    assert items[0]["key_points"] == ["parser fails"]


def test_malformed_output_drops_all_items():
    for content in ("", "not json at all", "```json\n{oops\n```", '{"items": 5}',
                    '{"no_items": []}', "[", '{"items": ["a string"]}'):
        assert extractor.parse_response(_envelope(content)) == [], content


def test_unknown_or_missing_kind_dropped():
    content = json.dumps({"items": [
        {"kind": "rocket-science", "key_points": ["x"], "phrases": []},
        {"key_points": ["x"], "phrases": []},
        {"kind": 42, "key_points": ["x"], "phrases": []},
        {"kind": "debug", "key_points": ["kept"], "phrases": []},
    ]})
    items = extractor.parse_response(_envelope(content))
    assert len(items) == 1 and items[0]["kind"] == "debug"


def test_count_violation_drops_item():
    # Four key points violates the <=3 cap: the item is dropped entirely.
    content = json.dumps({"items": [
        {"kind": "debug", "key_points": ["a", "b", "c", "d"], "phrases": []},
    ]})
    assert extractor.parse_response(_envelope(content)) == []


def test_non_string_members_dropped():
    content = json.dumps({"items": [
        {"kind": "debug", "key_points": ["ok", 7, None], "phrases": [True, "ok phrase"]},
    ]})
    items = extractor.parse_response(_envelope(content))
    assert items == [{"kind": "debug", "key_points": ["ok"], "phrases": ["ok phrase"]}]


def test_markdown_fenced_json_tolerated():
    content = '```json\n{"items": [{"kind": "bulk", "key_points": ["rename all"], "phrases": []}]}\n```'
    items = extractor.parse_response(_envelope(content))
    assert len(items) == 1 and items[0]["kind"] == "bulk"


# --------------------------------------------------------------------------
# Transport: timeout, retry, drop policy, no payload in errors
# --------------------------------------------------------------------------

def test_extract_noop_without_credentials(monkeypatch):
    monkeypatch.setattr(extractor, "EXTRACT_BASE_URL", "")
    monkeypatch.setattr(extractor, "EXTRACT_API_KEY", "")
    assert asyncio.run(extractor.extract(["ask"])) == []


def test_extract_retries_then_drops_batch(monkeypatch):
    calls = {"n": 0}

    def fake_post(payload, timeout):
        calls["n"] += 1
        raise RuntimeError("extraction HTTP 503")

    monkeypatch.setattr(extractor, "EXTRACT_BASE_URL", "https://upstream.example/v1")
    monkeypatch.setattr(extractor, "EXTRACT_API_KEY", "key")
    monkeypatch.setattr(extractor, "_post_chat_async", _fake_async(fake_post))
    monkeypatch.setattr(extractor.time, "sleep", lambda _s: None)
    assert asyncio.run(extractor.extract(["ask one", "ask two"])) == []
    assert calls["n"] == 2   # one attempt + one retry, then dropped


def test_extract_success_after_one_retry(monkeypatch):
    calls = {"n": 0}

    def fake_post(payload, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("extraction HTTP 503")
        content = json.dumps({"items": [{"kind": "debug", "key_points": ["ok"], "phrases": []}]})
        return _envelope(content)

    monkeypatch.setattr(extractor, "EXTRACT_BASE_URL", "https://upstream.example/v1")
    monkeypatch.setattr(extractor, "EXTRACT_API_KEY", "key")
    monkeypatch.setattr(extractor, "_post_chat_async", _fake_async(fake_post))
    monkeypatch.setattr(extractor.time, "sleep", lambda _s: None)
    items = asyncio.run(extractor.extract(["ask"]))
    assert len(items) == 1 and items[0]["kind"] == "debug"
    assert calls["n"] == 2


def test_extract_uses_direct_upstream_not_proxy(monkeypatch):
    # review M2: the extraction call must carry the upstream provider's own
    # base URL and key, and must never traverse the proxy (no /v1 through
    # litellm, no cursor-auto group, no master key).
    seen = {}

    def fake_post(payload, timeout):
        seen["url_base"] = extractor.EXTRACT_BASE_URL
        seen["model"] = payload["model"]
        seen["key"] = extractor.EXTRACT_API_KEY
        content = json.dumps({"items": [{"kind": "debug", "key_points": ["ok"], "phrases": []}]})
        return _envelope(content)

    monkeypatch.setattr(extractor, "EXTRACT_BASE_URL", "https://provider.example/v1")
    monkeypatch.setattr(extractor, "EXTRACT_API_KEY", "provider-key")
    monkeypatch.setattr(extractor, "_post_chat_async", _fake_async(fake_post))
    asyncio.run(extractor.extract(["ask"]))
    assert seen["url_base"] == "https://provider.example/v1"
    assert seen["model"] == extractor.EXTRACT_MODEL
    assert seen["key"] == "provider-key"


def test_http_error_message_carries_no_payload(monkeypatch):
    # review M4: never interpolate payloads into exceptions. Patch the
    # TRANSPORT (urlopen), not _post_chat, so the translation layer runs.
    class FakeHTTPError(urllib.error.HTTPError):
        def __init__(self):
            super().__init__("url", 500, "boom", hdrs=None, fp=None)

    def fake_urlopen(request, timeout):
        raise FakeHTTPError()

    monkeypatch.setattr(extractor, "EXTRACT_BASE_URL", "https://provider.example/v1")
    monkeypatch.setattr(extractor, "EXTRACT_API_KEY", "provider-key")
    monkeypatch.setattr(extractor.urllib.request, "urlopen", fake_urlopen)
    try:
        extractor._post_chat({"messages": [{"role": "user", "content": "secret-payload-abc"}]}, 1.0)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as error:
        assert "secret-payload-abc" not in str(error)
        assert "sk-" not in str(error)


def _fake_async(function):
    async def runner(payload, timeout):
        return function(payload, timeout)
    return runner


# --------------------------------------------------------------------------
# End-to-end through the learner's persistence path (C5: sanitized on disk)
# --------------------------------------------------------------------------

def test_real_extractor_output_flows_through_sanitize():
    with __import__("tempfile").TemporaryDirectory() as tmp:
        obs_path = os.path.join(tmp, "observations.jsonl")
        learner = learning.Learner(os.path.join(tmp, "learned.json"), obs_path)

        async def upstream(asks):
            # What the real provider returns when the user pasted a key.
            return [{"kind": "debug", "key_points": ["rotate the key sk-abcdefghijklmnop1234"],
                     "phrases": ["broken again"]}]

        learner.enqueue("the api key sk-abcdefghijklmnop1234 is broken again", "debug", "m", 1)
        assert learner.process_queue(upstream) == 1
        with open(obs_path, encoding="utf-8") as handle:
            blob = handle.read()
        assert "sk-abcdefghijklmnop1234" not in blob
        assert "[REDACTED]" in blob


if __name__ == "__main__":
    class _Monkey:
        """Tiny monkeypatch stand-in so the no-pytest runner can supply the
        `monkeypatch` argument the same way pytest would."""

        def __init__(self):
            self._saved = []

        def setattr(self, target, name, value):
            self._saved.append((target, name, getattr(target, name, None)))
            setattr(target, name, value)

        def undo(self):
            for target, name, old in reversed(self._saved):
                setattr(target, name, old)
            self._saved.clear()

    failures = 0
    import inspect
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            monkey = _Monkey()
            try:
                if "monkeypatch" in inspect.signature(function).parameters:
                    function(monkey)
                else:
                    function()
                print(f"PASS {name}")
            except AssertionError as error:
                failures += 1
                print(f"FAIL {name}: {error}")
            except Exception as error:  # noqa: BLE001
                failures += 1
                print(f"ERROR {name}: {type(error).__name__}: {error}")
            finally:
                monkey.undo()
    print("---")
    print("all green" if failures == 0 else f"{failures} failing")
    sys.exit(1 if failures else 0)
