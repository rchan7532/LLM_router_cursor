"""Tests for laya_scorer.py.

The integration test that actually loads convaiinnovations/laya is skipped
when the package is not installed. The rest exercise the module with mocks.

    python tests/test_laya_scorer.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import laya_scorer as scorer  # noqa: E402

try:
    import laya  # noqa: F401
    LAYA_INSTALLED = True
except ImportError:
    LAYA_INSTALLED = False


def test_inert_when_laya_missing():
    # When laya is not installed the public API must be a no-op and never
    # raise, so routing stays byte-identical.
    scorer.maybe_enqueue("any prompt text", "bulk")
    assert scorer.get_scores("any prompt text") is None


def test_key_is_stable_and_private():
    assert scorer._key("same text") == scorer._key("same text")
    assert scorer._key("text a") != scorer._key("text b")


class _FakeAgent:
    def predict(self, state, questions):
        answers = {}
        for kind in questions:
            answers[kind] = {"score": 1.5}  # mid-range on 0..2 scale
        return {"answers": answers}


def test_score_with_mock_agent():
    old_agent = scorer._agent
    scorer._agent = _FakeAgent()
    try:
        scores = scorer._score("refactor the parser")
        for kind in scorer.LAYA_KINDS:
            assert kind in scores
            assert 0.0 <= scores[kind] <= 1.0
        # 1.5 / 2.0 = 0.75
        assert abs(scores["bulk"] - 0.75) < 1e-9
    finally:
        scorer._agent = old_agent


def test_log_does_not_store_raw_text():
    with tempfile.TemporaryDirectory() as tmp:
        scorer.LAYA_SHADOW_PATH = os.path.join(tmp, "laya_shadow.jsonl")
        scorer._cache.clear()
        text = "this is secret prompt text"
        scorer._write_log(text, {"bulk": 0.5})
        with open(scorer.LAYA_SHADOW_PATH, encoding="utf-8") as handle:
            record = json.loads(handle.readline())
        assert "secret" not in json.dumps(record)
        assert record["key"] == scorer._key(text)
        assert record["scores"]["bulk"] == 0.5


def test_cache_roundtrip():
    scorer._cache.clear()
    key = scorer._key("roundtrip text")
    scorer._cache[key] = {"explain": 0.9}
    assert scorer.get_scores("roundtrip text") == {"explain": 0.9}


if LAYA_INSTALLED:
    def test_laya_loads_without_error():
        # Smoke test only: ensure the lazy loader does not crash when laya is
        # present. The actual checkpoint download is network-dependent and
        # intentionally not exercised here.
        agent = scorer._load_agent()
        assert agent is not None


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
