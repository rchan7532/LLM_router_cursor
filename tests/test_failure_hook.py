"""Tests for failure_hook.py (phase 3, reduced scope): 429/5xx recorded,
successes not, learning-off inert, hook never raises, no routing effect.

    python tests/test_failure_hook.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import failure_hook  # noqa: E402


class _KW(dict):
    """kwargs stand-in that also quacks like a Mapping."""


def _fresh(tmp, learn=True):
    failure_hook.RELIABILITY_PATH = os.path.join(tmp, "reliability.jsonl")
    failure_hook._learn_on = (lambda: True) if learn else (lambda: False)
    failure_hook.RELIABILITY.clear()


def test_429_recorded(tmp=None):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh(tmp)
        record = failure_hook.record_failure("openai/glm-5.3-flash", 429, now=1000.0)
        assert record == {"ts": 1000.0, "model": "openai/glm-5.3-flash", "status": 429}
        with open(failure_hook.RELIABILITY_PATH, encoding="utf-8") as handle:
            assert json.loads(handle.readline())["status"] == 429
        snap = failure_hook.snapshot()
        assert snap["openai/glm-5.3-flash"]["count_429"] == 1


def test_5xx_recorded(tmp=None):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh(tmp)
        failure_hook.record_failure("openai/kimi-k2.7-code", 503, now=1000.0)
        assert failure_hook.snapshot()["openai/kimi-k2.7-code"]["count_5xx"] == 1


def test_success_is_not_a_signal(tmp=None):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh(tmp)
        hook = failure_hook.ReliabilityHook()
        hook.log_success_event(_KW(model="openai/glm-5.3"), object(), 0, 0)
        import asyncio
        asyncio.run(hook.async_log_success_event(_KW(model="openai/glm-5.3"), object(), 0, 0))
        assert failure_hook.snapshot() == {}
        assert not os.path.exists(failure_hook.RELIABILITY_PATH)


def test_learning_off_hook_inert(tmp=None):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh(tmp, learn=False)
        hook = failure_hook.ReliabilityHook()
        kwargs = _KW(model="openai/glm-5.3", exception=type("E", (), {"status_code": 429})())
        hook.log_failure_event(kwargs, None, 0, 0)
        import asyncio
        asyncio.run(hook.async_log_failure_event(kwargs, None, 0, 0))
        assert failure_hook.snapshot() == {}
        assert not os.path.exists(failure_hook.RELIABILITY_PATH)


def test_hook_never_raises(tmp=None):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh(tmp)
        hook = failure_hook.ReliabilityHook()
        # Garbage in every field: no exception may escape.
        hook.log_failure_event(None, "junk", None, None)
        hook.log_failure_event(_KW(nonsense=True), 12345, "x", object())
        import asyncio
        asyncio.run(hook.async_log_failure_event(_KW(), None, None, None))
        asyncio.run(hook.async_post_call_failure_hook(None, None, None))


def test_no_payload_fields_in_record(tmp=None):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh(tmp)
        hook = failure_hook.ReliabilityHook()
        kwargs = _KW(model="openai/glm-5.3",
                     exception=type("E", (), {"status_code": 502})(),
                     messages=[{"role": "user", "content": "CANARY-prompt-text"}])
        hook.log_failure_event(kwargs, None, 0, 0)
        with open(failure_hook.RELIABILITY_PATH, encoding="utf-8") as handle:
            record = json.loads(handle.readline())
        assert set(record) == {"ts", "model", "status"}
        assert "CANARY" not in json.dumps(record)


def test_status_extraction_from_exception(tmp=None):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh(tmp)
        hook = failure_hook.ReliabilityHook()
        kwargs = _KW(model="openai/deepseek-v4.1-flash",
                     exception=type("E", (), {"status_code": 500})())
        hook.log_failure_event(kwargs, None, 0, 0)
        assert failure_hook.snapshot()["openai/deepseek-v4.1-flash"]["count_5xx"] == 1


def test_generic_error_counted(tmp=None):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh(tmp)
        failure_hook.record_failure("m", None, now=1.0)
        assert failure_hook.snapshot()["m"]["count_error"] == 1


def test_rotation(tmp=None):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh(tmp)
        failure_hook.RELIABILITY_MAX_BYTES = 10
        failure_hook.record_failure("m", 429, now=1.0)
        failure_hook.record_failure("m", 429, now=2.0)
        assert os.path.exists(failure_hook.RELIABILITY_PATH + ".1")


def test_record_failure_rejects_empty_model(tmp=None):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh(tmp)
        assert failure_hook.record_failure("", 429) is None
        assert failure_hook.record_failure(None, 429) is None


def test_last_failure_ts_returns_newest_failure(tmp=None):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh(tmp)
        assert failure_hook.last_failure_ts("openai/glm-5.3") is None
        failure_hook.record_failure("openai/glm-5.3", 429, now=1000.0)
        assert failure_hook.last_failure_ts("openai/glm-5.3") == 1000.0
        failure_hook.record_failure("openai/glm-5.3", 503, now=1001.0)
        assert failure_hook.last_failure_ts("openai/glm-5.3") == 1001.0
        assert failure_hook.last_failure_ts("openai/kimi-k2.7-code") is None


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
