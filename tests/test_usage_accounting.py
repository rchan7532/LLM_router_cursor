"""
Tests for the measurement plumbing: usage_state (pricing + receipt store),
usage_hook (post-call extractor), and routing_policy's client tag /
real-spend kind-alarm branch.

Run from the llm-router directory:

    python -m pytest tests/test_usage_accounting.py -v
    # or, without pytest:
    python tests/test_usage_accounting.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Isolation BEFORE import (same discipline as test_routing_policy): a
# throwaway state dir so neither the production log nor %TEMP% leftovers
# leak into these assertions. POLICY_USAGE_* must point at the same dir
# because usage_state resolves its paths at import time.
_TMP = tempfile.mkdtemp(prefix="usage-tests-")
os.environ["POLICY_STATE_DIR"] = _TMP
os.environ["POLICY_LOG"] = ""
os.environ["POLICY_USAGE_LOG"] = os.path.join(_TMP, "routing_usage.jsonl")
os.environ["POLICY_USAGE_STATE"] = os.path.join(_TMP, "usage_state.json")

import routing_policy as policy  # noqa: E402
import usage_hook  # noqa: E402
import usage_state  # noqa: E402


class Context:
    def __init__(self, messages, candidates=None, metadata=None):
        self.raw_messages = list(messages)
        self.structured_messages = list(messages)
        self.candidate_models = list(candidates) if candidates is not None else list(policy.PROFILES)
        self.metadata = metadata or {}
        self.signals = {}


def ask(text, first="seed question"):
    return [
        {"role": "system", "content": "You are Cursor."},
        {"role": "user", "content": first},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": text},
    ]


def run(context):
    return asyncio.run(policy.CursorAutoPolicy().run(context))


def reset():
    policy.STATE = policy.PolicyState()
    # A clean control view: no control.json (no lease/budget), pristine
    # learned (no bar bias / persisted trust). default_store() reads
    # POLICY_STATE_DIR, which an earlier test file (test_control) repoints at
    # a directory holding a hand-written control.json - inheriting that would
    # make these assertions order-dependent. Point at our own empty temp dir
    # so the decision is deterministic regardless of suite order.
    policy.CONTROL = policy.ControlStore(os.path.join(_TMP, "control.json"),
                                         os.path.join(_TMP, "learned.json")) \
        if policy.ControlStore is not None else None
    policy.LEARN_ENABLED = True
    policy._LAST_CURSOR_AUTO_SESSION.update({"key": None, "ask": ""})


class FakeUsage:
    def __init__(self, prompt_tokens, completion_tokens, cached_tokens=None):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.prompt_tokens_details = (
            type("D", (), {"cached_tokens": cached_tokens})()
            if cached_tokens is not None else None
        )


class FakeResponse:
    def __init__(self, usage):
        self.usage = usage


# --- usage_state: pricing --------------------------------------------------

def test_real_cost_splits_cached_from_fresh():
    # glm-5.3: in 1.13, out 3.54, cache_in 0.113 USD/1M; HKD 7.8/USD.
    # 100k prompt of which 90k cached, 1k output:
    cost = usage_state.real_cost_hkd("openai/glm-5.3", 100_000, 1_000, 90_000)
    expected = (10_000 / 1e6 * 1.13 + 90_000 / 1e6 * 0.113 + 1_000 / 1e6 * 3.54) * 7.8
    assert abs(cost - expected) < 1e-9
    # All-uncached must equal the naive full-price bill.
    full = usage_state.real_cost_hkd("openai/glm-5.3", 100_000, 1_000, 0)
    assert full > 4 * cost  # the cache discount is the whole 70-vs-15 story


def test_real_cost_clamps_cached_above_prompt_and_prices_unknown_as_zero():
    assert usage_state.real_cost_hkd("openai/glm-5.3", 100, 10, 500) == \
        usage_state.real_cost_hkd("openai/glm-5.3", 100, 10, 100)
    assert usage_state.real_cost_hkd("openai/nope-9", 1_000_000, 1_000_000, 0) == 0.0


def test_pricing_table_cannot_drift_from_profiles():
    """PRICING duplicates PROFILES costs on purpose (usage_state must stay
    stdlib-only so two litellm module instances can load it). This test is
    the drift guard: in/out must match to the cent, cache must be in/10 and
    match litellm-config.yaml's documented ratio."""
    for model, profile in policy.PROFILES.items():
        assert model in usage_state.PRICING, f"{model} missing from PRICING"
        cost_in, cost_out, cache_in = usage_state.PRICING[model]
        assert cost_in == profile.cost_in, model
        assert cost_out == profile.cost_out, model
        assert abs(cache_in - cost_in / 10) < 1e-12, model
    assert usage_state.HKD_PER_USD == policy.HKD_PER_USD


# --- usage_state: receipt store + reader ----------------------------------

def test_record_receipt_writes_log_and_view_roundtrip():
    session = "sess-roundtrip"
    real = usage_state.record_receipt(
        session=session, model="openai/glm-5.3", kind="design", client="cursor",
        prompt_tokens=100_000, completion_tokens=2_000, cached_tokens=90_000,
        est_cost_hkd=1.5,
    )
    assert real > 0
    view = usage_state.load_view(session)
    assert abs(view.real_hkd - real) < 1e-9
    assert view.kind_hkd[("openai/glm-5.3", "design")] == round(real, 6)
    assert view.client_hkd["cursor"] == round(real, 6)
    # The durable receipt carries both numbers for the audit trail.
    lines = open(usage_state.USAGE_LOG_PATH, encoding="utf-8").read().splitlines()
    record = json.loads(lines[-1])
    assert record["session"] == session and record["client"] == "cursor"
    assert record["cached_tokens"] == 90_000 and record["est_cost_hkd"] == 1.5


def test_load_view_unknown_session_and_missing_file_are_empty():
    view = usage_state.load_view("never-seen")
    assert view.real_hkd == 0.0 and not view.kind_hkd
    assert usage_state.load_view(None).real_hkd == 0.0


def test_load_view_picks_up_file_updates():
    session = "sess-mtime"
    unit = usage_state.real_cost_hkd("openai/qwen3-vl-flash", 100_000, 100)
    for _ in range(2):
        usage_state.record_receipt(session=session, model="openai/qwen3-vl-flash",
                                   kind="bulk", client="aider",
                                   prompt_tokens=100_000, completion_tokens=100)
    view = usage_state.load_view(session)
    assert view.real_hkd > unit, (view.real_hkd, unit)


# --- usage_hook: extraction --------------------------------------------------

def _kwargs(signals_policy=None, headers=None, model="openai/glm-5.3", signals=None):
    merged = dict(signals or {})
    if signals_policy:
        merged["policy"] = signals_policy
    return {
        "model": model,
        "metadata": ({"headers": headers} if headers is not None else {}),
        "litellm_params": {
            "metadata": {
                "routing_plugin_signals": merged
            }
        },
    }


def test_extract_uses_policy_signals_as_join_key():
    parts = usage_hook.extract_receipt_parts(
        _kwargs({"session": "abc", "kind": "design", "chosen": "openai/glm-5.3",
                 "client": "cursor", "est_cost_hkd": 0.7}),
        FakeResponse(FakeUsage(120_000, 1_500, 110_000)),
    )
    assert parts["session"] == "abc" and parts["kind"] == "design"
    assert parts["client"] == "cursor" and parts["est_cost_hkd"] == 0.7
    assert parts["prompt_tokens"] == 120_000 and parts["cached_tokens"] == 110_000


def test_extract_falls_back_to_headers_and_model_when_no_policy():
    parts = usage_hook.extract_receipt_parts(
        _kwargs(None, headers={"user-agent": "aider/0.86.2"}, model="glm-5.3-flash"),
        FakeResponse(FakeUsage(100, 20, None)),
    )
    assert parts["client"] == "aider"
    assert parts["model"] == "openai/glm-5.3-flash"  # normalised to the PRICING key
    assert parts["session"] is None and parts["kind"] is None


def test_extract_no_usage_returns_none():
    assert usage_hook.extract_receipt_parts(_kwargs({"session": "s"}), FakeResponse(None)) is None
    assert usage_hook.extract_receipt_parts(_kwargs(), None) is None


def test_recorder_end_to_end_never_raises():
    reset()
    hook = usage_hook.UsageRecorder()
    # dict-shaped usage (streaming assembly path) must work too.
    response = type("R", (), {"usage": {"prompt_tokens": 5000, "completion_tokens": 40,
                                        "prompt_tokens_details": {"cached_tokens": 4500}}})()
    asyncio.run(hook.async_log_success_event(
        _kwargs({"session": "sess-hook", "kind": "explain", "client": "other",
                 "est_cost_hkd": 0.1}), response, None, None))
    view = usage_state.load_view("sess-hook")
    assert view.real_hkd > 0 and view.client_hkd.get("other", 0) > 0
    # garbage in, silence out (C1)
    asyncio.run(hook.async_log_success_event("not a mapping", 42, None, None))


# --- routing_policy: client tag + real-spend alarm ---------------------------

def test_client_tag_from_headers_and_default():
    reset()
    ctx = Context(ask("quick question"), metadata={"headers": {"user-agent": "aider/0.86"}})
    run(ctx)
    assert ctx.signals["policy"]["client"] == "aider"

    # X-Client-Type (nginx path stamp) wins over a UA that doesn't name the
    # tool - aider's real UA lacks the string "aider", which misattributed
    # its first whole session as "other".
    reset()
    ctx = Context(ask("quick question"),
                  metadata={"headers": {"user-agent": "Aider/0.86.2 (https://aider.chat)",
                                        "X-Client-Type": "aider"}})
    run(ctx)
    assert ctx.signals["policy"]["client"] == "aider"

    reset()
    ctx = Context(ask("quick question"),
                  metadata={"headers": {"User-Agent": "Cursor-Server/2.0"}})
    run(ctx)
    assert ctx.signals["policy"]["client"] == "cursor"

    reset()
    ctx = Context(ask("quick question"), metadata={"headers": {"user-agent": "curl/8.0"}})
    run(ctx)
    assert ctx.signals["policy"]["client"] == "other"

    reset()
    ctx = Context(ask("quick question"))
    run(ctx)
    assert ctx.signals["policy"]["client"] == "unknown"


def _prime_real_spend(session, model, kind, total_hkd):
    """Write receipts until the session's real (model, kind) total reaches
    the target, using exact token counts for a priced model."""
    unit = usage_state.real_cost_hkd(model, 100_000, 0, 0)
    assert unit > 0
    calls = max(1, int(round(total_hkd / unit)))
    for _ in range(calls):
        usage_state.record_receipt(session=session, model=model, kind=kind,
                                   client="cursor", prompt_tokens=100_000,
                                   completion_tokens=0, cached_tokens=0)
    return usage_state.load_view(session)


def test_kind_alarm_fires_on_real_spend():
    """The 2026-09 kit-generation story (kimi, code_gen, 73 HKD): once the
    session's RECEIPTS for one premium model + kind cross KIND_ALARM_HKD,
    the next same-kind turn is forced to the cheapest model that still
    clears the bar, with reason kind-alarm and basis real. No RAM estimate
    is involved - this fires straight off usage_state.json."""
    reset()
    premium = "openai/kimi-k2.7-code"
    msgs = ask("implement the widget kit", first="implement the widget kit")
    session = policy.STATE.session_key(msgs, {})
    _prime_real_spend(session, premium, "code_gen", policy.KIND_ALARM_HKD + 1.0)

    ctx = Context(msgs)
    run(ctx)
    decision = ctx.signals["policy"]
    assert decision["kind"] == "code_gen", decision
    assert decision["kind_alarm"] is True, decision
    assert decision["kind_alarm_basis"] == "real", decision
    assert decision["kind_alarm_model"] == premium
    assert decision["reason"] == "kind-alarm", decision
    # Cheapest model clearing the code_gen bar is the flash, not another
    # premium: this is the outcome the alarm exists to produce.
    assert ctx.candidate_models == ["openai/glm-5.3-flash"], ctx.candidate_models
    assert policy._blended_cost(policy.PROFILES[ctx.candidate_models[0]]) < policy.PREMIUM_COST


def test_real_alarm_ignores_cheap_models_and_other_kinds():
    reset()
    # A cheap model may burn any amount without alarming: forcing "the
    # cheapest capable" would be a no-op anyway, and cheap pins are the goal.
    msgs = ask("rename every file across the repo", first="rename every file across the repo")
    session = policy.STATE.session_key(msgs, {})
    _prime_real_spend(session, "openai/qwen3-vl-flash", "bulk", 100.0)
    ctx = Context(msgs)
    run(ctx)
    assert ctx.signals["policy"]["kind_alarm"] is False, ctx.signals["policy"]

    # Premium spend booked to a DIFFERENT kind must not alarm this turn.
    msgs2 = ask("rename every file across the repo", first="shared seed for kinds")
    session2 = policy.STATE.session_key(msgs2, {})
    _prime_real_spend(session2, "openai/glm-5.3", "design", policy.KIND_ALARM_HKD + 5.0)
    ctx2 = Context(msgs2)  # kind: bulk, receipts only cover design -> no alarm
    run(ctx2)
    assert ctx2.signals["policy"]["kind_alarm"] is False


def test_real_spend_reported_in_decision_and_log_fields():
    reset()
    session_first = "seed real report"
    msgs = ask("explain this module", first=session_first)
    # 3x ~0.883 HKD receipts for this session: the decision must report the
    # rounded total the file holds (priming uses 100k-prompt calls, and the
    # reader rounds to 4 decimals).
    _prime_real_spend(policy.STATE.session_key(msgs, {}),
                      "openai/glm-5.3", "explain", 3.0)
    ctx = Context(msgs)
    run(ctx)
    decision = ctx.signals["policy"]
    assert decision["real_spend_hkd"] >= 2.5, decision
    assert decision["client"] == "unknown"


def test_missing_usage_module_keeps_decisions_working(monkeypatch=None):
    """Fail-open contract: with _usage_state absent, routing must not raise
    and the alarm must fall back to estimate-only behaviour."""
    reset()
    saved = policy._usage_state
    try:
        policy._usage_state = None
        ctx = Context(ask("design the retry policy"))
        run(ctx)
        assert ctx.signals["policy"]["chosen"] in policy.PROFILES
        assert ctx.signals["policy"]["real_spend_hkd"] == 0.0
        assert ctx.signals["policy"]["kind_alarm"] is False
    finally:
        policy._usage_state = saved


def test_client_does_not_change_routing_decision():
    """Attribution is telemetry: the same ask from aider and from cursor
    must pick the same model on a fresh STATE."""
    reset()
    a = Context(ask("write a script that renames every file across the repo"),
                metadata={"headers": {"user-agent": "aider/0.86"}})
    run(a)
    reset()
    b = Context(ask("write a script that renames every file across the repo"),
                metadata={"headers": {"user-agent": "cursor-server/2.0"}})
    run(b)
    assert a.candidate_models == b.candidate_models
    assert a.signals["policy"]["client"] != b.signals["policy"]["client"]


def test_receipt_roundtrip_with_task():
    reset()
    usage_state.record_receipt(session="s1", model="openai/glm-5.3-flash",
                               kind="code_gen", client="aider", task="module-kit",
                               prompt_tokens=1000, completion_tokens=100, cached_tokens=0)
    records = [json.loads(l) for l in open(usage_state.USAGE_LOG_PATH, encoding="utf-8")]
    assert records[-1]["task"] == "module-kit"
    view = usage_state.load_view("s1")
    assert view.task_hkd[("module-kit", "aider")] > 0.0


def test_receipt_task_none_keeps_old_shape():
    reset()
    usage_state.record_receipt(session="s2", model="openai/glm-5.3-flash",
                               kind="factual", client="cursor",
                               prompt_tokens=10, completion_tokens=5)
    view = usage_state.load_view("s2")
    assert view.task_hkd == {}


def test_hook_extract_task_from_policy_signals():
    kwargs = _kwargs(signals_policy={
        "session": "s3", "kind": "code_gen", "client": "aider",
        "chosen": "openai/glm-5.3-flash", "est_cost_hkd": 0.01, "task": "crud-endpoint"})
    parts = usage_hook.extract_receipt_parts(kwargs, FakeResponse(FakeUsage(100, 20)))
    assert parts["task"] == "crud-endpoint"


def test_hook_extract_task_absent_is_none():
    kwargs = _kwargs(signals_policy={
        "session": "s4", "kind": "factual", "client": "cursor",
        "chosen": "openai/glm-5.3-flash", "est_cost_hkd": 0.001})
    parts = usage_hook.extract_receipt_parts(kwargs, FakeResponse(FakeUsage(50, 10)))
    assert parts["task"] is None


def test_hook_extract_falls_back_to_annotator_signals():
    """Escape-hatch groups (mimo-v2.6-pro) never build a `policy` dict; the
    routing plugin leaves only top-level annotator_* keys. The receipt must
    pick those up so a hand-picked model joins by_task / task_redo."""
    kwargs = _kwargs(model="openai/mimo-v2.6-pro", signals={
        "policy": "skipped-not-fleet-group",
        "annotator_session": "sess-dx",
        "annotator_task": "mimo-trial",
        "annotator_client": "cursor",
    })
    parts = usage_hook.extract_receipt_parts(kwargs, FakeResponse(FakeUsage(200, 30, 150)))
    assert parts["task"] == "mimo-trial"
    assert parts["session"] == "sess-dx"
    assert parts["client"] == "cursor"
    assert parts["kind"] is None
    # Fleet `policy` dict wins over a stale annotation.
    kwargs_both = _kwargs(
        signals_policy={"session": "sess-fleet", "task": "fleet-task",
                        "client": "aider", "kind": "code_gen"},
        signals={"annotator_session": "sess-stale", "annotator_task": "stale"})
    parts = usage_hook.extract_receipt_parts(kwargs_both, FakeResponse(FakeUsage(1, 1)))
    assert parts["task"] == "fleet-task" and parts["session"] == "sess-fleet"


def test_duration_seconds_derivation():
    assert usage_hook._duration_seconds(100.0, 102.5) == 2.5
    # datetimes and junk both tolerated
    import datetime as _dt
    start = _dt.datetime(2026, 1, 1, 12, 0, 0)
    end = _dt.datetime(2026, 1, 1, 12, 0, 3)
    assert usage_hook._duration_seconds(start, end) == 3.0
    assert usage_hook._duration_seconds(None, None) is None
    assert usage_hook._duration_seconds("a", "b") is None
    assert usage_hook._duration_seconds(10.0, 5.0) is None   # negative
    assert usage_hook._duration_seconds(0.0, 10 ** 7) is None  # absurd


def test_record_receipt_persists_duration():
    reset()
    session = "sess-duration"
    usage_state.record_receipt(
        session=session, model="openai/mimo-v2.6-pro", kind=None, client="cursor",
        prompt_tokens=1000, completion_tokens=50, cached_tokens=0,
        duration_s=1.23456, est_cost_hkd=None)
    lines = open(usage_state.USAGE_LOG_PATH, encoding="utf-8").read().splitlines()
    record = json.loads(lines[-1])
    assert record["duration_s"] == 1.235
    # Absent timing writes null, not a fabricated zero (latency averages
    # must count only timed receipts).
    usage_state.record_receipt(
        session=session, model="openai/mimo-v2.6-pro", kind=None, client="cursor",
        prompt_tokens=10, completion_tokens=1)
    assert json.loads(open(usage_state.USAGE_LOG_PATH, encoding="utf-8")
                      .read().splitlines()[-1])["duration_s"] is None
    # Negative/garbage durations clamp to null too.
    usage_state.record_receipt(
        session=session, model="openai/mimo-v2.6-pro", kind=None, client="cursor",
        prompt_tokens=10, completion_tokens=1, duration_s=-3)
    assert json.loads(open(usage_state.USAGE_LOG_PATH, encoding="utf-8")
                      .read().splitlines()[-1])["duration_s"] is None


def test_recorder_end_to_end_records_duration():
    reset()
    hook = usage_hook.UsageRecorder()
    response = type("R", (), {"usage": {"prompt_tokens": 100, "completion_tokens": 5}})()
    asyncio.run(hook.async_log_success_event(
        _kwargs({"session": "sess-lat", "kind": "explain", "client": "other"}),
        response, 1000.0, 1003.25))
    lines = open(usage_state.USAGE_LOG_PATH, encoding="utf-8").read().splitlines()
    assert json.loads(lines[-1])["duration_s"] == 3.25


def test_escape_hatch_annotator_attaches_task_and_session():
    """mimo-v2.6-pro (single-deployment group) skips routing but the receipt
    must still carry task/session/client so the task comparison works."""
    reset()
    context = Context(ask("do the thing [[task:mimo-trial]]"),
                      candidates=["openai/mimo-v2.6-pro"])
    run(context)
    signals = context.signals
    assert signals["policy"] == "skipped-not-fleet-group"
    assert signals["annotator_task"] == "mimo-trial"
    assert isinstance(signals.get("annotator_session"), str)
    assert signals["annotator_client"] == "unknown"  # no headers in the test Context

    # Sticky across turns of the same session; [[task:]] clears.
    second = Context(ask("continue the thing"),
                     candidates=["openai/mimo-v2.6-pro"])
    run(second)
    assert second.signals["annotator_task"] == "mimo-trial"
    third = Context(ask("done [[task:]]"), candidates=["openai/mimo-v2.6-pro"])
    run(third)
    assert third.signals.get("annotator_task") is None
    # And the sticky map forgets it after the clear.
    fourth = Context(ask("more work"), candidates=["openai/mimo-v2.6-pro"])
    run(fourth)
    assert fourth.signals.get("annotator_task") is None


def test_escape_hatch_annotator_client_from_headers():
    """Real Cursor calls arrive with nginx-stamped X-Client-Type; the
    annotator must attribute them like the fleet path does."""
    reset()
    context = Context(ask("do the thing [[task:mimo-trial]]"),
                      candidates=["openai/mimo-v2.6-pro"],
                      metadata={"headers": {"X-Client-Type": "cursor"}})
    run(context)
    assert context.signals["annotator_client"] == "cursor"


def test_escape_hatch_annotator_never_touches_routing():
    """Candidate list is untouched and no policy dict is fabricated."""
    reset()
    candidates = ["openai/mimo-v2.6-pro"]
    context = Context(ask("anything [[task:mimo-trial]]"), candidates=list(candidates))
    run(context)
    assert context.candidate_models == candidates
    assert isinstance(context.signals["policy"], str)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                reset()
                fn()
                print(f"PASS {name}")
            except Exception as error:  # noqa: BLE001
                failures += 1
                import traceback
                print(f"FAIL {name}: {error}")
                traceback.print_exc()
    raise SystemExit(1 if failures else 0)
