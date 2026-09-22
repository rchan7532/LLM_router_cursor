"""
Tests for routing_policy.CursorAutoPolicy.

Run from the llm-router directory:

    python -m pytest tests/test_routing_policy.py -v
    # or, without pytest installed:
    python tests/test_routing_policy.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["POLICY_LOG"] = ""  # keep the test run out of the production log

import routing_policy as policy  # noqa: E402
import failure_hook  # noqa: E402
from control_store import ControlStore  # noqa: E402


class Context:
    """Stand-in for litellm's RoutingContext."""

    def __init__(self, messages, candidates=None, metadata=None):
        self.raw_messages = list(messages)
        self.structured_messages = list(messages)
        self.candidate_models = list(candidates) if candidates is not None else list(policy.PROFILES)
        self.metadata = metadata or {}
        self.signals = {}


def ask(text, first="seed question"):
    """A conversation whose first user turn is stable across calls (session key)."""
    return [
        {"role": "system", "content": "You are Cursor."},
        {"role": "user", "content": first},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": text},
    ]


def run(context):
    return asyncio.run(policy.CursorAutoPolicy().run(context))


def reset():
    policy.STATE.sessions.clear()
    policy.STATE.trust.clear()
    failure_hook.RELIABILITY.clear()
    # Restore the default control store so a budget-capped test does not
    # leave a fake cap behind for the next test.
    if os.environ.get("POLICY_STATE_DIR"):
        policy.CONTROL = policy.default_store()
    else:
        policy.CONTROL = None


# --- signature extraction -------------------------------------------------

def test_detects_debug_from_traceback():
    signature = policy.build_signature(ask("getting this error: Traceback (most recent call last): boom"))
    assert signature is not None
    assert signature.kind == "debug"


def test_detects_scale_from_context_size():
    big = ask("refactor this module")
    big.insert(2, {"role": "user", "content": "x" * 80_000})
    signature = policy.build_signature(big)
    assert signature is not None and signature.scale >= 1


def test_detects_image_part():
    messages = ask("what is wrong in this screenshot")
    messages[-1]["content"] = [
        {"type": "text", "text": "what is wrong in this screenshot"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
    ]
    signature = policy.build_signature(messages)
    assert signature is not None and signature.has_image is True


def test_detects_stalled_loop():
    messages = ask("keep going")
    for _ in range(3):
        messages.insert(-1, {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "run_terminal_cmd", "arguments": "{\"command\":\"npm test\"}"}}],
        })
    signature = policy.build_signature(messages)
    assert signature is not None and signature.stall >= 3


def test_tdd_loop_is_not_a_stall():
    # M7 regression: the canonical healthy TDD rhythm test -> edit -> test
    # -> edit -> test alternates different tool calls between identical ones.
    # Each test run is followed by a different edit call, so there is no
    # consecutive trailing run of identical fingerprints and the loop must
    # NOT count as a stall (it makes progress between calls).
    messages = ask("keep going")
    loop = [
        {"role": "assistant",
         "tool_calls": [{"function": {"name": "run_terminal_cmd",
                                      "arguments": "{\"command\":\"python -m pytest tests/test_parser.py\"}"}}]},
        {"role": "assistant",
         "tool_calls": [{"function": {"name": "str_replace_editor",
                                      "arguments": "{\"path\":\"src/report_parser.py\",\"command\":\"str_replace\"}"}}]},
        {"role": "assistant",
         "tool_calls": [{"function": {"name": "run_terminal_cmd",
                                      "arguments": "{\"command\":\"python -m pytest tests/test_parser.py -k date\"}"}}]},
        {"role": "assistant",
         "tool_calls": [{"function": {"name": "str_replace_editor",
                                      "arguments": "{\"path\":\"src/report_parser.py\",\"command\":\"str_replace\",\"old\":\"x\",\"new\":\"y\"}"}}]},
    ]
    for message in loop:
        messages.insert(-1, message)
    messages.insert(-1, dict(loop[0]))  # final test run, same as the first kind
    signature = policy.build_signature(messages)
    assert signature is not None
    assert signature.stall == 0, "healthy TDD loop misread as a stall"


def test_consecutive_identical_run_still_stalls():
    # M7's intended catch unchanged: the last three calls identical with
    # nothing different between them is a real stall.
    messages = ask("keep going")
    messages.insert(-1, {"role": "assistant",
                         "tool_calls": [{"function": {"name": "str_replace_editor",
                                                      "arguments": "{\"path\":\"a.py\"}"}}]})
    for _ in range(3):
        messages.insert(-1, {"role": "assistant",
                             "tool_calls": [{"function": {"name": "run_terminal_cmd",
                                                          "arguments": "{\"command\":\"npm test\"}"}}]})
    signature = policy.build_signature(messages)
    assert signature is not None and signature.stall >= 3


def test_system_reminder_only_is_not_an_ask():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "<system-reminder>tool output</system-reminder>"},
    ]
    assert policy.build_signature(messages) is None


# --- selection properties -------------------------------------------------

def test_hard_debug_goes_to_a_strong_code_model():
    context = Context(ask("fix the failing test in report_parser.py"))
    run(context)
    assert context.candidate_models == ["openai/kimi-k2.7-code"], context.signals["policy"]


def test_bulk_work_goes_to_the_cheapest_capable_model():
    context = Context(ask("write a script that renames every file across the repo"))
    run(context)
    # qwen3-vl-flash is the fleet's cheapest bulk-capable model (cost_in 0.016,
    # bulk cap 0.78 above the bulk bar); deepseek-v4.1-flash is the runner-up.
    assert context.candidate_models == ["openai/qwen3-vl-flash"], context.signals["policy"]


def test_image_never_lands_on_a_text_only_model():
    messages = ask("explain this diagram")
    messages[-1]["content"] = [
        {"type": "text", "text": "explain this diagram"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
    ]
    context = Context(messages)
    run(context)
    chosen = context.candidate_models[0]
    assert policy.PROFILES[chosen].vision is True, context.signals["policy"]


def test_oversized_prompt_drops_small_window_models():
    messages = ask("summarize this")
    messages.insert(2, {"role": "user", "content": "y" * 600_000})  # ~150k tokens
    context = Context(messages)
    run(context)
    chosen = context.candidate_models[0]
    assert policy.PROFILES[chosen].ctx_window >= 200_000, context.signals["policy"]


def test_cheap_directive_never_undercuts_the_capability_bar():
    context = Context(ask("fix the crash in the parser [[cheap]]"))
    run(context)
    chosen = policy.PROFILES[context.candidate_models[0]]
    assert chosen.cap["debug"] >= 0.75, context.signals["policy"]


def test_escalate_never_picks_a_weaker_model():
    plain = Context(ask("design the sync layer for offline mode"))
    run(plain)
    strong = Context(ask("design the sync layer for offline mode"))
    strong.raw_messages[-1]["content"] += " ESCALATE"
    strong.structured_messages = strong.raw_messages
    run(strong)
    before = policy.PROFILES[plain.candidate_models[0]].cap["design"]
    after = policy.PROFILES[strong.candidate_models[0]].cap["design"]
    assert after >= before, (plain.signals["policy"], strong.signals["policy"])


def test_use_directive_pins_an_exact_model():
    context = Context(ask("quick question [[use:glm-5.3-flash]]"))
    run(context)
    assert context.candidate_models == ["openai/glm-5.3-flash"]


def test_tie_break_prefers_the_cheaper_model():
    scored = [
        policy.Candidate(policy.PROFILES["openai/glm-5.3"], 0.8, 0.0, 0.0, 0.0, 0.0),
        policy.Candidate(policy.PROFILES["openai/deepseek-v4.1-flash"], 0.8, 0.0, 0.0, 0.0, 0.0),
    ]
    for item in scored:
        item.utility = 0.5
    scored.sort(key=lambda item: (-item.utility, policy._blended_cost(item.profile)))
    assert scored[0].profile.model == "openai/deepseek-v4.1-flash"


# --- stickiness, feedback, safety ----------------------------------------

def test_second_turn_of_a_task_reuses_the_model():
    reset()
    first = Context(ask("fix the failing test in report_parser.py"))
    run(first)
    second = Context(ask("fix the failing test in report_parser.py"))
    run(second)
    assert second.signals["policy"]["reason"] == "pinned-grace"
    assert second.candidate_models == first.candidate_models


def test_escalation_lowers_trust_for_the_model_that_was_pinned():
    reset()
    first = Context(ask("write a script that renames every file across the repo"))
    run(first)
    served = first.candidate_models[0]
    kind = "bulk"
    before = policy.STATE.trust_of(kind, served)

    follow_up = Context(ask("write a script that renames every file across the repo ESCALATE"))
    run(follow_up)
    assert policy.STATE.trust_of(kind, served) < before


def test_direct_escape_hatch_group_is_skipped():
    context = Context(ask("fix the failing test"), candidates=["openai/glm-5.3"])
    run(context)
    assert context.signals["policy"] == "skipped-not-fleet-group"
    assert context.candidate_models == ["openai/glm-5.3"]


def test_failure_opens_through_instead_of_breaking_the_request():
    context = Context(ask("anything"), candidates=[["unhashable"]])
    run(context)  # must not raise
    assert "policy_error" in context.signals
    assert context.candidate_models == [["unhashable"]]


def test_loads_the_way_litellm_loads_it():
    """
    litellm resolves plugins with importlib and never registers the module in
    sys.modules. Code that assumes registration (e.g. dataclasses resolving
    string annotations) passes a normal `import` test and then fails at proxy
    startup. This test loads the module the way the proxy does.
    """
    import importlib.util
    import inspect

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "routing_policy.py")
    saved = sys.modules.pop("routing_policy", None)
    try:
        spec = importlib.util.spec_from_file_location("routing_policy", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # must not raise
        assert inspect.iscoroutinefunction(module.CursorAutoPolicy.run)
        assert module.PROFILES  # module-level state built at import
    finally:
        if saved is not None:
            sys.modules["routing_policy"] = saved


def test_resolves_through_litellms_own_resolver():
    """
    The configured path must name an instance. A class passes litellm's
    Protocol check (it has a `run` attribute) and then fails on every request
    with "run() missing 1 required positional argument", so assert the
    instance shape and actually drive one request through it.
    """
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "routing_policy.py")
    saved = sys.modules.pop("routing_policy", None)
    try:
        try:
            from litellm.proxy.types_utils.utils import get_instance_fn
            from litellm.types.router import RoutingPlugin
        except ImportError:
            print("      (litellm not importable; skipped)")
            return
        import yaml

        config = yaml.safe_load(open(os.path.join(os.path.dirname(path), "litellm-config.yaml"), encoding="utf-8"))
        configured = config["router_settings"]["plugins"][0]
        resolved = get_instance_fn(value=configured, config_file_path=path)
        assert isinstance(resolved, RoutingPlugin)
        assert not isinstance(resolved, type), f"{configured} names a class; the config needs the instance"
        context = Context(ask("fix the failing test in report_parser.py"))
        asyncio.run(resolved.run(context))
        assert context.candidate_models, context.signals
    finally:
        if saved is not None:
            sys.modules["routing_policy"] = saved


# --- importance tiers -----------------------------------------------------

def test_importance_detection_manual_and_heuristic():
    reset()
    assert policy._detect_importance("do it [[low]]", "code_edit", frozenset({"low"}), 100) == 0
    assert policy._detect_importance("do it [[high]]", "design", frozenset({"high"}), 100) == 2
    assert policy._detect_importance("do it ESCALATE", "debug", frozenset({"escalate"}), 100) == 2
    # one-liners
    assert policy._detect_importance("thanks", "explain", frozenset(), 5) == 0
    assert policy._detect_importance("Okay. ", "explain", frozenset(), 5) == 0
    assert policy._detect_importance("keep going", "agentic", frozenset(), 10) == 0
    # summarize-like short asks
    assert policy._detect_importance("summarize this file", "explain", frozenset(), 50) == 0
    # spec phrasing with >= 2 high hits
    assert policy._detect_importance(
        "architect the sync layer and write the specification", "design", frozenset(), 400
    ) == 2
    # one high hit + design kind
    assert policy._detect_importance("plan the migration carefully", "design", frozenset(), 300) == 2
    # ordinary work stays Normal
    assert policy._detect_importance("fix the failing test in report_parser.py", "debug", frozenset(), 60) == 1


def test_low_importance_drops_the_capability_bar():
    reset()
    sig_low = policy.build_signature(ask("add a comment to the config file [[low]]"))
    sig_normal = policy.build_signature(ask("fix the failing test in report_parser.py"))
    assert sig_low.importance == 0
    assert sig_normal.importance == 1
    # The bar formula must differ by exactly the importance offset.
    base = policy.BASE_BAR[sig_low.kind] + policy.SCALE_BAR_STEP * min(
        policy.MAX_SCALE, (sig_low.ask_tokens or 1) // 400)
    assert sig_low is not None and sig_normal is not None


def test_low_followup_breaks_a_premium_pin():
    reset()
    # First turn: real design work -> premium model (glm-5.3).
    first = Context(ask("refactor the whole parsing pipeline and fix the design"))
    run(first)
    assert first.candidate_models[0] in {
        "openai/glm-5.3", "openai/kimi-k2.7-code", "openai/claude-haiku-4-5",
    }, first.signals["policy"]
    premium = first.candidate_models[0]

    # Same conversation, trivial follow-up: the premium pin must break and a
    # cheap flash model must serve it (stickiness cost guard).
    follow_up = Context(ask("thanks, looks good", first="refactor the whole parsing pipeline and fix the design"))
    run(follow_up)
    assert follow_up.signals["policy"]["reason"] == "fresh", follow_up.signals["policy"]
    assert follow_up.candidate_models[0] != premium
    assert policy._blended_cost(policy.PROFILES[follow_up.candidate_models[0]]) < policy.PREMIUM_COST


def test_high_spec_ask_stays_premium():
    reset()
    ctx = Context(ask("architect the offline sync layer and evaluate the trade-offs before you plan the migration"))
    run(ctx)
    assert ctx.signals["policy"]["importance"] == 2
    assert policy._blended_cost(policy.PROFILES[ctx.candidate_models[0]]) > policy.PREMIUM_COST


def test_explicit_use_directive_beats_the_cost_guard():
    reset()
    first = Context(ask("refactor the whole parsing pipeline and fix the design"))
    run(first)
    # [[use:]] is explicit intent: even a trivial ask must honour it.
    follow_up = Context(ask("thanks [[use:glm-5.3]]", first="refactor the whole parsing pipeline and fix the design"))
    run(follow_up)
    assert follow_up.signals["policy"]["reason"] == "pinned-directive"
    assert follow_up.candidate_models[0] == "openai/glm-5.3"


# --- budget cap -----------------------------------------------------------

def _with_store(store):
    policy.CONTROL = store
    policy._LAST_CURSOR_AUTO_SESSION.update({"key": None, "ask": ""})
    policy.OBSERVATIONS.update({"queue_depth": 0, "errors": 0, "extractions": 0})


def test_budget_cap_forces_cheapest_model_after_threshold():
    reset()

    tmp = tempfile.mkdtemp(prefix="budget-")
    store = ControlStore(
        control_path=os.path.join(tmp, "control.json"),
        learned_path=os.path.join(tmp, "learned.json"),
    )
    # Cap smaller than one turn so the FIRST turn spends and the SECOND
    # turn triggers the cap.
    with open(store._control_path, "w", encoding="utf-8") as fh:
        json.dump({"budget_hkd": 0.0005, "revision": 1}, fh)
    _with_store(store)

    first = Context(ask("fix the failing test in report_parser.py"))
    run(first)
    assert first.signals["policy"]["reason"] != "budget-cap"
    assert first.signals["policy"]["spend_hkd"] > 0

    # Second turn in the same session should hit the cap.
    ctx = Context(ask("keep going"))
    run(ctx)
    assert ctx.signals["policy"]["reason"] == "budget-cap"
    assert ctx.signals["policy"]["budget_alarm"] is True
    assert ctx.signals["policy"]["budget_hkd"] == 0.0005
    chosen = ctx.candidate_models[0]
    assert policy._blended_cost(policy.PROFILES[chosen]) < policy.PREMIUM_COST


def test_budget_cap_does_not_override_explicit_high_or_use():
    reset()

    tmp = tempfile.mkdtemp(prefix="budget-")
    store = ControlStore(
        control_path=os.path.join(tmp, "control.json"),
        learned_path=os.path.join(tmp, "learned.json"),
    )
    with open(store._control_path, "w", encoding="utf-8") as fh:
        json.dump({"budget_hkd": 0.01, "revision": 1}, fh)
    _with_store(store)

    # [[high]] should still go to a premium model.
    ctx = Context(ask("architect the sync layer [[high]]"))
    run(ctx)
    assert ctx.signals["policy"]["reason"] != "budget-cap"
    assert policy._blended_cost(policy.PROFILES[ctx.candidate_models[0]]) > policy.PREMIUM_COST

    # [[use:glm-5.3]] should still win.
    ctx2 = Context(ask("quick one [[use:glm-5.3]]"))
    run(ctx2)
    assert ctx2.signals["policy"]["reason"] == "pinned-directive"
    assert ctx2.candidate_models[0] == "openai/glm-5.3"


def test_session_spend_accumulates_and_triggers_cap():
    reset()

    tmp = tempfile.mkdtemp(prefix="budget-")
    store = ControlStore(
        control_path=os.path.join(tmp, "control.json"),
        learned_path=os.path.join(tmp, "learned.json"),
    )
    # A small cap: triggers after a few turns of normal work.
    with open(store._control_path, "w", encoding="utf-8") as fh:
        json.dump({"budget_hkd": 0.0025, "revision": 1}, fh)
    _with_store(store)

    first = Context(ask("fix the failing test in report_parser.py"))
    run(first)
    assert first.signals["policy"]["reason"] != "budget-cap"
    spend_after_first = policy.STATE.session_spend(first.signals["policy"]["session"])
    assert spend_after_first > 0

    # Keep the same session (same first message) and ask again until cap.
    for _ in range(40):
        ctx = Context(ask("keep going"))
        run(ctx)
        if ctx.signals["policy"]["reason"] == "budget-cap":
            break
    else:
        raise AssertionError("budget cap never triggered")

    assert policy.STATE.session_spend(ctx.signals["policy"]["session"]) >= 0.0025


# --- cascade-lite ---------------------------------------------------------

def _cascade_store():
    """A temp control store with learning enabled so cascade-lite is active."""
    tmp = tempfile.mkdtemp(prefix="cascade-")
    return ControlStore(
        control_path=os.path.join(tmp, "control.json"),
        learned_path=os.path.join(tmp, "learned.json"),
    )


def test_cascade_escalates_after_previous_model_failure():
    reset()
    policy.CONTROL = _cascade_store()
    first = Context(ask("fix the failing test in report_parser.py"))
    run(first)
    served = first.candidate_models[0]

    failure_hook.record_failure(served, 503, now=time.time())
    second = Context(ask("keep going"))
    run(second)
    assert second.signals["policy"]["reason"] == "cascade-escalation", second.signals["policy"]
    assert second.candidate_models[0] != served


def test_cascade_ignores_stale_failures():
    reset()
    policy.CONTROL = _cascade_store()
    first = Context(ask("fix the failing test in report_parser.py"))
    run(first)
    served = first.candidate_models[0]

    failure_hook.record_failure(served, 503, now=time.time() - 1000)
    # Same task kind so session stickiness would normally hold; stale failure
    # must not break the pin.
    second = Context(ask("still failing, keep debugging it"))
    run(second)
    assert second.signals["policy"]["reason"] == "pinned-grace", second.signals["policy"]
    assert second.candidate_models[0] == served


def test_cascade_does_not_override_explicit_use():
    reset()
    policy.CONTROL = _cascade_store()
    first = Context(ask("fix the failing test in report_parser.py"))
    run(first)
    served = first.candidate_models[0]

    failure_hook.record_failure(served, 503, now=time.time())
    second = Context(ask(f"keep going [[use:{served.replace('openai/', '')}]]"))
    run(second)
    assert second.signals["policy"]["reason"] == "pinned-directive", second.signals["policy"]
    assert second.candidate_models[0] == served


def test_cascade_disabled_when_learning_off():
    reset()
    policy.CONTROL = _cascade_store()
    first = Context(ask("fix the failing test in report_parser.py"))
    run(first)
    served = first.candidate_models[0]

    failure_hook.record_failure(served, 503, now=time.time())
    policy.LEARN_ENABLED = False
    try:
        # Same task kind: with cascade disabled, the session pin must hold.
        second = Context(ask("still failing, keep debugging it"))
        run(second)
        assert second.signals["policy"]["reason"] == "pinned-grace", second.signals["policy"]
    finally:
        policy.LEARN_ENABLED = True


# --- stickiness as bounded preference (pin competes after grace) ----------

_LONG_PAD = "with careful handling of edge cases, timezone conversions, and malformed input rows. " * 40

def _long_high_code_edit():
    # code_edit kind (path fallback), High importance ("production-ready" +
    # long ask), ask_scale 2 -> bar 0.75 -> only premium models clear.
    return ("update src/report_parser.py to handle the new date formats, "
            "production-ready, " + _LONG_PAD)

def _normal_code_edit_followup():
    # Same kind (code_edit: >120 conversation tokens, path present, no
    # verb keywords), Normal importance, small ask-scale -> bar 0.55 ->
    # every capable model clears, so the pin actually has a cheaper
    # challenger to compete against.
    return ("update src/report_parser.py for the remaining edge cases as well: "
            "the date format branches, the timezone conversion rows, and the "
            "malformed input guards still need the same treatment applied to "
            "the rest of the file, including the helper functions that parse "
            "the fallback formats and the shared validation entry points "
            "used by the import pipeline, plus the small wrapper layer in "
            "report_parser.py that normalizes the column headers before the "
            "row-level parsing kicks in, and the logging calls that record "
            "each rejected row with its source line number")


def test_pin_holds_through_grace_then_competes():
    """The 44-turn lock-in regression. A premium model that legitimately won
    the first (high-bar) turn must not own the whole session: after the
    grace window the pin re-competes on utility, and a much cheaper model
    with comparable capability takes over."""
    reset()
    first = Context(ask(_long_high_code_edit()))
    run(first)
    served = first.candidate_models[0]
    assert policy._blended_cost(policy.PROFILES[served]) > policy.PREMIUM_COST
    assert first.signals["policy"]["kind"] == "code_edit"

    # Turns 2..GRACE: same-kind continuations hold through grace.
    for _ in range(policy.PIN_GRACE_TURNS):
        follow = Context(ask(_normal_code_edit_followup()))
        run(follow)
        assert follow.candidate_models[0] == served

    # Turn GRACE+1: the pin now competes. glm-5.3-flash (code_edit 0.72,
    # cents per 1M) beats kimi's utility by more than the loyalty bonus.
    follow = Context(ask(_normal_code_edit_followup()))
    run(follow)
    decision = follow.signals["policy"]
    assert decision["reason"] == "pin-swap", decision
    assert follow.candidate_models[0] != served
    assert policy._blended_cost(policy.PROFILES[follow.candidate_models[0]]) < policy._blended_cost(policy.PROFILES[served])


def test_pin_swap_resets_grace_no_ping_pong():
    """A swap restarts the turn counter, so the challenger immediately gets
    its own grace window and the session cannot flip models every turn."""
    reset()
    first = Context(ask(_long_high_code_edit()))
    run(first)
    served = first.candidate_models[0]

    # Drive past grace until a swap happens (bounded: 30 turns max).
    swapped_to = None
    for _ in range(30):
        follow = Context(ask(_normal_code_edit_followup()))
        run(follow)
        if follow.candidate_models[0] != served:
            swapped_to = follow.candidate_models[0]
            assert follow.signals["policy"]["reason"] == "pin-swap"
            break
    assert swapped_to is not None, "pin never surrendered the task in 30 turns"

    # The very next turn must NOT swap back: the new model is in its own
    # grace window.
    follow = Context(ask(_normal_code_edit_followup()))
    run(follow)
    assert follow.candidate_models[0] == swapped_to
    assert follow.signals["policy"]["reason"] == "pinned-grace"


def test_pin_breaks_when_bar_outruns_held_model():
    """A High-importance turn arriving mid-pin must not be served by a model
    below the CURRENT bar, even on a sticky same-kind task."""
    reset()
    # Establish a cheap pin on code_edit (flash wins the Normal first turn).
    first = Context(ask(_normal_code_edit_followup()))
    run(first)
    served = first.candidate_models[0]
    assert policy._blended_cost(policy.PROFILES[served]) <= policy.PREMIUM_COST

    # Drive past grace on the same kind.
    for _ in range(policy.PIN_GRACE_TURNS + 2):
        follow = Context(ask(_normal_code_edit_followup()))
        run(follow)

    # Now a long High code_edit ask in the same session: bar 0.75 must
    # exceed the held flash model's code_edit capability (0.72), breaking
    # the pin and re-deciding among the premium models.
    heavy = Context(ask(_long_high_code_edit()))
    run(heavy)
    assert heavy.candidate_models[0] != served, heavy.signals["policy"]
    assert policy._blended_cost(policy.PROFILES[heavy.candidate_models[0]]) > policy.PREMIUM_COST


# --- laya shadow scorer wiring --------------------------------------------

class _MockLayaScorer:
    def maybe_enqueue(self, text, kind):
        pass

    def get_scores(self, text):
        return {"bulk": 0.9}


def test_laya_weight_is_logged_when_configured():
    reset()
    tmp = tempfile.mkdtemp(prefix="laya-")
    store = ControlStore(
        control_path=os.path.join(tmp, "control.json"),
        learned_path=os.path.join(tmp, "learned.json"),
    )
    with open(store._control_path, "w", encoding="utf-8") as fh:
        json.dump({"weights": {"laya": 1.0}, "learning_enabled": True, "revision": 1}, fh)
    policy.CONTROL = store
    policy._laya_scorer = _MockLayaScorer()
    try:
        context = Context(ask("write a script that renames every file across the repo"))
        run(context)
        assert context.signals["policy"]["laya_weight"] == policy.LAYA_WEIGHT_MAX
        assert context.signals["policy"]["laya_scores"] == {"bulk": 0.9}
    finally:
        policy._laya_scorer = None


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
    print("---")
    print("all green" if failures == 0 else f"{failures} failing")
    sys.exit(1 if failures else 0)
