"""B1 golden-replay harness: locks "today's routing" as a replayable corpus.

The review's B1 fix requires a mode matrix: decisions under learning-off,
control-file-absent, and control-disabled must be byte-identical to the
baseline. Phase A builds the harness and locks the baseline over the built
control surface; phase B (learner) only adds cases to the same corpus.

Run from the llm-router directory:

    python -X utf8 tests/test_golden.py           # assert against the golden
    python -X utf8 tests/test_golden.py --update  # regenerate deliberately

The golden records the FULL decision tuple per replay step (chosen, reason,
gate, kind, scale, bar, stall, directives) for every case in three modes:

  baseline        - the shipped defaults, no control file, learning on
  no-control-file - CONTROL points at a directory with no control.json
  learning-off    - control.json sets learning_enabled=false
  policy-learn-0  - POLICY_LEARN=0 (the hard kill-switch)

Phase A's corpus covers the degraded modes that apply to built code: the
control file (absent / learning-disabled) and the env kill-switch. Phase B
extends the mode matrix with the learned-state modes (review B1):

  no-learned-file    - learning on, learned.json absent (pristine)
  corrupt-learned    - learning on, learned.json torn on disk
  below-gate-learned - learning on, learned state present but every row
                       sits under the 8-observation evidence gate
  learn-0-with-state - POLICY_LEARN=0 with learned state present

All four must replay byte-identical to baseline. One extra counterfactual
case, learned-bias-changes-decision, proves an above-gate learned bias DOES
change a decision when learning is enabled.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["POLICY_LOG"] = ""  # never write a decision log from the harness
# The baseline replay must be independent of any leftover learned.json on this
# machine (the module default POLICY_STATE_DIR is /app/logs, which on Windows
# resolves to C:\app\logs): point state at a throwaway dir before import.
_tmp_state = tempfile.mkdtemp(prefix="router-golden-state-")
os.environ["POLICY_STATE_DIR"] = _tmp_state
os.environ["LEARNED_PATH"] = os.path.join(_tmp_state, "learned.json")
os.environ["CONTROL_PATH"] = os.path.join(_tmp_state, "control.json")

import routing_policy as policy  # noqa: E402
from control_store import ControlStore  # noqa: E402

GOLDEN_PATH = os.path.join(ROOT, "tests", "golden_routing.json")

MODES = ("baseline", "no-control-file", "learning-off", "policy-learn-0",
         "no-learned-file", "corrupt-learned", "below-gate-learned",
         "learn-0-with-state")
# The fail-open contract: every mode after the first must equal the baseline.
DEGRADED_MODES = MODES[1:]


def _user(text):
    return {"role": "user", "content": text}


def _tool_call(name, arguments):
    return {"role": "assistant",
            "tool_calls": [{"function": {"name": name, "arguments": arguments}}]}


def _conversation(turns, first="seed question"):
    """A multi-turn conversation: seed user turn, then one message per turn."""
    messages = [
        {"role": "system", "content": "You are Cursor."},
        {"role": "user", "content": first},
        {"role": "assistant", "content": "ok"},
    ]
    for turn in turns:
        messages.append(turn)
    return messages


def _ask_step(text, first="seed question"):
    return {"messages": _conversation([_user(text)], first=first)}


def corpus():
    """The fixed replay corpus. Never reorder or edit entries in place:
    append new cases at the end and regenerate the golden deliberately."""
    cases: list[dict] = []

    def case(name, steps):
        cases.append({"name": name, "steps": steps})

    def ask_step(text, first="seed question"):
        return _ask_step(text, first)

    def multi_step(texts, first="seed question"):
        # One conversation, several turns: only the last user message moves
        # forward; earlier asks stay in history (session key stays stable).
        messages = [
            {"role": "system", "content": "You are Cursor."},
            {"role": "user", "content": first},
            {"role": "assistant", "content": "ok"},
        ]
        step = {"messages": list(messages)}
        for text in texts:
            messages.append({"role": "user", "content": text})
            messages.append({"role": "assistant", "content": "ok"})
            step = {"messages": list(messages)[:-1]}
        return step

    # 1-3. canonical asks from the design's verification section
    case("debug-traceback", [ask_step("fix the failing test in report_parser.py")])
    case("bulk-rename", [ask_step("write a script that renames every file across the repo")])
    case("design-sync-layer", [ask_step("design the offline sync layer and its trade-offs")])

    # 4. image turn must never land on a text-only model
    image_messages = _conversation([{
        "role": "user",
        "content": [
            {"type": "text", "text": "what is wrong in this screenshot"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
        ],
    }])
    case("image-turn", [{"messages": image_messages}])

    # 5. oversized prompt drops small-window models
    big = _conversation([_user("summarize this")])
    big.insert(2, {"role": "user", "content": "y" * 600_000})
    case("oversized-prompt", [{"messages": big}])

    # 6. healthy TDD loop (M7 regression guard inside the golden too)
    tdd = _conversation([
        _user("keep going"),
        _tool_call("run_terminal_cmd", "{\"command\":\"python -m pytest tests/test_parser.py\"}"),
        _tool_call("str_replace_editor", "{\"path\":\"src/report_parser.py\",\"command\":\"str_replace\"}"),
        _tool_call("run_terminal_cmd", "{\"command\":\"python -m pytest tests/test_parser.py -k date\"}"),
        _tool_call("str_replace_editor", "{\"path\":\"src/report_parser.py\",\"command\":\"str_replace\",\"old\":\"x\",\"new\":\"y\"}"),
        _user("keep going"),
    ])
    case("tdd-loop-healthy", [{"messages": tdd}])

    # 7. a real stall: the last three calls identical, nothing between
    stalled = _conversation([
        _user("keep going"),
        _tool_call("str_replace_editor", "{\"path\":\"a.py\"}"),
        _tool_call("run_terminal_cmd", "{\"command\":\"npm test\"}"),
        _tool_call("run_terminal_cmd", "{\"command\":\"npm test\"}"),
        _tool_call("run_terminal_cmd", "{\"command\":\"npm test\"}"),
        _user("keep going"),
    ])
    case("stalled-loop", [{"messages": stalled}])

    # 8. ESCALATE word raises the effective bar
    case("escalate-word", [ask_step("design the sync layer for offline mode ESCALATE")])

    # 9. [[cheap]] directive
    case("cheap-directive", [ask_step("fix the crash in the parser [[cheap]]")])

    # 10. [[use:...]] pins an exact model
    case("use-directive", [ask_step("quick one [[use:glm-5.3-flash]]")])

    # 11. reminder-only final turn falls through (no ask)
    reminder = [
        {"role": "system", "content": "You are Cursor."},
        {"role": "user", "content": "<system-reminder>tool output</system-reminder>"},
    ]
    case("reminder-only", [{"messages": reminder}])

    # 12. multi-turn stickiness sequence (same task, three turns)
    case("stickiness-sequence", [
        multi_step(["fix the failing test in report_parser.py"]),
        multi_step(["fix the failing test in report_parser.py",
                    "fix the failing test in report_parser.py"]),
    ])

    # 13. second session interleaved (different opener -> different session)
    case("second-session", [
        ask_step("fix the failing test in report_parser.py"),
        ask_step("write a script that renames every file across the repo",
                 first="a completely different opening ask"),
    ])

    # 14. direct escape-hatch group skips the policy untouched
    case("escape-hatch-group", [
        {"messages": _conversation([_user("fix the failing test")]),
         "candidates": ["openai/glm-5.3"]},
    ])

    # 15. factual lookup classifies and routes cheap
    case("factual-lookup", [ask_step("what is the difference between a tuple and a list")])

    # 16. importance: a trivial follow-up in a LONG thread (premium model
    # pinned by stickiness) must break to a cheap flash model.
    long_thread = [
        {"role": "system", "content": "You are Cursor."},
        {"role": "user", "content": "refactor the whole parsing pipeline and fix the design"},
        {"role": "assistant", "content": "ok"},
    ]
    for _ in range(12):
        long_thread.append({"role": "user", "content": "continue the refactor"})
        long_thread.append({"role": "assistant", "content": "ok"})
    long_thread.append({"role": "user", "content": "thanks, looks good"})
    case("low-importance-followup", [
        multi_step(["refactor the whole parsing pipeline and fix the design"]),
        {"messages": list(long_thread)},
    ])

    # 17. importance: explicit [[low]] drops the bar even for a code ask
    case("low-directive", [ask_step("fix the typo in the readme [[low]]")])

    # 18. importance: spec phrasing goes High and stays premium
    case("high-spec-ask", [ask_step(
        "architect the new offline sync layer: write the specification, "
        "evaluate the trade-offs between crdt and event sourcing, and plan "
        "the migration step by step"
    )])

    return cases


def _run_step(step):
    """Drive one replay step through a fresh CursorAutoPolicy instance."""
    context = type("Ctx", (), {})()
    context.raw_messages = list(step["messages"])
    context.structured_messages = list(step["messages"])
    context.candidate_models = list(step.get("candidates", policy.PROFILES))
    context.metadata = {}
    context.signals = {}
    asyncio.run(policy.CursorAutoPolicy().run(context))
    decision = context.signals.get("policy")
    if not isinstance(decision, dict):
        return {"signal": decision if isinstance(decision, str) else None}
    return {
        "chosen": decision.get("chosen"),
        "reason": decision.get("reason"),
        "gate": decision.get("gate"),
        "kind": decision.get("kind"),
        "scale": decision.get("scale"),
        "importance": decision.get("importance"),
        "bar": decision.get("bar"),
        "stall": decision.get("stall"),
        "directives": decision.get("directives"),
        "cascade": decision.get("cascade"),
        "cascade_fallback": decision.get("cascade_fallback"),
        "laya_weight": decision.get("laya_weight"),
        "laya_scores": decision.get("laya_scores"),
        "candidates": context.candidate_models,
    }


def _reset_state():
    policy.STATE = policy.PolicyState()
    policy.CONTROL = None
    policy.LEARN_ENABLED = True
    policy.LEARNER = None
    policy._LAST_CURSOR_AUTO_SESSION.update({"key": None, "ask": ""})
    policy.OBSERVATIONS.update({"queue_depth": 0, "errors": 0, "extractions": 0})


def _learned_row(state, kind, model, value, count):
    """Force an above- or below-gate learned row directly into RAM state."""
    state.trust.rows[(kind, model)] = {"count": float(count), "value": float(value),
                                       "last_ts": __import__("time").time()}


def replay_all():
    """Replay the corpus in every mode; return {mode: {case: [steps]}}."""
    import tempfile
    import time as _time

    results: dict[str, dict] = {}
    for mode in MODES:
        _reset_state()
        control_dir = None
        if mode != "baseline":
            control_dir = tempfile.mkdtemp(prefix="router-golden-")
            store = ControlStore(os.path.join(control_dir, "control.json"),
                                 os.path.join(control_dir, "learned.json"))
            policy.CONTROL = store
            if mode == "learning-off":
                # Write learning_enabled=false through the store's own file
                # path, then force the store to see it.
                with open(os.path.join(control_dir, "control.json"), "w", encoding="utf-8") as handle:
                    json.dump({"learning_enabled": False}, handle)
                store._control_stamp = None  # force a re-read on next access
        if mode in ("policy-learn-0", "learn-0-with-state"):
            policy.LEARN_ENABLED = False

        # Learned-state modes: shape learned.json (and the store's view of it)
        # before the replay runs.
        if mode in ("no-learned-file", "corrupt-learned", "below-gate-learned",
                    "learn-0-with-state"):
            learned_path = os.path.join(control_dir, "learned.json")
            os.environ["POLICY_STATE_DIR"] = control_dir
            os.environ["LEARNED_PATH"] = learned_path
            os.environ["CONTROL_PATH"] = os.path.join(control_dir, "control.json")
            if mode == "corrupt-learned":
                with open(learned_path, "w", encoding="utf-8") as handle:
                    handle.write("{ torn learned state")
            elif mode in ("below-gate-learned", "learn-0-with-state"):
                # Present but BELOW the 8-observation evidence gate: must
                # apply at zero strength. (learn-0-with-state additionally
                # disables learning; the file must have zero effect either way.)
                import learning as _learning
                state = _learning.LearnedState()
                _learned_row(state, "debug", "openai/kimi-k2.7-code", -0.40, 4)
                _learning.save_learned(learned_path, state)
                import os as _os
                _os.utime(learned_path, (_time.time() + 5, _time.time() + 5))
            elif mode == "no-learned-file":
                pass  # nothing on disk: the pristine case

        mode_results: dict = {}
        for entry in corpus():
            steps = []
            for step in entry["steps"]:
                steps.append(_run_step(step))
            mode_results[entry["name"]] = steps
        results[mode] = mode_results

        # The counterfactual case: learning ENABLED, an above-gate learned
        # bias present -> the decision must actually move.
        if mode == "learned-bias-probe":
            pass  # handled separately after replay_all (see main)

        if control_dir:
            import shutil
            shutil.rmtree(control_dir, ignore_errors=True)
        for key in ("POLICY_STATE_DIR", "LEARNED_PATH", "CONTROL_PATH"):
            os.environ.pop(key, None)

    # The B1 fail-open property itself, asserted inside the harness: the
    # degraded modes must equal the baseline for every case and step.
    drift = []
    for mode in DEGRADED_MODES:
        if results[mode] != results["baseline"]:
            for case_name in results["baseline"]:
                if results[mode].get(case_name) != results["baseline"][case_name]:
                    drift.append(f"{mode}/{case_name}")
    return results, drift


def learned_bias_changes_decision() -> bool:
    """The counterfactual proving the learned tables are LIVE: with learning
    enabled and an above-gate learned state present, a decision must differ
    from the same request served with pristine learned state. Planted max
    negative learned trust on qwen3-vl-flash for bulk work (-0.40 effective,
    20 observed, above the 8-observation gate) docks 0.30*0.40 of utility and
    hands the bulk ask to the runner-up qwen3.8-flash; the baseline picks
    qwen3-vl-flash."""
    import tempfile
    import time
    import learning as _learning
    from control_store import ControlStore

    control_dir = tempfile.mkdtemp(prefix="router-golden-probe-")
    try:
        learned_path = os.path.join(control_dir, "learned.json")
        os.environ["POLICY_STATE_DIR"] = control_dir
        os.environ["LEARNED_PATH"] = learned_path
        os.environ["CONTROL_PATH"] = os.path.join(control_dir, "control.json")

        def run_once():
            # A FRESH store per run (post-write) sees the learned file; the
            # learner is left unbuilt so no flush can overwrite the planted
            # state.
            policy.CONTROL = ControlStore(os.path.join(control_dir, "control.json"), learned_path)
            policy.LEARNER = None
            policy.LEARN_ENABLED = True
            policy.STATE = policy.PolicyState()
            return _run_step(_ask_step("write a script that renames every file across the repo"))

        baseline_result = run_once()
        state = _learning.LearnedState()
        # Above-gate evidence (20 >= 8), clamped to -TRUST_PERSIST_MAX.
        state.trust.rows[("bulk", "openai/qwen3-vl-flash")] = {
            "count": 20.0, "value": -1.0, "last_ts": time.time(),
        }
        _learning.save_learned(learned_path, state)
        learned_result = run_once()
        # On the current fleet the bulk runner-up after a max negative trust
        # on qwen3-vl-flash is qwen3.8-flash (next-best bulk cap per dollar).
        return (baseline_result.get("chosen") == "openai/qwen3-vl-flash"
                and learned_result.get("chosen") == "openai/qwen3.8-flash")
    finally:
        import shutil
        shutil.rmtree(control_dir, ignore_errors=True)
        for key in ("POLICY_STATE_DIR", "LEARNED_PATH", "CONTROL_PATH"):
            os.environ.pop(key, None)
        policy.CONTROL = None
        policy.LEARNER = None
        policy.LEARN_ENABLED = True


def main() -> int:
    if "--update" in sys.argv:
        results, drift = replay_all()
        if drift:
            print("REFUSING to update: degraded modes already drift from baseline:")
            for item in drift:
                print(f"  - {item}")
            return 1
        payload = {"_comment": "Generated by tests/test_golden.py --update. "
                               "Regenerate only deliberately (phase gates).",
                   "modes": results}
        with open(GOLDEN_PATH, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1, sort_keys=True)
        print(f"golden written: {GOLDEN_PATH}")
        return 0

    with open(GOLDEN_PATH, encoding="utf-8") as handle:
        golden = json.load(handle)
    results, drift = replay_all()
    failures = []
    for mode in MODES:
        if results[mode] != golden["modes"].get(mode):
            for case_name in golden["modes"].get(mode, {}):
                if results[mode].get(case_name) != golden["modes"][mode][case_name]:
                    failures.append(f"{mode}/{case_name}")
    if drift:
        failures.extend(f"degraded-mode drift: {item}" for item in drift)
    # The counterfactual: learning ENABLED with an above-gate learned bias
    # must actually move a decision. Without this, byte-identity in the
    # degraded modes could equally prove the learned tables are dead code.
    if not learned_bias_changes_decision():
        failures.append("learned-bias probe: above-gate learned bias did NOT change the decision")
    if failures:
        print("FAIL golden replay")
        for item in failures:
            print(f"  - {item}")
        return 1
    total = sum(len(steps) for mode in MODES for steps in results[mode].values())
    print(f"golden replay: all green ({len(corpus())} cases x {len(MODES)} modes, {total} decisions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
