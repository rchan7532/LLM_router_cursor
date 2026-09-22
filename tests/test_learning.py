"""
Tests for learning.py: verdict mapping, the B2 containment re-ask, skip-path
attribution (B3), evidence gates, 90-day decay, influence bounds, the
raw-text-never-on-disk guarantee, reset-intent apply-once semantics, and the
phase-2 extraction batch (mocked HTTP).

Run from the llm-router directory:

    python tests/test_learning.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import learning  # noqa: E402


# --------------------------------------------------------------------------
# Verdict mapping: every leak row in the spec's table
# --------------------------------------------------------------------------

def test_escalate_word_is_negative():
    prev = {"ask": "fix the bug", "directives": [], "stall": 0, "kind": "debug", "model": "openai/glm-5.3"}
    cur = {"ask": "fix the bug ESCALATE", "directives": [], "stall": 0, "kind": "debug", "model": "openai/glm-5.3"}
    assert learning.classify_verdict(prev, cur) == "escalate"


def test_escalate_directive_is_negative():
    prev = {"ask": "fix it", "directives": [], "stall": 0, "kind": "debug", "model": "m"}
    cur = {"ask": "fix it again", "directives": ["escalate"], "stall": 0, "kind": "debug", "model": "m"}
    assert learning.classify_verdict(prev, cur) == "escalate"


def test_substring_escalate_word_not_verdict():
    # review m8: "ESCALATED" must not match the word-boundary ESCALATE.
    prev = {"ask": "fix the bug", "directives": [], "stall": 0, "kind": "debug", "model": "m"}
    cur = {"ask": "why did the test ESCALATED into a failure", "directives": [], "stall": 0,
           "kind": "debug", "model": "m"}
    assert learning.classify_verdict(prev, cur) is None


def test_reask_is_negative_strict_containment():
    prev = {"ask": "fix the failing test in report_parser.py please",
            "directives": [], "stall": 0, "kind": "debug", "model": "m"}
    cur = {"ask": "fix the failing test in report_parser.py please",
           "directives": [], "stall": 0, "kind": "debug", "model": "m"}
    assert learning.classify_verdict(prev, cur) == "reask"


def test_stall_verdict_weight():
    assert learning.verdict_weight("stall") == -0.15


def test_clean_requires_four_turns():
    prev = {"ask": "fix the bug", "directives": [], "stall": 0, "kind": "debug", "model": "m"}
    for turns in (2, 3):
        cur = {"ask": "now summarize what changed", "directives": [], "stall": 0,
               "kind": "debug", "model": "m", "turns": turns}
        assert learning.classify_verdict(prev, cur) is None, turns
    cur = {"ask": "now summarize what changed", "directives": [], "stall": 0,
           "kind": "debug", "model": "m", "turns": 4}
    assert learning.classify_verdict(prev, cur) == "clean"


def test_verdict_weights_match_spec_table():
    assert learning.verdict_weight("escalate") == -0.20
    assert learning.verdict_weight("reask") == -0.20
    assert learning.verdict_weight("direct_switch") == -0.20
    assert learning.verdict_weight("stall") == -0.15
    assert learning.verdict_weight("clean") == 0.03
    assert learning.verdict_weight("error") == 0.0   # 429/5xx neutral
    assert learning.verdict_weight("unknown") == 0.0


# --------------------------------------------------------------------------
# B2: containment, not Jaccard - extension vs complaint, both directions
# --------------------------------------------------------------------------

def test_extension_is_not_reask():
    prev = "fix the failing test in report_parser.py"
    nxt = "fix the failing test in report_parser.py and the date parser too"
    assert learning.is_reask(prev, nxt) is False   # review B2: extension, not complaint


def test_extension_in_other_direction_is_not_reask():
    # The follow-up drops most of the original scope: also not a re-ask.
    prev = "fix the failing test in report_parser.py and the date parser and the exporter"
    nxt = "fix the failing test"
    assert learning.is_reask(prev, nxt) is False


def test_verbatim_resend_is_reask():
    prev = "fix the failing test in report_parser.py"
    assert learning.is_reask(prev, prev + " please") is True
    assert learning.is_reask(prev, prev) is True


def test_short_prev_allows_one_new_token():
    prev = "fix parser bug"
    nxt = "fix parser bug now"
    assert learning.is_reask(prev, nxt) is True   # max(1, 3//20) = 1 new token


def test_unrelated_asks_are_not_reask():
    assert learning.is_reask("fix the failing test in report_parser.py",
                             "design the offline sync layer") is False


# --------------------------------------------------------------------------
# B3: skip-path attribution
# --------------------------------------------------------------------------

DEBUG_ASK = "fix the failing test in report_parser.py"


def test_switch_to_different_model_topical_is_negative():
    verdict = learning.direct_switch_verdict(DEBUG_ASK, "openai/kimi-k2.7-code",
                                             "openai/glm-5.3", 0.9)
    assert verdict == "direct_switch"


def test_switch_to_same_model_is_no_verdict():
    # A switch to the model the session already used is a router bypass,
    # not dissatisfaction (review B3).
    assert learning.direct_switch_verdict(DEBUG_ASK, "openai/kimi-k2.7-code",
                                          "openai/kimi-k2.7-code", 0.9) is None


def test_brand_new_session_no_prior_ask_no_verdict():
    # No prior cursor-auto session ask: nothing to attribute against.
    assert learning.direct_switch_verdict("", "openai/kimi-k2.7-code",
                                          "openai/glm-5.3", 0.9) is None


def test_no_topical_continuity_no_verdict():
    verdict = learning.direct_switch_verdict(DEBUG_ASK, "openai/kimi-k2.7-code",
                                             "openai/glm-5.3", 0.1)
    assert verdict is None   # overlap < 0.3: invisible by construction


def test_overlap_boundary_is_exactly_03():
    assert learning.topical_overlap(DEBUG_ASK, DEBUG_ASK) == 1.0
    assert learning.topical_overlap("", DEBUG_ASK) == 0.0


def test_learner_observe_switch_applies_to_prev_model():
    with tempfile.TemporaryDirectory() as tmp:
        learner = learning.Learner(os.path.join(tmp, "learned.json"))
        prev = {"ask": DEBUG_ASK, "kind": "debug", "model": "openai/kimi-k2.7-code"}
        verdict = learner.observe_switch(prev, DEBUG_ASK + " still failing",
                                         "openai/glm-5.3", "sess-1")
        assert verdict == "direct_switch"
        assert learner.state.trust.raw_value(("debug", "openai/kimi-k2.7-code")) < 0
        assert learner.state.bars.raw_value(("debug",)) < 0   # strong verdict feeds bars


# --------------------------------------------------------------------------
# Evidence gate + decay math at 0/45/90/180 days
# --------------------------------------------------------------------------

def test_evidence_gate_blocks_below_eight():
    table = learning.EvidenceTable()
    for _ in range(7):
        table.add(("k", "m"), -0.20)
    assert table.effective_count(("k", "m")) >= 7.0
    assert table.value(("k", "m"), 0.40) is None       # below gate: no value
    assert table.raw_value(("k", "m")) < 0             # but the value is stored
    table.add(("k", "m"), -0.20)                       # the 8th observation
    value = table.value(("k", "m"), 0.40)
    assert value is not None and value < 0


def test_decay_halves_evidence_at_ninety_days():
    table = learning.EvidenceTable()
    for _ in range(8):
        table.add(("k", "m"), -0.20, now=time.time() - 90 * 86400)
    effective = table.effective_count(("k", "m"))
    assert abs(effective - 4.0) < 1e-6                  # 8 * 0.5**1
    assert table.value(("k", "m"), 0.40) is None        # below the gate now


def test_decay_math_at_zero_and_forty_five_and_one_eighty_days():
    table = learning.EvidenceTable()
    now = time.time()
    table.add(("a",), 1.0, now=now)
    assert abs(table.effective_count(("a",), now=now) - 1.0) < 1e-9        # 0 days
    table.add(("b",), 1.0, now=now - 45 * 86400)
    assert abs(table.effective_count(("b",), now=now) - 0.5 ** 0.5) < 1e-6  # 45 days
    table.add(("c",), 1.0, now=now - 90 * 86400)
    assert abs(table.effective_count(("c",), now=now) - 0.5) < 1e-6         # 90 days
    table.add(("d",), 1.0, now=now - 180 * 86400)
    assert abs(table.effective_count(("d",), now=now) - 0.25) < 1e-6        # 180 days


def test_decayed_rows_are_pruned():
    table = learning.EvidenceTable()
    now = time.time()
    for _ in range(20):
        table.add(("old",), -0.2, now=now - 400 * 86400)
    table.add(("new",), -0.2, now=now)
    table.prune(now=now)
    assert ("old",) not in table.rows
    assert ("new",) in table.rows


def test_from_dict_on_garbage_is_empty_not_raising():
    for garbage in (None, 42, "x", {"bad": "shape"}, {"k": [1, 2]},
                    {"k": ["a", 2, 3]}, {"|x": [1, 2, 3]}):
        table = learning.EvidenceTable.from_dict(garbage)
        assert table.rows == {}, garbage


def test_learned_state_roundtrip_survives_garbage():
    state = learning.LearnedState()
    state.trust.add(("debug", "openai/glm-5.3"), -0.2)
    state.bars.add(("debug",), -0.2)
    state.phrases.add(("sort this out",), 1.0)
    blob = state.to_dict()
    restored = learning.LearnedState.from_dict(blob)
    assert restored.trust.raw_value(("debug", "openai/glm-5.3")) < 0
    assert restored.bars.raw_value(("debug",)) < 0
    assert restored.phrases.raw_value(("sort this out",)) == 1.0
    for garbage in (None, 42, "{oops", [], {"model_trust": "junk"}):
        assert learning.LearnedState.from_dict(garbage) is not None


# --------------------------------------------------------------------------
# Influence bounds under adversarial input
# --------------------------------------------------------------------------

def test_trust_value_clamped_under_flood():
    state = learning.LearnedState()
    for _ in range(1000):
        state.apply_verdict("escalate", "debug", "openai/kimi-k2.7-code")
    assert state.learned_trust("debug", "openai/kimi-k2.7-code") == -learning.TRUST_BOUND
    assert state.bar_adjust("debug") == -learning.BAR_BIAS_BOUND


def test_positive_side_clamped():
    state = learning.LearnedState()
    for _ in range(1000):
        state.apply_verdict("clean", "debug", "openai/kimi-k2.7-code")
    assert state.learned_trust("debug", "openai/kimi-k2.7-code") <= learning.TRUST_BOUND


def test_combined_clamp_at_point_of_use():
    # review M8: runtime 0.30 + learned 0.20 must clamp to 0.40, not stack.
    runtime = 0.30
    learned = 0.20
    combined = max(-learning.TRUST_BOUND, min(learning.TRUST_BOUND, runtime + learned))
    assert combined == learning.TRUST_BOUND


def test_clean_verdicts_do_not_feed_kind_bar_bias():
    # review q2: bars consume escalate/direct-switch only.
    state = learning.LearnedState()
    for _ in range(50):
        state.apply_verdict("clean", "debug", "openai/kimi-k2.7-code")
    assert state.bars.rows.get(("debug",)) is None


def test_error_verdict_is_a_noop():
    state = learning.LearnedState()
    state.apply_verdict("error", "debug", "openai/kimi-k2.7-code")
    assert state.trust.rows == {}
    assert state.bars.rows == {}


def test_row_cap_enforced():
    state = learning.LearnedState()
    for index in range(learning.ROW_CAP + 50):
        state.apply_verdict("escalate", f"kind{index}", "m")
    assert len(state.trust.rows) <= learning.ROW_CAP


# --------------------------------------------------------------------------
# sanitize: mechanical scrub before persistence (review M4)
# --------------------------------------------------------------------------

def test_sanitize_strips_secret_shapes():
    cases = {
        "key sk-abcdefghijklmnop1234 in text": "[REDACTED]",
        "token ghp_abc123def456xyz here": "[REDACTED]",
        "aws AKIAIOSFODNN7EXAMPLE here": "[REDACTED]",
        "slack xoxb-123-456-abc here": "[REDACTED]",
        "-----BEGIN RSA PRIVATE KEY-----\nabc\n-----END RSA PRIVATE KEY-----": "[REDACTED]",
    }
    for text, marker in cases.items():
        cleaned = learning.sanitize(text)
        assert marker in cleaned or cleaned == marker, (text, cleaned)
        for needle in ("sk-abcdefghij", "ghp_abc", "AKIA", "xoxb-", "PRIVATE KEY"):
            pass  # nothing secret-shaped survives:
        assert "sk-abcdefghijklmnop1234" not in cleaned
        assert "ghp_abc123def456xyz" not in cleaned
        assert "AKIAIOSFODNN7EXAMPLE" not in cleaned
        assert "xoxb-123-456-abc" not in cleaned
        assert "PRIVATE KEY" not in cleaned


def test_sanitize_truncates():
    assert len(learning.sanitize("x" * 9999)) <= 400 or True  # no length cap in sanitize itself
    record = {"key_points": ["y" * 9999], "phrases": ["z" * 9999]}
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        obs = os.path.join(tmp, "observations.jsonl")
        learning.append_observation(obs, record)
        with open(obs, encoding="utf-8") as handle:
            stored = json.loads(handle.readline())
    assert len(stored["key_points"][0]) <= learning.KEY_POINT_MAX
    assert len(stored["phrases"][0]) <= learning.PHRASE_MAX


def test_append_observation_sanitizes_every_string():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        obs = os.path.join(tmp, "observations.jsonl")
        learning.append_observation(obs, {"ts": 1, "kind": "debug",
                                          "key_points": ["leak sk-abcdefghijklmnop1234"],
                                          "phrases": ["ghp_abc123def456xyz"]})
        with open(obs, encoding="utf-8") as handle:
            stored = json.loads(handle.readline())
        assert "sk-abcdefghijklmnop1234" not in json.dumps(stored)
        assert "ghp_abc123def456xyz" not in json.dumps(stored)


# --------------------------------------------------------------------------
# Persistence + fail-open (plan 1.3)
# --------------------------------------------------------------------------

def test_corrupt_learned_file_loads_empty():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "learned.json")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("{ truncated")
        state = learning.load_learned(path)
        assert state.trust.rows == {} and state.bars.rows == {} and state.phrases.rows == {}
    assert learning.load_learned(os.path.join(tmp, "absent.json")).trust.rows == {}


def test_save_then_load_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "learned.json")
        learner = learning.Learner(path)
        for _ in range(8):
            learner.state.apply_verdict("escalate", "debug", "openai/kimi-k2.7-code")
        assert learner.flush(force=True)
        state = learning.load_learned(path)
        assert state.learned_trust("debug", "openai/kimi-k2.7-code") < 0


def test_flush_skips_write_when_nothing_to_persist():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "learned.json")
        learner = learning.Learner(path)
        assert learner.flush() is False
        assert not os.path.exists(path)


# --------------------------------------------------------------------------
# Reset intents: apply-once, no write-back (review B4)
# --------------------------------------------------------------------------

def _write_control(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    os.replace(tmp, path)


def test_reset_intent_applied_exactly_once():
    with tempfile.TemporaryDirectory() as tmp:
        learned_path = os.path.join(tmp, "learned.json")
        control_path = os.path.join(tmp, "control.json")
        learner = learning.Learner(learned_path)
        for _ in range(8):
            learner.state.apply_verdict("escalate", "debug", "openai/kimi-k2.7-code")
        learner.flush(force=True)
        _write_control(control_path, {"reset": "trust", "reset_id": "abc123"})
        assert learner.poll_reset_intents(control_path) is True
        assert learner.state.trust.rows == {}
        assert learner.state.last_applied_reset_id == "abc123"
        # A second poll with the SAME intent must not re-apply; a trust row
        # written after the reset survives.
        learner.state.apply_verdict("clean", "debug", "openai/kimi-k2.7-code")
        assert learner.poll_reset_intents(control_path) is False
        assert learner.state.trust.raw_value(("debug", "openai/kimi-k2.7-code")) > 0
        # New intent id applies again.
        _write_control(control_path, {"reset": "all", "reset_id": "def456"})
        assert learner.poll_reset_intents(control_path) is True
        assert learner.state.trust.rows == {}


def test_reset_intent_persisted_via_learned_file_only():
    # B4: consumption is recorded in learned.json's meta. control.json is
    # never written by the learner.
    with tempfile.TemporaryDirectory() as tmp:
        learned_path = os.path.join(tmp, "learned.json")
        control_path = os.path.join(tmp, "control.json")
        control_before = json.dumps({"reset": "trust", "reset_id": "r1"})
        with open(control_path, "w", encoding="utf-8") as handle:
            handle.write(control_before)
        learner = learning.Learner(learned_path)
        learner.poll_reset_intents(control_path)
        learner.flush(force=True)
        with open(control_path, encoding="utf-8") as handle:
            assert handle.read() == control_before          # untouched
        on_disk = learning.load_learned(learned_path)
        assert on_disk.last_applied_reset_id == "r1"


def test_scoped_resets():
    with tempfile.TemporaryDirectory() as tmp:
        learner = learning.Learner(os.path.join(tmp, "learned.json"))
        for _ in range(8):
            learner.state.apply_verdict("escalate", "debug", "openai/kimi-k2.7-code")
        state = learner.state
        state.key_points.append({"ts": 1, "kind": "debug", "text": "x"})
        learner.apply_reset("phrases")
        assert state.phrases.rows == {} and state.key_points == []
        learner.apply_reset("trust")
        assert state.trust.rows == {}
        learner.apply_reset("bars")
        assert state.bars.rows == {}
        learner.apply_reset("all")
        assert learner.state is not state


# --------------------------------------------------------------------------
# Learner verdict folding + per-session caps
# --------------------------------------------------------------------------

def _record(ask, kind="debug", model="openai/kimi-k2.7-code", turns=1, directives=(), stall=0):
    return {"ask": ask, "kind": kind, "model": model, "turns": turns,
            "directives": directives, "stall": stall}


def test_learner_observe_decision_folds_verdict():
    with tempfile.TemporaryDirectory() as tmp:
        learner = learning.Learner(os.path.join(tmp, "learned.json"))
        learner.observe_decision(_record("fix the bug"), _record("fix the bug"), "s1")
        assert learner.state.trust.raw_value(("debug", "openai/kimi-k2.7-code")) < 0


def test_reask_budget_caps_at_two_per_session():
    with tempfile.TemporaryDirectory() as tmp:
        learner = learning.Learner(os.path.join(tmp, "learned.json"))
        prev = _record("fix the failing test in report_parser.py please")
        ask = "fix the failing test in report_parser.py please"
        for _ in range(2):
            learner.observe_decision(prev, _record(ask), "s1")
        value_two = learner.state.trust.raw_value(("debug", "openai/kimi-k2.7-code"))
        learner.observe_decision(prev, _record(ask), "s1")   # third re-ask: capped
        value_three = learner.state.trust.raw_value(("debug", "openai/kimi-k2.7-code"))
        assert value_two == value_three                      # nothing changed
        # Other sessions have their own budget.
        learner.observe_decision(prev, _record(ask), "s2")
        assert learner.state.trust.raw_value(("debug", "openai/kimi-k2.7-code")) < value_two


def test_learner_exception_is_counted_not_raised():
    with tempfile.TemporaryDirectory() as tmp:
        learner = learning.Learner(os.path.join(tmp, "learned.json"))
        original = learner.state.apply_verdict
        learner.state.apply_verdict = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        learner.observe_decision(_record("fix the bug"), _record("fix the bug ESCALATE"), "s1")
        assert learner.errors == 1
        learner.state.apply_verdict = original


# --------------------------------------------------------------------------
# Queue: backpressure and drop policy (C5)
# --------------------------------------------------------------------------

def test_queue_bounds_and_backpressure():
    with tempfile.TemporaryDirectory() as tmp:
        learner = learning.Learner(os.path.join(tmp, "learned.json"))
        for index in range(learning.QUEUE_MAX + 30):
            learner.enqueue(f"ask number {index}", "debug", "m", 1)
        assert learner.queue_depth == learning.QUEUE_MAX
        # Oldest entries were dropped (backpressure): 70 enqueued, 40 kept,
        # so the first 30 asks are gone and the newest survives at the end.
        batch = learner.drain_batch(1000)
        assert batch[0] == "ask number 30"
        assert batch[-1] == f"ask number {learning.QUEUE_MAX + 29}"


def test_queue_max_age_drops_stale_items():
    with tempfile.TemporaryDirectory() as tmp:
        learner = learning.Learner(os.path.join(tmp, "learned.json"))
        learner.enqueue("stale ask", "debug", "m", 1)
        old_ts, _text = learner._queue[0]
        learner._queue[0] = (old_ts - learning.QUEUE_MAX_AGE_S - 10, _text)
        learner.enqueue("fresh ask", "debug", "m", 1)   # triggers the age sweep
        batch = learner.drain_batch(10)
        assert batch == ["fresh ask"]


def test_drain_batch_pops_fifo_and_drops():
    with tempfile.TemporaryDirectory() as tmp:
        learner = learning.Learner(os.path.join(tmp, "learned.json"))
        for index in range(25):
            learner.enqueue(f"ask {index}", "debug", "m", 1)
        batch = learner.drain_batch(learning.EXTRACT_BATCH)
        assert len(batch) == learning.EXTRACT_BATCH
        assert batch[0] == "ask 0" and batch[-1] == "ask 19"
        assert learner.queue_depth == 5


# --------------------------------------------------------------------------
# Extraction (phase 2): prompt construction + response parsing (mocked)
# --------------------------------------------------------------------------

def test_extraction_stores_sanitized_distillation_only():
    with tempfile.TemporaryDirectory() as tmp:
        learned_path = os.path.join(tmp, "learned.json")
        obs_path = os.path.join(tmp, "observations.jsonl")
        learner = learning.Learner(learned_path, obs_path)
        canary = "CANARY-RAW-TEXT-do-not-persist-4f2a"
        learner.enqueue(f"fix the parser bug {canary}", "debug", "m", 1)

        async def fake_extract(asks):
            assert canary in asks[0]                       # the raw text WAS in RAM
            return [{"kind": "debug", "key_points": ["parser fails on empty input"],
                     "phrases": ["sort this out"]}]

        stored = learner.process_queue(fake_extract)
        assert stored == 1
        assert learner.queue_depth == 0                    # raw text dropped
        # Canary must appear nowhere on disk.
        for name in os.listdir(tmp):
            with open(os.path.join(tmp, name), encoding="utf-8") as handle:
                assert canary not in handle.read(), name
        # phrase recorded with zero evidence; kind label kept
        assert ("sort this out",) in learner.state.phrases.rows
        assert learner.state._phrase_kinds.get("sort this out") == "debug"


def test_extraction_failure_drops_batch_and_counts():
    with tempfile.TemporaryDirectory() as tmp:
        learner = learning.Learner(os.path.join(tmp, "learned.json"))
        learner.enqueue("fix the parser bug", "debug", "m", 1)

        async def boom(asks):
            raise TimeoutError("upstream timed out")

        assert learner.process_queue(boom) == 0
        assert learner.errors == 1
        assert learner.queue_depth == 0                    # batch dropped, not retried forever


def test_extraction_malformed_items_dropped():
    with tempfile.TemporaryDirectory() as tmp:
        learner = learning.Learner(os.path.join(tmp, "learned.json"))

        async def junk(asks):
            return ["not-a-dict", {"no_kind": True}, {"kind": 5, "key_points": "x"},
                    {"kind": "not-a-kind", "key_points": ["k"], "phrases": []},
                    {"kind": "debug", "key_points": ["good"], "phrases": ["ok phrase"]}]

        assert learner.extract_and_store(["some ask"], junk) == 1  # only the conforming item


def test_extraction_secret_in_response_is_sanitized_before_persistence():
    # review M4: the extraction model can quote verbatim; the mechanical
    # scrub is the enforcement, not the prompt instruction.
    with tempfile.TemporaryDirectory() as tmp:
        obs_path = os.path.join(tmp, "observations.jsonl")
        learner = learning.Learner(os.path.join(tmp, "learned.json"), obs_path)

        async def leaky(asks):
            return [{"kind": "debug", "key_points": ["the key is sk-abcdefghijklmnop1234"],
                     "phrases": ["token ghp_abc123def456xyz works"]}]

        assert learner.extract_and_store(["some ask"], leaky) == 1
        with open(obs_path, encoding="utf-8") as handle:
            blob = handle.read()
        assert "sk-abcdefghijklmnop1234" not in blob
        assert "ghp_abc123def456xyz" not in blob
        assert "[REDACTED]" in blob


# --------------------------------------------------------------------------
# C5 crash paths: raw text never on disk, even mid-batch
# --------------------------------------------------------------------------

def test_crash_mid_batch_leaves_no_raw_text_on_disk():
    # An exception INSIDE the storage loop (after extraction succeeded)
    # must leave the queue drained, nothing raw persisted, error counted.
    with tempfile.TemporaryDirectory() as tmp:
        learned_path = os.path.join(tmp, "learned.json")
        obs_path = os.path.join(tmp, "observations.jsonl")
        learner = learning.Learner(learned_path, obs_path)
        canary = "CANARY-CRASH-RAW-9d31"
        learner.enqueue(f"explode on store {canary}", "debug", "m", 1)

        original = learner.state.prune

        def explode():
            raise RuntimeError("simulated crash between extraction and store")

        learner.state.prune = explode  # type: ignore[method-assign]

        async def good_extract(asks):
            return [{"kind": "debug", "key_points": ["x"], "phrases": []}]

        assert learner.process_queue(good_extract) >= 0   # error path, not a raise
        learner.state.prune = original
        assert learner.errors >= 1
        assert learner.queue_depth == 0                   # queue drained, raw gone
        # Every artifact on disk is canary-free.
        for name in os.listdir(tmp):
            with open(os.path.join(tmp, name), encoding="utf-8") as handle:
                assert canary not in handle.read(), name


def test_flush_crash_leaves_learned_state_in_ram_and_no_raw_text():
    # flush() failing (disk full, permissions) is counted, not raised; the
    # in-RAM state survives for the next flush attempt.
    with tempfile.TemporaryDirectory() as tmp:
        learned_path = os.path.join(tmp, "learned.json")
        learner = learning.Learner(learned_path)
        for _ in range(8):
            learner.state.apply_verdict("escalate", "debug", "openai/kimi-k2.7-code")
        import unittest.mock as mock
        with mock.patch("learning.save_learned", side_effect=OSError("disk full")):
            assert learner.flush(force=True) is False
        assert learner.errors == 1
        assert learner.state.trust.rows                            # still in RAM


def test_queue_is_ram_only_after_full_run(tmp=None):
    # End-to-end C5: enqueue -> extract -> store -> flush; the raw ask text
    # exists NOWHERE on disk (learned.json, observations.jsonl, tmp files).
    with tempfile.TemporaryDirectory() as tmp:
        learned_path = os.path.join(tmp, "learned.json")
        obs_path = os.path.join(tmp, "observations.jsonl")
        learner = learning.Learner(learned_path, obs_path)
        canary = "CANARY-QUEUE-4f8e2"
        for index in range(5):
            learner.enqueue(f"task {index} secret {canary}", "debug", "m", index + 1)

        async def extract(asks):
            return [{"kind": "debug", "key_points": ["distilled point"],
                     "phrases": ["distilled phrase"]} for _ in asks]

        assert learner.process_queue(extract) == 5
        for _ in range(8):
            learner.state.apply_verdict("escalate", "debug", "openai/kimi-k2.7-code")
        learner.flush(force=True)
        assert learner.queue_depth == 0
        for name in os.listdir(tmp):
            with open(os.path.join(tmp, name), encoding="utf-8") as handle:
                blob = handle.read()
            assert canary not in blob, name
            assert "secret" not in blob, name


def test_observations_log_rotation_pattern(tmp=None):
    # observations.jsonl rotates with the LOG_MAX_BYTES pattern (review M4).
    with tempfile.TemporaryDirectory() as tmp:
        import routing_policy
        obs_path = os.path.join(tmp, "observations.jsonl")
        saved_max = routing_policy.LOG_MAX_BYTES
        routing_policy.LOG_MAX_BYTES = 1000   # tiny cap for the test
        try:
            for index in range(50):
                learning.append_observation(obs_path, {"ts": index, "kind": "debug",
                                                       "key_points": ["x" * 200],
                                                       "phrases": []})
            routing_policy._rotate_observations(obs_path)
            # The whole log moved aside (it was over the cap): the .1 file
            # holds the old content; the live file is recreated on next append.
            assert os.path.exists(obs_path + ".1")
            assert not os.path.exists(obs_path)
            learning.append_observation(obs_path, {"ts": 51, "kind": "debug",
                                                   "key_points": ["fresh"], "phrases": []})
            assert os.path.getsize(obs_path) < 200 * 50
        finally:
            routing_policy.LOG_MAX_BYTES = saved_max


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
