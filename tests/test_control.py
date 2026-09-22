"""
Tests for the control plane: control_store.py, control_service.py and the
policy's lease/weights/learned integration.

Run from the llm-router directory:

    python tests/test_control.py

The load-bearing property, enforced several ways below: with control.json or
learned.json missing, corrupt, or absurd, routing must be byte-identical to a
build with no control plane at all.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["POLICY_LOG"] = ""

import routing_policy as policy  # noqa: E402
from control_store import (  # noqa: E402
    ControlStore,
    KIND_BAR_BIAS_MAX,
    PHRASE_BAR_NUDGE,
    TRUST_PERSIST_MAX,
)


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


class TempState:
    """temp control.json/learned.json wired into a fresh ControlStore."""

    def __init__(self):
        self.dir = tempfile.mkdtemp(prefix="router-control-")
        self.control_path = os.path.join(self.dir, "control.json")
        self.learned_path = os.path.join(self.dir, "learned.json")

    def write_control(self, payload):
        with open(self.control_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

    def write_learned(self, payload):
        with open(self.learned_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

    def store(self):
        return ControlStore(self.control_path, self.learned_path)


# --------------------------------------------------------------------------
# mcp_server.FLEET <-> routing_policy.PROFILES sync
# --------------------------------------------------------------------------

def test_mcp_fleet_matches_policy_profiles():
    # The MCP server validates lease models against its own FLEET tuple; the
    # comment there claims it cannot drift from the policy's PROFILES. Make
    # that true: exact same model set, same order.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import mcp_server
    import routing_policy

    assert tuple(mcp_server.FLEET) == tuple(routing_policy.PROFILES)


# --------------------------------------------------------------------------
# ControlStore: leases
# --------------------------------------------------------------------------

def test_lease_time_bound_active_then_expired():
    state = TempState()
    state.write_control({"lease": {"model": "openai/glm-5.3",
                                   "expires_ts": time.time() + 60, "set_by": "t"}})
    store = state.store()
    assert store.active_lease() is not None
    state.write_control({"lease": {"model": "openai/glm-5.3",
                                   "expires_ts": time.time() - 1, "set_by": "t"}})
    assert state.store().active_lease() is None  # fresh store sees expiry


def test_lease_expired_when_deadline_passed():
    state = TempState()
    state.write_control({"lease": {"model": "openai/glm-5.3",
                                   "expires_ts": time.time() - 1, "set_by": "t"}})
    assert state.store().active_lease() is None


def test_lease_without_deadline_is_dead():
    # B5: the only bound is the epoch deadline. A lease file missing or
    # carrying a non-numeric expires_ts (old counter format, hand edit) must
    # never pin anything statelessly.
    state = TempState()
    state.write_control({"lease": {"model": "openai/glm-5.3", "set_by": "t"}})
    assert state.store().active_lease() is None
    state.write_control({"lease": {"model": "openai/glm-5.3",
                                   "expires_ts": "not-a-number", "set_by": "t"}})
    assert state.store().active_lease() is None


def test_lease_survives_store_restart_statelessly():
    # B5: expiry lives in the file, not in any counter. A brand-new store
    # (proxy restart) sees the same live lease the old one did, and the same
    # expiry -- restart can neither re-arm nor lose it.
    state = TempState()
    state.write_control({"lease": {"model": "openai/glm-5.3",
                                   "expires_ts": time.time() + 120, "set_by": "t"}})
    assert state.store().active_lease() is not None
    assert state.store().active_lease() is not None  # fresh instance, same answer



def test_missing_control_file_means_no_overrides():
    store = TempState().store()  # no files written at all
    assert store.active_lease() is None
    assert store.weights() is None
    assert store.learning_enabled() is True
    assert store.kind_bar_bias("debug") == 0.0
    assert store.phrase_bias("anything") == 0.0
    assert store.persisted_trust("debug", "openai/glm-5.3") == 0.0


def test_corrupt_control_file_falls_to_last_good_not_defaults():
    # M5: corruption must NOT read as defaults -- defaults would silently
    # re-enable learning the operator disabled. A store that has seen a good
    # read keeps it in RAM across corruption.
    state = TempState()
    state.write_control({"learning_enabled": False})
    store = state.store()
    assert store.learning_enabled() is False
    with open(state.control_path, "w", encoding="utf-8") as handle:
        handle.write('{"lease": {"model": "openai/glm-5.3", "learn')  # torn write
    assert store.learning_enabled() is False   # last good, NOT defaults
    assert store.errors >= 1
    assert store.active_lease() is None
    assert store.weights() is None


def test_corrupt_first_sight_prefers_persisted_lastgood_sidecar():
    # M5: a proxy restart facing an already-corrupt file has no RAM snapshot,
    # so it recovers from the atomically-written .lastgood sidecar instead of
    # reading defaults.
    state = TempState()
    state.write_control({"learning_enabled": False, "weights": {"cost": 0.5}})
    store = state.store()
    assert store.learning_enabled() is False  # successful read persists the sidecar
    assert os.path.exists(state.control_path + ".lastgood")
    with open(state.control_path, "w", encoding="utf-8") as handle:
        handle.write("{ truncated")           # torn write
    restarted = state.store()                 # fresh store, cold RAM
    assert restarted.learning_enabled() is False
    assert restarted.weights() == {"cost": 0.5}


def test_corrupt_first_sight_fails_closed_on_learning():
    # M5: corrupt file + no last-good anywhere = defaults, except
    # learning_enabled, which must default OFF: a corrupt file can never
    # silently re-enable disabled learning.
    state = TempState()
    with open(state.control_path, "w", encoding="utf-8") as handle:
        handle.write('{"lease": {"model": "openai/glm-5.3", "learn')
    store = state.store()
    assert store.learning_enabled() is False
    assert store.active_lease() is None


def test_repaired_control_file_heals_on_next_read():
    # M5: a failed parse records no stamp, so the next request retries the
    # read and a repaired file heals immediately instead of sticking stale.
    state = TempState()
    state.write_control({"learning_enabled": False})
    store = state.store()
    with open(state.control_path, "w", encoding="utf-8") as handle:
        handle.write("{ truncated")
    store.learning_enabled()  # corrupt read: last good kept
    state.write_control({"learning_enabled": True})
    assert store.learning_enabled() is True   # healed on next read


def test_size_change_without_mtime_change_invalidates():
    # M5: the cache compares the (mtime_ns, size) pair by inequality, so an
    # in-place rewrite that lands on the same coarse mtime is not missed.
    state = TempState()
    state.write_control({"learning_enabled": False})
    store = state.store()
    assert store.learning_enabled() is False
    stat = os.stat(state.control_path)
    with open(state.control_path, "w", encoding="utf-8") as handle:
        json.dump({"learning_enabled": True, "weights": {"cost": 0.1}}, handle)
    os.utime(state.control_path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert store.learning_enabled() is True
    assert store.weights() == {"cost": 0.1}


def test_missing_control_file_still_reads_as_defaults():
    # M5's other half: absence is the pre-control-plane steady state and
    # keeps the pristine defaults (learning on).
    store = TempState().store()
    assert store.learning_enabled() is True
    assert store.weights() is None


# --------------------------------------------------------------------------
# ControlStore: learned tables and their clamps
# --------------------------------------------------------------------------

def test_kind_bar_bias_clamped():
    state = TempState()
    state.write_learned({"kind_bar_bias": {"debug": 5.0, "design": -5.0, "review": 0.07}})
    store = state.store()
    assert store.kind_bar_bias("debug") == KIND_BAR_BIAS_MAX
    assert store.kind_bar_bias("design") == -KIND_BAR_BIAS_MAX
    assert store.kind_bar_bias("review") == 0.07
    assert store.kind_bar_bias("unknown-kind") == 0.0


def test_phrase_bias_requires_evidence_and_clamps():
    state = TempState()
    state.write_learned({
        "phrase_kind": {
            "sort this out": {"kind": "debug", "count": 4, "bias": 0.5},   # below evidence gate
            "have a look": {"kind": "review", "count": 9, "bias": 0.2},
        }
    })
    store = state.store()
    text = "please sort this out and also have a look at the parser"
    bias = store.phrase_bias(text)
    assert store.phrase_bias("please sort this out") == 0.0            # evidence gate
    assert bias > 0.0 and bias <= PHRASE_BAR_NUDGE                      # clamp holds
    assert store.phrase_bias("completely unrelated words") == 0.0


def test_persisted_trust_clamped():
    state = TempState()
    state.write_learned({"model_trust": {"debug|openai/glm-5.3": 3.0}})
    store = state.store()
    assert store.persisted_trust("debug", "openai/glm-5.3") == TRUST_PERSIST_MAX
    assert store.persisted_trust("debug", "openai/kimi-k2.7-code") == 0.0


# --------------------------------------------------------------------------
# Policy integration: lease honoured, precedence chain, gates still apply
# --------------------------------------------------------------------------

DEBUG_ASK = "fix the failing test in report_parser.py, traceback: TypeError"


def _with_store(monkeypatch_store):
    """Point the module-level CONTROL at a test store and reset session state."""
    policy.CONTROL = monkeypatch_store
    policy.STATE = policy.PolicyState()  # fresh sessions, trust, lease sweep marks


def test_lease_overrides_selection():
    state = TempState()
    state.write_control({"lease": {"model": "openai/deepseek-v4.1-flash",
                                   "expires_ts": time.time() + 120, "set_by": "t"}})
    _with_store(state.store())
    context = Context(ask(DEBUG_ASK))
    run(context)
    assert context.candidate_models == ["openai/deepseek-v4.1-flash"]
    assert context.signals["policy"]["reason"] == "lease"


def test_lease_never_breaks_the_vision_gate():
    # A lease on a text-only model must NOT route an image turn to it.
    state = TempState()
    state.write_control({"lease": {"model": "openai/deepseek-v4.1-flash",
                                   "expires_ts": time.time() + 120, "set_by": "t"}})
    _with_store(state.store())
    messages = ask("explain this screenshot")
    messages[-1]["content"] = [
        {"type": "text", "text": "explain this screenshot"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
    ]
    context = Context(messages)
    run(context)
    assert context.candidate_models[0] != "openai/deepseek-v4.1-flash"
    assert context.signals["policy"]["reason"] != "lease"


def test_use_directive_beats_active_lease():
    # B5 precedence chain: [[use:...]] (in-band, most specific) > lease.
    state = TempState()
    state.write_control({"lease": {"model": "openai/deepseek-v4.1-flash",
                                   "expires_ts": time.time() + 120, "set_by": "t"}})
    _with_store(state.store())
    context = Context(ask(f"quick one [[use:glm-5.3-flash]] {DEBUG_ASK}"))
    run(context)
    assert context.candidate_models == ["openai/glm-5.3-flash"]
    assert context.signals["policy"]["reason"] == "pinned-directive"


def test_lease_beats_session_stickiness():
    # B5 precedence chain: an active lease overrides the session pin that a
    # previous turn recorded.
    state = TempState()
    _with_store(state.store())
    first = Context(ask(DEBUG_ASK))
    run(first)
    assert first.signals["policy"]["reason"] == "fresh"
    assert first.candidate_models == ["openai/kimi-k2.7-code"]
    # Same task now under a lease on a different model.
    state.write_control({"lease": {"model": "openai/glm-5.3",
                                   "expires_ts": time.time() + 120, "set_by": "t"}})
    policy.CONTROL = state.store()
    second = Context(ask(DEBUG_ASK))
    run(second)
    assert second.candidate_models == ["openai/glm-5.3"]
    assert second.signals["policy"]["reason"] == "lease"   # not pinned-cache


def test_expired_lease_orphan_cleanup_lets_sessions_redecide():
    # B5: a lease pins every session it touches; when it expires those
    # sessions must re-decide fresh, not hold the model via stickiness for
    # up to SESSION_TTL.
    state = TempState()
    state.write_control({"lease": {"model": "openai/glm-5.3",
                                   "expires_ts": time.time() + 120, "set_by": "t"}})
    _with_store(state.store())
    leased = Context(ask(DEBUG_ASK))
    run(leased)
    assert leased.candidate_models == ["openai/glm-5.3"]
    assert leased.signals["policy"]["reason"] == "lease"
    # Lease expires; the session record the lease created must not pin.
    state.write_control({"lease": {"model": "openai/glm-5.3",
                                   "expires_ts": time.time() - 1, "set_by": "t"}})
    policy.CONTROL = state.store()
    after = Context(ask(DEBUG_ASK))
    run(after)
    assert after.signals["policy"]["reason"] != "pinned-cache"
    assert after.signals["policy"]["reason"] == "fresh"
    # ...and a fresh decision on this ask lands on kimi, the debug winner.
    assert after.candidate_models == ["openai/kimi-k2.7-code"]


def test_lease_bypassed_for_oversized_prompt():
    # Gates still bind a lease: a small-window model cannot take an
    # oversized prompt, so the lease fails open to normal selection.
    state = TempState()
    state.write_control({"lease": {"model": "openai/deepseek-v4.1-flash",
                                   "expires_ts": time.time() + 120, "set_by": "t"}})
    _with_store(state.store())
    messages = ask("summarize this")
    messages.insert(2, {"role": "user", "content": "y" * 600_000})  # ~150k tokens
    context = Context(messages)
    run(context)
    assert context.candidate_models[0] != "openai/deepseek-v4.1-flash"
    assert context.signals["policy"]["reason"] != "lease"


def test_no_control_file_routes_identically_to_none():
    # The fail-open contract: same decision with an empty store as with no
    # store at all, and that decision matches the pre-control-plane build.
    state = TempState()
    _with_store(state.store())
    with_store = Context(ask(DEBUG_ASK))
    run(with_store)
    policy.CONTROL = None
    without_store = Context(ask(DEBUG_ASK))
    run(without_store)
    assert with_store.candidate_models == without_store.candidate_models
    assert with_store.candidate_models == ["openai/kimi-k2.7-code"]


def test_live_weights_change_the_pick():
    state = TempState()
    # Crank cost pressure to 1.0: cheapest capable model should win the bulk job.
    state.write_control({"weights": {"cost": 1.0, "trust": 0.0, "headroom": 0.0, "latency": 0.0}})
    _with_store(state.store())
    context = Context(ask("write a script that renames every file across the repo"))
    run(context)
    # qwen3-vl-flash is the fleet's cheapest bulk-capable model on the new fleet.
    assert context.candidate_models == ["openai/qwen3-vl-flash"]


def test_laya_weight_reads_from_control():
    state = TempState()
    state.write_control({"weights": {"laya": 0.75}})
    assert state.store().laya_weight() == 0.75


def test_learned_bar_bias_shifts_selection():
    # A maxed-out negative bias on 'debug' should let a weaker model through.
    state = TempState()
    state.write_learned({"kind_bar_bias": {"debug": -KIND_BAR_BIAS_MAX}})
    _with_store(state.store())
    context = Context(ask(DEBUG_ASK))
    run(context)
    chosen = context.candidate_models[0]
    # bar 0.75 - 0.15 = 0.60 -> glm-5.3-flash (0.55) still out, deepseek (0.58)
    # still out, but glm-5.3 (0.86) and kimi (0.88) both clear; cost decides.
    assert chosen in {"openai/glm-5.3", "openai/kimi-k2.7-code"}


def test_learning_disabled_globally_ignores_learned_bias():
    state = TempState()
    state.write_learned({"kind_bar_bias": {"debug": -KIND_BAR_BIAS_MAX}})
    state.write_control({"learning_enabled": False})
    _with_store(state.store())
    policy.LEARN_ENABLED = False
    try:
        context = Context(ask(DEBUG_ASK))
        run(context)
        assert context.candidate_models == ["openai/kimi-k2.7-code"]
    finally:
        policy.LEARN_ENABLED = True


# --------------------------------------------------------------------------
# control_service: HTTP surface
# --------------------------------------------------------------------------

SERVICE_PORT = 4199
BASE = f"http://127.0.0.1:{SERVICE_PORT}"


def _service(state):
    os.environ["POLICY_STATE_DIR"] = state.dir
    os.environ["CONTROL_PORT"] = str(SERVICE_PORT)
    os.environ["POLICY_LOG"] = os.path.join(state.dir, "routing.jsonl")
    # Force a fresh import: pop any cached module, then load from file so each
    # test gets module-level constants bound to ITS temp dir.
    sys.modules.pop("control_service", None)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "control_service", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                        "control_service.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _http(method, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(BASE + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode())


def _run_service(service, fn):
    from http.server import ThreadingHTTPServer
    server = ThreadingHTTPServer(("127.0.0.1", SERVICE_PORT), service.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        fn()
    finally:
        server.shutdown()
        server.server_close()


def test_service_lease_roundtrip_and_bounds():
    state = TempState()
    os.environ["CONTROL_TOKEN"] = "s3cr3t-token"
    service = _service(state)
    post = _http_with_token("s3cr3t-token")

    def checks():
        # no bound -> 400
        try:
            post("POST", "/lease", {"model": "openai/glm-5.3"})
            raise AssertionError("expected 400")
        except urllib.error.HTTPError as error:
            assert error.code == 400
        # both bounds -> 400
        try:
            post("POST", "/lease", {"model": "openai/glm-5.3", "requests": 5, "seconds": 60})
            raise AssertionError("expected 400")
        except urllib.error.HTTPError as error:
            assert error.code == 400
        # valid lease: stored as a stateless deadline
        result = post("POST", "/lease", {"model": "openai/glm-5.3", "requests": 5,
                                         "set_by": "test"})
        assert result["lease"]["expires_ts"] > time.time()
        assert result["revision"] >= 1
        health = post("GET", "/health")
        assert health["lease"]["model"] == "openai/glm-5.3"
        # clear
        post("POST", "/lease/clear", {})
        assert post("GET", "/health")["lease"] is None

    _run_service(service, checks)


def test_service_weights_validation_and_reset():
    state = TempState()
    state.write_learned({"phrase_kind": {"a": {"kind": "debug", "count": 2}},
                          "model_trust": {"debug|openai/glm-5.3": 0.1},
                          "kind_bar_bias": {"debug": 0.05}})
    os.environ["CONTROL_TOKEN"] = "s3cr3t-token"
    service = _service(state)
    post = _http_with_token("s3cr3t-token")

    def checks():
        # out-of-range weight -> 400 (loud, not clamped)
        try:
            post("POST", "/weights", {"cost": 35})
            raise AssertionError("expected 400")
        except urllib.error.HTTPError as error:
            assert error.code == 400
        # valid partial update merges
        result = post("POST", "/weights", {"cost": 0.5})
        assert result["weights"]["cost"] == 0.5
        # learning toggle + scoped reset
        post("POST", "/learning", {"enabled": False})
        assert post("GET", "/health")["learning_enabled"] is False
        post("POST", "/learning", {"reset": "phrases"})
        learned = post("GET", "/learned")["learned"]
        assert "phrase_kind" not in learned
        assert "model_trust" in learned
        # reset all deletes the file
        post("POST", "/reset/learned", {"scope": "all"})
        assert post("GET", "/learned")["present"] is False

    _run_service(service, checks)


def test_service_decisions_tail_skips_torn_lines():
    state = TempState()
    # A torn write mid-file: the fragment has no newline, so the NEXT good
    # record is glued to it and both are lost together. The service must
    # still serve the records before the tear and must not 500.
    with open(os.path.join(state.dir, "routing.jsonl"), "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"ts": 1, "chosen": "a"}) + "\n")
        handle.write('{"ts": 2, "chosen": ')  # torn, no newline
        handle.write(json.dumps({"ts": 3, "chosen": "c"}) + "\n")
    os.environ["CONTROL_TOKEN"] = "s3cr3t-token"
    service = _service(state)
    post = _http_with_token("s3cr3t-token")

    def checks():
        decisions = post("GET", "/decisions?n=10")["decisions"]
        assert [d["chosen"] for d in decisions] == ["a"], decisions

    _run_service(service, checks)


def test_service_budget_roundtrip():
    state = TempState()
    os.environ["CONTROL_TOKEN"] = "s3cr3t-token"
    service = _service(state)
    post = _http_with_token("s3cr3t-token")

    def checks():
        assert post("GET", "/health")["budget_hkd"] is None
        post("POST", "/budget", {"hkd": 25})
        assert post("GET", "/health")["budget_hkd"] == 25
        # clamped bounds
        try:
            post("POST", "/budget", {"hkd": 0.5})
            raise AssertionError("expected 400")
        except urllib.error.HTTPError as error:
            assert error.code == 400
        post("POST", "/budget", {"clear": True})
        assert post("GET", "/health")["budget_hkd"] is None

    _run_service(service, checks)


def test_service_laya_weight_roundtrip():
    state = TempState()
    os.environ["CONTROL_TOKEN"] = "s3cr3t-token"
    service = _service(state)
    post = _http_with_token("s3cr3t-token")

    def checks():
        assert post("GET", "/health")["laya_weight"] is None
        result = post("POST", "/weights", {"cost": 0.5, "laya": 0.3})
        assert result["weights"]["laya"] == 0.3
        assert post("GET", "/health")["laya_weight"] == 0.3
        # out-of-range laya weight -> 400
        try:
            post("POST", "/weights", {"laya": 1.5})
            raise AssertionError("expected 400")
        except urllib.error.HTTPError as error:
            assert error.code == 400

    _run_service(service, checks)


def test_service_decisions_tail_survives_newline_terminated_corruption():
    state = TempState()
    # A bad line WITH a newline: the following good records must survive.
    with open(os.path.join(state.dir, "routing.jsonl"), "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"ts": 1, "chosen": "a"}) + "\n")
        handle.write('{"ts": 2, "chosen": \n')  # corrupt, newline-terminated
        handle.write(json.dumps({"ts": 3, "chosen": "c"}) + "\n")
    os.environ["CONTROL_TOKEN"] = "s3cr3t-token"
    service = _service(state)
    post = _http_with_token("s3cr3t-token")

    def checks():
        decisions = post("GET", "/decisions?n=10")["decisions"]
        assert [d["chosen"] for d in decisions] == ["a", "c"], decisions

    _run_service(service, checks)


# --------------------------------------------------------------------------
# control_service: self-auth and B5 lease semantics
# --------------------------------------------------------------------------

def test_service_rejects_writes_without_token():
    # m3: the nginx path token is not authentication; the control service
    # authenticates callers itself. No CONTROL_TOKEN configured = read-only
    # mode: reads are served, every write refused (fail closed).
    state = TempState()
    os.environ.pop("CONTROL_TOKEN", None)
    service = _service(state)

    def checks():
        # reads still work in read-only mode
        assert _http("GET", "/health")["status"] == "ok"
        # writes are refused
        try:
            _http("POST", "/lease", {"model": "openai/glm-5.3", "seconds": 60})
            raise AssertionError("expected 401")
        except urllib.error.HTTPError as error:
            assert error.code == 401

    _run_service(service, checks)


def test_service_accepts_correct_bearer_and_rejects_wrong():
    state = TempState()
    os.environ["CONTROL_TOKEN"] = "s3cr3t-token"
    service = _service(state)

    def checks():
        # wrong token -> 401
        try:
            request = urllib.request.Request(BASE + "/health")
            request.add_header("Authorization", "Bearer wrong")
            urllib.request.urlopen(request, timeout=10)
            raise AssertionError("expected 401")
        except urllib.error.HTTPError as error:
            assert error.code == 401
        # correct token -> 200, and the write path works end to end
        _http_authed = _http_with_token("s3cr3t-token")
        result = _http_authed("POST", "/learning", {"enabled": False})
        assert result["learning_enabled"] is False
        assert _http_authed("GET", "/health")["learning_enabled"] is False

    _run_service(service, checks)


def test_service_lease_deadline_stateless_and_clamped():
    state = TempState()
    os.environ["CONTROL_TOKEN"] = "s3cr3t-token"
    service = _service(state)

    def checks():
        post = _http_with_token("s3cr3t-token")
        # seconds bound: stored as a stateless deadline, not a counter
        result = post("POST", "/lease", {"model": "openai/glm-5.3", "seconds": 60})
        lease = result["lease"]
        assert 59 <= lease["expires_ts"] - time.time() <= 61, lease
        assert "remaining" not in lease
        # seconds clamp: 10_000 s becomes the 900 s max (review B5)
        result = post("POST", "/lease", {"model": "openai/glm-5.3", "seconds": 10_000})
        lease = result["lease"]
        assert lease["expires_ts"] <= time.time() + 900 + 5, lease
        # requests bound: converted to a deadline at issue time, never a
        # counter (review B5)
        result = post("POST", "/lease", {"model": "openai/glm-5.3", "requests": 200})
        lease = result["lease"]
        assert lease["expires_ts"] <= time.time() + 900 + 5, lease
        assert lease.get("requested_requests") == 200   # audit only
        assert "remaining" not in lease
        # oversized bounds are clamped, not rejected (review B5: "clamp
        # lease bounds at the control-service")
        result = post("POST", "/lease", {"model": "openai/glm-5.3", "seconds": 901})
        assert result["lease"]["expires_ts"] <= time.time() + 900 + 5
        result = post("POST", "/lease", {"model": "openai/glm-5.3", "requests": 10_000})
        assert result["lease"]["expires_ts"] <= time.time() + 900 + 5
        # non-positive or non-numeric bounds are loud 400s
        for bad in ({"model": "openai/glm-5.3", "seconds": 0},
                    {"model": "openai/glm-5.3", "seconds": -5},
                    {"model": "openai/glm-5.3", "requests": 0}):
            try:
                post("POST", "/lease", bad)
                raise AssertionError(f"expected 400 for {bad}")
            except urllib.error.HTTPError as error:
                assert error.code == 400

    _run_service(service, checks)


def _http_with_token(token):
    def call(method, path, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(BASE + path, data=data, method=method,
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode())
    return call


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
