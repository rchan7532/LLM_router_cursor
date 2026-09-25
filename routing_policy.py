"""
Task-aware policy router for the `cursor-auto` model group.

Loaded by the LiteLLM proxy from `router_settings.plugins`:

    router_settings:
      plugins:
        - routing_policy.router_policy

Contract (`litellm.types.router.RoutingPlugin`): an async ``run(context)`` that
receives the `litellm_params.model` string of every deployment in the group in
``context.candidate_models`` and narrows that list to the models allowed to
serve this request. Returning the list untouched means "no opinion" and the
router's own selection applies.

Pipeline, in order:

    1. hard gates      drop models that cannot serve the request at all
                       (no vision, prompt larger than the window)
    2. task signature  what kind of work the newest human ask is
    3. capability bar  the minimum capability that work needs
    4. utility         capability - cost + learned trust + headroom
    5. stickiness      keep the model pinned to the task so the provider
                       prompt cache stays warm; re-decide on a new ask or a
                       directive
    6. feedback        an escalation or a stalled loop lowers that model's
                       trust for that kind of work

Directives the caller can put in the newest message:

    [[escalate]]        raise the capability bar one step for this task
    [[cheap]]           drop to the cheapest model that satisfies the gates
    [[use:<model>]]     pin a specific profile key for this task

Plain ``ESCALATE`` (case-sensitive) also works, so the habit carries over.

Precedence (review B5): an in-band ``[[use:...]]`` directive wins for that
request; otherwise an active control-plane lease (reason ``lease``) wins
over session stickiness; only then does stickiness or a fresh decision
apply. A lease is a stateless epoch-seconds deadline read from control.json,
so restarts cannot re-arm or lose it.

The plugin never raises. Any internal failure leaves the candidate list
untouched, so the request still gets served by the router's default selection
rather than failing on a router bug.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import tempfile
import time
from typing import Any, Mapping, NamedTuple, Sequence

try:
    # Optional at import time: the control plane (leases, live weights,
    # learned tables). Every use is fail-open; a missing module or unreadable
    # file means "no overrides" and routing is byte-identical to a build
    # without it. Tests enforce that property.
    from control_store import ControlStore, default_store
except ImportError:  # pragma: no cover - deployed together, split for tests
    ControlStore = None  # type: ignore[assignment]
    default_store = None  # type: ignore[assignment]

try:
    # Optional at import time: the cross-session learner (phase 1 verdicts
    # + persistence, phase 2 extraction). Same fail-open discipline; tests
    # enforce byte-identical routing when it is absent or disabled.
    import learning as _learning
    from learning import Learner
    import extractor as _extractor
except ImportError:  # pragma: no cover - deployed together, split for tests
    _learning = None  # type: ignore[assignment]
    Learner = None  # type: ignore[assignment]
    _extractor = None  # type: ignore[assignment]

try:
    # Optional at import time: the phase-3 reliability hook exposes the
    # newest per-model failure timestamp. Used for cascade-lite escalation.
    import failure_hook as _failure_hook
except ImportError:  # pragma: no cover
    _failure_hook = None  # type: ignore[assignment]

try:
    # Optional at import time: local shadow scorer (convaiinnovations/laya).
    # Default weight 0 means it only logs; it never breaks routing.
    import laya_scorer as _laya_scorer
except ImportError:  # pragma: no cover
    _laya_scorer = None  # type: ignore[assignment]

# --------------------------------------------------------------------------
# Tuning (env-overridable so the VPS can retune without a code edit)
# --------------------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


W_COST = _env_float("POLICY_W_COST", 0.35)        # price pressure
W_TRUST = _env_float("POLICY_W_TRUST", 0.30)       # learned success weight
W_HEADROOM = _env_float("POLICY_W_HEADROOM", 0.10)  # room left in the window
W_LATENCY = _env_float("POLICY_W_LATENCY", 0.10)   # speed pressure
TRUST_FLOOR = _env_float("POLICY_TRUST_FLOOR", -0.40)
TRUST_CEILING = _env_float("POLICY_TRUST_CEILING", 0.40)
SESSION_TTL = _env_int("POLICY_SESSION_TTL", 3600)
SESSION_CAP = _env_int("POLICY_SESSION_CAP", 512)
LOG_PATH = os.environ.get("POLICY_LOG") or (
    "/app/logs/routing.jsonl"
    if os.name != "nt" and os.path.isdir("/app/logs")
    else os.path.join(tempfile.gettempdir(), "routing.jsonl")
)
LOG_MAX_BYTES = _env_int("POLICY_LOG_MAX_BYTES", 5_000_000)
DISABLED = os.environ.get("POLICY_DISABLE", "").lower() in {"1", "true", "yes"}
LEARN_ENABLED = os.environ.get("POLICY_LEARN", "1").lower() not in {"0", "false", "no"}

# Cascade-lite: after an observed upstream failure, the previous model is
# excluded for this many seconds and its trust is docked. A one-turn window
# is enough to hand a retry to the next best model without creating a
# persistent blacklist (litellm's own cooldowns handle that).
CASCADE_FAILURE_WINDOW_S = _env_float("POLICY_CASCADE_WINDOW_S", 120.0)
CASCADE_TRUST_PENALTY = _env_float("POLICY_CASCADE_TRUST_PENALTY", -0.25)
# Stickiness as a bounded preference, not an override. Grace window: turns
# 1..N of a pin hold unconditionally (the cache really is warm, and the
# first turns of a task are where continuity pays). After the window the
# held model must still WIN on utility against a decaying loyalty bonus.
# Without this, one expensive early classification (spec-like first turn on
# kimi/glm-5.3) locks the whole session: execution turns never re-compete,
# because agent-loop asks ("keep going" + tool output) read as Normal
# importance and the pin only broke on a Low-importance follow-up.
PIN_GRACE_TURNS = _env_int("POLICY_PIN_GRACE_TURNS", 4)
PIN_BONUS_START = _env_float("POLICY_PIN_BONUS_START", 0.12)
PIN_BONUS_FLOOR = _env_float("POLICY_PIN_BONUS_FLOOR", 0.03)
# Premium run ceiling (2026-09-22 kit-generation incident): a premium model
# held by session stickiness must be re-justified after this many consecutive
# turns. The loyalty bonus decays, but when the classifier put the bar above
# every flash model the "competition" was premium-vs-premium and never
# produced a cheap winner - kimi ran 44 turns (~73 HKD) that way. The
# ceiling forces a periodic fresh decision where cost pressure applies to
# the FULL fleet again. Cheap models are exempt: their pins are the point.
PIN_PREMIUM_CEIL = _env_int("POLICY_PIN_PREMIUM_CEIL", 10)
# Per-(model, task-kind) spend alarm. When one model has been used for >=
# this many HKD on one kind inside a session, subsequent same-kind turns in
# that session are forced to the cheapest capable model until the alarm is
# cleared. A soft pause: the router keeps serving, but it stops the runaway.
KIND_ALARM_HKD = _env_float("POLICY_KIND_ALARM_HKD", 20.0)
# Stale-ask execution demotion (2026-09-25 design-pin complaint): in an agent
# loop the newest human ask does not change while the agent works, so a
# design-shaped ask re-classifies as `design` on EVERY turn and the premium
# bar that was right for "identify the problem" also gates the 30 mechanical
# follow-through turns. Once the SAME ask has driven this many turns WITH
# tool activity, the decision is made and the remaining work is execution:
# re-signature the turn as agentic at Normal importance with no ask-scale so
# flash models clear the bar. A fresh human message, any directive, or a
# non-planning kind all skip or reset it. If the cheap model then stalls or
# fails, stall detection and cascade-lite raise the bar back - the safety
# net for a wrong demotion is one bad turn, not a bad session.
EXEC_PHASE_TURNS = _env_int("POLICY_EXEC_PHASE_TURNS", 8)
EXEC_PHASE_KINDS = frozenset({"design", "refactor", "debug", "review"})
EXEC_PHASE_KIND = "agentic"
# How long an alarm epoch survives without a live control-plane clear. If
# the control service increments the alarm_epoch we remember it here; old
# in-memory alarms are dropped when the epoch advances. A manual clear also
# bumps the epoch so every proxy forgets its stale alarms on the next read.
KIND_ALARM_TTL_S = _env_float("POLICY_KIND_ALARM_TTL_S", 30.0)

# Laya shadow scorer: max additive influence on utility. The control-plane
# weight is 0..1; it is scaled down so full weight still only nudges.
LAYA_WEIGHT_MAX = _env_float("POLICY_LAYA_WEIGHT_MAX", 0.15)

# Learning-loop cadence (phase 2): the extraction batcher polls this often
# and the learner flushes learned.json / reset intents on this cadence.
# Both run OUTSIDE the request path.
_EXTRACT_POLL_S = _env_float("POLICY_EXTRACT_POLL_S", 30.0)
_FLUSH_POLL_S = _env_float("POLICY_FLUSH_POLL_S", 60.0)
_EXTRACTION_TASK: Any = None
_TASKS_STARTED = False

# The control store (control.json + learned.json). One per process; reads are
# mtime-cached, so the steady-state cost per request is one os.stat. Never
# constructed when the module is unavailable, and never used on a request path
# without the try/except that already wraps _run.
CONTROL = default_store() if default_store is not None else None

# ---------------------------------------------------------------------------
# Learning plumbing (phase 1 + phase 2). Everything here is optional,
# lazily built, and fail-open: with POLICY_LEARN=0, learning disabled in
# control.json, or a missing learning module, the hooks are inert and
# routing is byte-identical to a build without any of it (review B1).
#
# OBSERVATIONS is the learner's queue-depth/error counter exposed for
# telemetry (review M6): the decision log carries it, so the control
# service's /health can show it without any new writable file.
# ---------------------------------------------------------------------------

LEARNER = None
OBSERVATIONS: dict[str, int] = {"queue_depth": 0, "errors": 0, "extractions": 0}
_LOG_PATH_ENV = LOG_PATH


def _learn_enabled() -> bool:
    """C8 precedence for learning: POLICY_LEARN=0 (LEARN_ENABLED) wins over
    control.json learning_enabled. No control read happens when the env
    switch is off, and no learning hook fires unless both say enabled."""
    if not LEARN_ENABLED or CONTROL is None or Learner is None:
        return False
    try:
        return bool(CONTROL.learning_enabled())
    except Exception:  # noqa: BLE001 - fail closed for learning, never routing
        return False


def _get_learner():
    """Lazily build the process-wide learner (queue + learned state)."""
    global LEARNER
    if LEARNER is None and _learning is not None:
        learned_path = os.environ.get(
            "LEARNED_PATH",
            os.path.join(
                os.environ.get("POLICY_STATE_DIR")
                or ("/app/logs" if os.path.isdir("/app/logs") else tempfile.gettempdir()),
                "learned.json",
            ),
        )
        LEARNER = Learner(learned_path, os.environ.get("POLICY_OBSERVATIONS_PATH", ""))
        LEARNER.state = _learning.load_learned(learned_path)
    return LEARNER


def _start_extraction_loop() -> None:
    """Supervised phase-2 background task: drains the ask queue in batches
    of EXTRACT_BATCH through the DIRECT-to-upstream extractor (review M2:
    never through this proxy). Supervised = recreated if it dies. Runs
    only when learning is enabled AND extraction credentials exist."""
    global _EXTRACTION_TASK
    learner = _get_learner()
    if learner is None or _extractor is None:
        return
    if not _extractor.EXTRACT_BASE_URL or not _extractor.EXTRACT_API_KEY:
        return

    async def _loop():
        while True:
            await asyncio.sleep(_EXTRACT_POLL_S)
            current = _get_learner()
            if current is None or not _learn_enabled():
                continue
            try:
                while current.queue_depth >= _learning.EXTRACT_BATCH:
                    current.process_queue(_extractor.extract)
                    OBSERVATIONS["extractions"] = current.extractions
                    if current.observations_path and current.state.key_points:
                        _rotate_observations(current.observations_path)
            except Exception:  # noqa: BLE001 - the supervisor must survive
                pass
            OBSERVATIONS["queue_depth"] = current.queue_depth
            OBSERVATIONS["errors"] = current.errors

    try:
        _EXTRACTION_TASK = asyncio.get_event_loop().create_task(_loop())
    except RuntimeError:
        _EXTRACTION_TASK = None  # no running loop; queue still drains on flush polls


def _rotate_observations(path: str) -> None:
    """LOG_MAX_BYTES-style rotation for observations.jsonl (review M4)."""
    try:
        if os.path.exists(path) and os.path.getsize(path) > LOG_MAX_BYTES:
            os.replace(path, path + ".1")
    except OSError:
        pass


async def _flush_poll() -> None:
    """Periodic learner upkeep outside the request path: reset-intent
    consumption and learned.json persistence."""
    while True:
        await asyncio.sleep(_FLUSH_POLL_S)
        learner = _get_learner()
        if learner is None or not _learn_enabled():
            continue
        try:
            control_path = os.environ.get(
                "CONTROL_PATH", os.path.join(
                    os.environ.get("POLICY_STATE_DIR")
                    or ("/app/logs" if os.path.isdir("/app/logs") else tempfile.gettempdir()),
                    "control.json",
                )
            )
            learner.poll_reset_intents(control_path)
            learner.flush()
            OBSERVATIONS["queue_depth"] = learner.queue_depth
            OBSERVATIONS["errors"] = learner.errors
        except Exception:  # noqa: BLE001
            pass


def _ensure_background_tasks() -> None:
    """Start the learner's supervised loops once, per event loop."""
    global _TASKS_STARTED
    if _TASKS_STARTED or Learner is None or _learning is None:
        return
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return  # no running loop (unit tests); hooks stay synchronous-safe
    _TASKS_STARTED = True
    _start_extraction_loop()
    try:
        asyncio.get_event_loop().create_task(_flush_poll())
    except RuntimeError:
        pass


def _record_learner_state() -> None:
    try:
        learner = _get_learner()
        if learner is not None:
            OBSERVATIONS["queue_depth"] = learner.queue_depth
            OBSERVATIONS["errors"] = learner.errors
            OBSERVATIONS["extractions"] = learner.extractions
    except Exception:  # noqa: BLE001
        pass


def _effective_weights() -> tuple[float, float, float, float]:
    """Env weights overridden by live control-plane values, if any. The store
    clamps nothing here because control_service validates 0..1 at write time;
    a hand-edited out-of-range file is still guarded by the fallback tuple."""
    cost, trust, headroom, latency = W_COST, W_TRUST, W_HEADROOM, W_LATENCY
    if CONTROL is not None:
        try:
            overrides = CONTROL.weights()
            if overrides:
                cost = float(overrides.get("cost", cost))
                trust = float(overrides.get("trust", trust))
                headroom = float(overrides.get("headroom", headroom))
                latency = float(overrides.get("latency", latency))
        except Exception:  # noqa: BLE001 - weights are optional
            pass
    return cost, trust, headroom, latency


def _laya_weight() -> float:
    """Scaled laya influence, 0..LAYA_WEIGHT_MAX. Absent or invalid control
    values read as 0, so the shadow scorer cannot move routing unless the
    operator explicitly enables it."""
    if CONTROL is None:
        return 0.0
    try:
        raw = CONTROL.laya_weight()
        if isinstance(raw, (int, float)):
            return max(0.0, min(1.0, float(raw))) * LAYA_WEIGHT_MAX
    except Exception:  # noqa: BLE001
        pass
    return 0.0

# Capability bar for a task, per task kind, at scale 0..2. A model is eligible
# when its declared capability for the kind meets the bar.
BASE_BAR: dict[str, float] = {
    "code_edit": 0.55,
    "code_gen": 0.65,
    "refactor": 0.75,
    "debug": 0.75,
    "review": 0.70,
    "design": 0.80,
    "explain": 0.55,
    "bulk": 0.50,
    "writing": 0.55,
    "factual": 0.45,
    "agentic": 0.65,
}
SCALE_BAR_STEP = 0.05  # each scale step raises the bar this much
MAX_SCALE = 3
# Importance tiers: capability-bar offset per tier (Low / Normal / High).
# -0.10 lets cheap flash models clear the bar for trivial asks; +0.10 keeps
# premium models for spec/plan work. One step, deliberately conservative:
# wrong side of one step costs a fraction of a cent, not a bad answer.
IMPORTANCE_OFFSETS = (-0.10, 0.0, 0.10)
# A model above this blended USD/1M cost is "premium" for the stickiness
# cost guard: it may not hold a Low-importance follow-up via session pin.
# glm-5.3-flash (0.23) and the flashes sit below; kimi/glm-5.3/haiku above.
PREMIUM_COST = 1.0

# --------------------------------------------------------------------------
# Model profiles
#
# Keys MUST equal the deployment's `litellm_params.model` string in
# litellm-config.yaml. `cap` is the model's capability per task kind, 0..1,
# judged once and then refined by the trust table. `cost` is USD per 1M tokens
# (input, output) and must match the deployment's model_info so the utility
# function and litellm's spend log agree. `latency` is a relative class, 0 =
# fastest. `strengths`/`weak` are labels only, kept for the log.
# --------------------------------------------------------------------------


class Profile(NamedTuple):
    """
    Capacity facts for one model.

    A NamedTuple, not a dataclass, on purpose: litellm loads this module with
    ``importlib`` without registering it in ``sys.modules``, and ``dataclasses``
    resolves string annotations through ``sys.modules`` (Python 3.14 raises
    ``AttributeError: 'NoneType' object has no attribute '__dict__'``). Nothing
    in this module may depend on the loader having registered it.
    """

    model: str
    vision: bool
    ctx_window: int
    cost_in: float      # USD per 1M input tokens
    cost_out: float     # USD per 1M output tokens
    latency: int        # 0 fastest .. 3 slowest
    cap: Mapping[str, float]
    strengths: tuple[str, ...] = ()
    weak: tuple[str, ...] = ()


# Prices are real, from the operator's HKD list (2026-09-21) at 7.8 HKD/USD.
# The fleet was live-probed: no id is listed without a 200 from a 1-token
# completion (kimi-k2.7-code, deepseek-v4.1-flash, qwen3-vl-flash verified).
# Wrong capability numbers do not break routing; they only skew quality
# pressure, so nudge them as real usage shows which models actually deliver.
PROFILES: dict[str, Profile] = {
    "openai/qwen3-vl-flash": Profile(
        model="openai/qwen3-vl-flash",
        vision=True,
        ctx_window=262_000,
        cost_in=0.016,
        cost_out=0.18,
        latency=0,
        cap={
            "code_edit": 0.60, "code_gen": 0.55, "refactor": 0.52, "debug": 0.50,
            "review": 0.52, "design": 0.48, "explain": 0.75, "bulk": 0.78,
            "writing": 0.70, "factual": 0.75, "agentic": 0.55,
        },
        strengths=("explain", "bulk", "factual"),
        weak=("design", "refactor"),
    ),
    "openai/glm-5.3-flash": Profile(
        model="openai/glm-5.3-flash",
        vision=True,
        ctx_window=200_000,
        cost_in=0.10,
        cost_out=0.34,
        latency=0,
        cap={
            "code_edit": 0.72, "code_gen": 0.70, "refactor": 0.58, "debug": 0.55,
            "review": 0.58, "design": 0.50, "explain": 0.80, "bulk": 0.75,
            "writing": 0.75, "factual": 0.82, "agentic": 0.68,
        },
        strengths=("explain", "factual", "bulk", "code_edit", "code_gen"),
        weak=("design",),
    ),
    "openai/qwen3.8-flash": Profile(
        model="openai/qwen3.8-flash",
        vision=False,
        ctx_window=262_000,
        cost_in=0.12,
        cost_out=0.38,
        latency=0,
        cap={
            "code_edit": 0.65, "code_gen": 0.62, "refactor": 0.58, "debug": 0.58,
            "review": 0.55, "design": 0.52, "explain": 0.72, "bulk": 0.78,
            "writing": 0.68, "factual": 0.72, "agentic": 0.60,
        },
        strengths=("bulk", "explain", "factual"),
        weak=("design",),
    ),
    "openai/deepseek-v4.1-flash": Profile(
        model="openai/deepseek-v4.1-flash",
        vision=False,   # text-only: must never receive an image turn
        ctx_window=128_000,
        cost_in=0.30,
        cost_out=1.21,
        latency=0,
        cap={
            "code_edit": 0.62, "code_gen": 0.60, "refactor": 0.58, "debug": 0.58,
            "review": 0.55, "design": 0.52, "explain": 0.72, "bulk": 0.82,
            "writing": 0.70, "factual": 0.78, "agentic": 0.60,
        },
        strengths=("bulk", "factual"),
        weak=("design", "review"),
    ),
    "openai/kimi-k2.7-code": Profile(
        model="openai/kimi-k2.7-code",
        vision=True,
        ctx_window=262_000,
        cost_in=0.77,
        cost_out=3.22,
        latency=1,
        cap={
            "code_edit": 0.86, "code_gen": 0.90, "refactor": 0.90, "debug": 0.88,
            "review": 0.84, "design": 0.70, "explain": 0.72, "bulk": 0.70,
            "writing": 0.70, "factual": 0.70, "agentic": 0.90,
        },
        strengths=("code_gen", "refactor", "debug", "agentic"),
        weak=("writing", "factual", "design"),
    ),
    "openai/glm-5.3": Profile(
        model="openai/glm-5.3",
        vision=False,
        ctx_window=200_000,
        cost_in=1.13,
        cost_out=3.54,
        latency=1,
        cap={
            "code_edit": 0.85, "code_gen": 0.85, "refactor": 0.84, "debug": 0.86,
            "review": 0.86, "design": 0.88, "explain": 0.85, "bulk": 0.80,
            "writing": 0.85, "factual": 0.85, "agentic": 0.88,
        },
        strengths=("design", "debug", "review", "agentic"),
    ),
    "openai/claude-haiku-4-5": Profile(
        model="openai/claude-haiku-4-5",
        vision=True,
        ctx_window=200_000,
        cost_in=1.01,
        cost_out=5.03,
        latency=1,
        cap={
            "code_edit": 0.80, "code_gen": 0.78, "refactor": 0.78, "debug": 0.76,
            "review": 0.78, "design": 0.76, "explain": 0.88, "writing": 0.92,
            "factual": 0.84, "bulk": 0.68, "agentic": 0.80,
        },
        strengths=("writing", "explain", "factual"),
        weak=("bulk",),
    ),
}

DIRECTIVE_RE = re.compile(r"\[\[\s*(escalate|cheap|use:\s*([A-Za-z0-9._\-/]+)|low|high)\s*\]\]", re.IGNORECASE)
ESCALATE_WORD = "ESCALATE"
# Escalate-word guarding (2026-09-25 deploy-stage incident): the bare word
# "ESCALATE" appears in pasted agent advice ("type ESCALATE and I'll route
# the debugging turn...") and router docs, where it is a SUGGESTION, not an
# instruction from the user. A bare substring hit must not escalate. It is
# only honoured when it is the user speaking in first person or a clearly
# imperative framing, and never when the ask is itself a question about the
# router or quotes/mentions the directive. The [[escalate]] bracket form is
# unambiguous and unaffected.
ESCALATE_NEGATION_RE = re.compile(
    r"(suggest|suggests|suggested|suggesting|recommend|type\s+escalate"
    r"|use\s+escalate|with\s+escalate|escalate\s+then|about\s+escalate"
    r"|of\s+escalate|for\s+escalate|an?\s+escalate|not\s+escalate|no\s+escalate"
    r"|whether\s+to\s+escalate|if\s+we\s+escalate|escalate\s+the\s+debugging"
    r"|\[\[escalate\]\]|escalate\b\s*(keyword|directive|word|command))",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------
# Task signatures
# --------------------------------------------------------------------------

KIND_KEYWORDS: dict[str, tuple[str, ...]] = {
    "refactor": (
        "refactor", "restructure", "reorganize", "rename across", "extract",
        "split into", "deduplicate", "decouple", "migrate", "port to",
    ),
    "debug": (
        "bug", "error", "traceback", "stack trace", "exception", "fails",
        "failing", "crash", "regression", "flaky", "not working", "broken",
        "why does", "why is", "root cause", "fix",
    ),
    "review": (
        "review", "audit", "critique", "inspect", "code smell", "find issues",
        "security review", "check this", "what's wrong with",
    ),
    "design": (
        "design", "architecture", "architect", "plan", "approach", "trade-off",
        "tradeoff", "schema", "data model", "adr", "strategy", "how should",
        "best way to", "scalab",
    ),
    "code_gen": (
        "implement", "create", "write", "add", "build", "scaffold", "generate",
        "new component", "new endpoint", "set up", "wire up",
    ),
    "bulk": (
        "every file", "all files", "across the repo", "across all", "bulk",
        "mass ", "script that", "for each", "rename all",
    ),
    "explain": (
        "explain", "walk me through", "how does", "what does", "summarize",
        "document", "teach me", "describe",
    ),
    "factual": ("what is", "what's", "who is", "define", "how many", "how much"),
}

CODE_FENCE_RE = re.compile(r"```")
# Fenced-payload stripping (2026-09-22 kit-generation incident): a
# kit-generation ask is long BECAUSE of its pasted template. Measuring the
# raw text inflated both ask_tokens (bar scale +3) and the importance read,
# which is exactly backwards for mechanical expansion work. The router must
# size the bar and read importance from the INSTRUCTION, not the payload.
CODE_FENCE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
PATH_RE = re.compile(r"[\w./\-]+\.(?:ts|tsx|js|jsx|py|go|rs|java|rb|cs|cpp|c|h|sql|yaml|yml|json|md|sh)\b")
ERROR_RE = re.compile(r"(traceback \(most recent|error:|exception:|at [\w.$]+\(|panic:)", re.IGNORECASE)
IMAGE_PART_TYPES = frozenset({"image_url", "image", "input_image"})


class Signature:
    """What the newest human ask actually is. Plain class: see the Profile note."""

    __slots__ = (
        "kind", "scale", "est_tokens", "has_image", "tool_fanout",
        "text", "directives", "use_model", "stall",
        "importance", "ask_tokens",
    )

    def __init__(
        self,
        kind: str,
        scale: int,
        est_tokens: int,
        has_image: bool,
        tool_fanout: int,
        text: str,
        directives: frozenset[str] = frozenset(),
        use_model: "str | None" = None,
        stall: int = 0,
        importance: int = 1,
        ask_tokens: int = 0,
    ) -> None:
        self.kind = kind
        self.scale = scale
        self.est_tokens = est_tokens
        self.has_image = has_image
        self.tool_fanout = tool_fanout
        self.text = text
        self.directives = directives
        self.use_model = use_model
        self.stall = stall
        # Importance: 0 = low, 1 = normal, 2 = high. Derived from the
        # MARGINAL ask (this message), not the accumulated thread - a trivial
        # follow-up in a long thread stays Low (see _detect_importance).
        self.importance = importance
        # Token size of the marginal ask alone; the bar uses this instead of
        # the whole-conversation est_tokens so long threads don't inflate the
        # capability bar for simple questions.
        self.ask_tokens = ask_tokens if ask_tokens else len(text) // 4


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, Mapping):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _has_image(content: Any) -> bool:
    if not isinstance(content, list):
        return False
    return any(
        isinstance(part, Mapping) and part.get("type") in IMAGE_PART_TYPES for part in content
    )


def _is_reminder_only(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    without = re.sub(r"<system-reminder>.*?</system-reminder>", "", stripped, flags=re.DOTALL | re.IGNORECASE)
    return not without.strip()


def _newest_human_ask(messages: Sequence[Mapping[str, Any]]) -> tuple[str, Any] | None:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        text = _message_text(message.get("content"))
        if _is_reminder_only(text):
            continue
        return text, message.get("content")
    return None


def _tool_calls(messages: Sequence[Mapping[str, Any]]) -> list[str]:
    """Fingerprints of the most recent assistant tool calls, oldest first."""
    prints: list[str] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            for call in calls:
                if isinstance(call, Mapping):
                    fn = call.get("function") or {}
                    prints.append(f"{fn.get('name')}:{str(fn.get('arguments'))[:200]}")
        elif isinstance(message.get("content"), list):
            for part in message["content"]:
                if isinstance(part, Mapping) and part.get("type") == "tool_use":
                    prints.append(f"{part.get('name')}:{str(part.get('input'))[:200]}")
    return prints


def _estimate_tokens(messages: Sequence[Mapping[str, Any]]) -> int:
    chars = 0
    for message in messages:
        content = message.get("content")
        chars += len(content) if isinstance(content, str) else len(_message_text(content))
        tools = message.get("tool_calls")
        if isinstance(tools, list):
            chars += len(str(tools))
    return chars // 4


LOW_ASK_RE = re.compile(
    r"^(thanks|thank you|thx|ok(?:ay)?|got it|nice|good|great|perfect|understood|pls|please"
    r"|hi|hello|hey|yes|no|nope|sure|done|go ahead|continue|keep going|next|proceed"
    r"|looks good|looks great|works fine|all good|that works|ship it|much appreciated"
    r"|awesome|exactly|agreed|correct|right|my bad|sorry)"
    r"([,!. ]+(thanks|thank you|thx|looks good|looks great|works fine|all good"
    r"|that works|ship it|much appreciated|awesome|nice|great|perfect|team))*"
    r"[.!?]?\s*$",
    re.IGNORECASE,
)
LOW_PHRASES = (
    "what does this", "what is this", "summarize", "summary", "tl;dr", "in short",
    "briefly", "quick question", "one thing", "remind me", "run it", "rerun",
    "run the", "try again", "revert", "undo", "commit", "push", "deploy",
    "add a comment", "add comments", "rename the", "rename this", "fix the typo",
    "remove the", "delete the", "move the", "bump the version", "update the readme",
    "same again", "again please",
)
HIGH_PHRASES = (
    "architect", "architecture", "design the", "design a", "spec", "specification",
    "plan the", "plan a", "roadmap", "trade-off", "tradeoffs", "evaluate",
    "refactor the", "migrate the", "migration", "rewrite the", "restructure",
    "audit the", "review the whole", "critical", "production-ready",
    "think through", "step by step plan", "break down",
)


def _detect_importance(ask: str, kind: str, directives: frozenset[str], est_tokens: int) -> int:
    """0 = Low, 1 = Normal, 2 = High, from the MARGINAL ask only.

    Two goals, in order: never send real work to a weak model, and never pay
    a premium model for a trivial follow-up. Manual [[low]] / [[high]] win
    outright; the heuristic must stay conservative (prefer Normal on doubt).
    """
    lowered = ask.lower().strip()
    if "low" in directives:
        return 0
    if "high" in directives:
        return 2
    if "escalate" in directives:
        return 2
    # One-liners and acknowledgements: the answer is worth cents, not HKD.
    if LOW_ASK_RE.match(lowered):
        return 0
    # Phrase path: a short ask that STARTS with a follow-up command
    # ("summarize this", "run the tests", "rename the field"). The ask must
    # be short AND start with the phrase: "commit everything with a full
    # paragraph explaining each change" starts with the word but carries
    # real instruction payload, so the phrase window is bounded.
    low_hits = sum(1 for phrase in LOW_PHRASES if lowered.startswith(phrase))
    if low_hits >= 1 and len(lowered) < 80:
        return 0
    # Spec / architecture / plan phrasing, or a code-bearing multi-constraint
    # ask: this is where a strong model pays for itself. Word boundaries:
    # "architect" must not double-count inside "architecture".
    high_hits = sum(
        1 for phrase in HIGH_PHRASES
        if re.search(r"\b" + re.escape(phrase) + r"\b", lowered)
    )
    if high_hits >= 2:
        return 2
    if high_hits >= 1 and (kind in {"design", "refactor", "review"} or len(lowered) > 400):
        return 2
    return 1


# Quoted-narrative guard (2026-09-25 deploy-stage incident): when a user
# pastes an agent's advice or a review summary, the quoted sentences carry
# debug/review vocabulary ("fails", "fix", "found 3 bugs") that describes
# PAST or HYPOTHETICAL work, not the ask. Two guards:
#   1. Attribution strip: sentences that merely report what an agent said or
#      did (quoted or attributed speech) are removed before kind detection.
#   2. Interrogative tail: if the ask ENDS in a short question, the question
#      is the ask; narrative vocabulary above it is context. Error-pastebins
#      are still honoured (ERROR_RE wins) because a real traceback pasted
#      into a question-shaped ask still needs debugging.
QUOTE_ATTRIBUTION_RE = re.compile(
    r"(?:^|\n)\s*(?:>?|\"|“|')\s*"
    r"(?:cursor(?:'s)?|the\s+(?:router|agent|subagent|review(?:er)?)|it|this|they|he|she)\b"
    r"[^:;\n]{0,80}(?:said|wrote|replied|answered|advised|suggested|recommends?|noted|found|added|flagged|reported)"
    r"[^\n]*",
    re.IGNORECASE,
)
INTERROGATIVE_TAIL_RE = re.compile(r"[^\n?]{0,200}\?\s*$")
# Report-shape guard: a pasted agent report ("Follow-up review findings",
# "Current state", "Next task when you resume") is CONTEXT. When it ends in
# a short question, the question is the ask and the narrative's debug/review
# vocabulary describes past work, not this request. Keyword scoring then
# reads only the question tail. Real error pastes bypass this entirely:
# ERROR_RE is checked before the guard, so "Traceback... what's wrong?"
# still debugs.
REPORT_SHAPE_RE = re.compile(
    r"(follow-up review|review of|findings|fixed and committed|all fixed"
    r"|current state|next task|blocked on|when you resume|status report"
    r"|summary of|here is what|here's what|what was done)",
    re.IGNORECASE,
)


def _strip_quoted_narrative(ask: str) -> str:
    """Remove attributed/quoted narration lines for KIND detection only.

    The directive scan still runs on the raw text (a directive inside a
    quote is still usually meant), but kind and importance read the ask
    minus the narration. Returns the text unchanged when nothing matches.
    """
    return QUOTE_ATTRIBUTION_RE.sub("\n", ask)


def _detect_kind(ask: str, est_tokens: int, has_image: bool, messages: Sequence[Mapping[str, Any]] | None = None) -> str:
    instruction = _strip_quoted_narrative(ask)
    # A real pasted error ALWAYS debugs. Checked on BOTH the marginal ask and
    # the recent thread: a user who pastes a traceback and then asks "what am
    # I doing wrong?" has the error one turn back, and that is still a debug
    # turn (2026-09-25 regression pair).
    error_context = ask if messages is None else ask + "\n" + "\n".join(
        _message_text(message.get("content")) for message in messages[-4:]
    )
    if ERROR_RE.search(error_context):
        return "debug"
    # Report-shape guard: pasted agent report + short question tail. The
    # report's vocabulary (bugs, deploy, review) is about past work; only
    # the question tail is the ask.
    tail_match = INTERROGATIVE_TAIL_RE.search(instruction)
    if tail_match and REPORT_SHAPE_RE.search(instruction):
        instruction = tail_match.group(0)
    scores: dict[str, float] = {}
    lowered = instruction.lower()
    for kind, words in KIND_KEYWORDS.items():
        hits = sum(1 for word in words if word in lowered)
        if hits:
            scores[kind] = hits + len(words) * 0.01
    if CODE_FENCE_RE.search(ask) and PATH_RE.search(ask) and "code_gen" not in scores:
        scores["code_edit"] = scores.get("code_edit", 0) + 1.0
    if not scores:
        # No signal: short asks are lookups, long ones are work on code.
        if est_tokens < 120:
            return "factual"
        return "code_edit" if PATH_RE.search(ask) or has_image else "agentic"
    order = ("debug", "refactor", "review", "design", "bulk", "code_gen", "explain", "factual", "code_edit")
    return max(order, key=lambda kind: (scores.get(kind, 0.0), -order.index(kind)))


def _detect_stall(messages: Sequence[Mapping[str, Any]]) -> int:
    """Detect a stuck loop WITHOUT penalising healthy iterative work (review M7).

    The previous version counted occurrences of the newest fingerprint
    anywhere in the last six calls, so the canonical TDD rhythm
    test -> edit -> test -> edit -> test - different calls making real
    progress between them - read as a stall. A stall now requires a
    CONSECUTIVE trailing run: the last k >= 3 tool-call fingerprints must all
    be identical, with no different call between them. The reported count is
    the length of that run, so the verdict weight stays proportional."""
    prints = _tool_calls(messages)[-6:]
    if len(prints) < 3:
        return 0
    newest = prints[-1]
    run_length = 0
    for item in reversed(prints):
        if item != newest:
            break
        run_length += 1
    return run_length if run_length >= 3 else 0


def build_signature(messages: Sequence[Mapping[str, Any]]) -> Signature | None:
    ask_pair = _newest_human_ask(messages)
    if ask_pair is None:
        return None
    ask, raw_content = ask_pair
    est_tokens = _estimate_tokens(messages)
    has_image = _has_image(raw_content) or any(
        _has_image(message.get("content")) for message in messages[-4:]
    )

    scale = 0
    if est_tokens > 12_000:
        scale += 1
    if est_tokens > 40_000:
        scale += 1
    if len(messages) > 20:
        scale += 1
    scale = min(scale, MAX_SCALE)

    directives: set[str] = set()
    use_model: str | None = None
    for match in DIRECTIVE_RE.finditer(ask):
        token = match.group(1).strip()
        if token.lower().startswith("use:"):
            use_model = match.group(2)
            directives.add("use")
        else:
            directives.add(token.lower())
    if ESCALATE_WORD in ask and not ESCALATE_NEGATION_RE.search(ask):
        directives.add("escalate")
    if "[[cheap]]" in ask.lower():
        directives.add("cheap")

    kind = _detect_kind(ask, est_tokens, has_image, messages)
    # The instruction text with fenced payload stripped AND quoted narrative
    # removed. Kit-generation asks are long because of a pasted template, not
    # because the instruction is complex; agent-advice pastes are heavy with
    # debug/review vocabulary that is not the ask. Both skew importance and
    # the bar when read raw (2026-09-22 kit incident, 2026-09-25 deploy
    # incident). The whole-thread est_tokens is untouched, so context
    # fitting and headroom still see the true size.
    instruction = CODE_FENCE_BLOCK_RE.sub(" ", _strip_quoted_narrative(ask))
    return Signature(
        kind=kind,
        scale=scale,
        est_tokens=est_tokens,
        has_image=has_image,
        tool_fanout=len(_tool_calls(messages)),
        text=ask,
        directives=frozenset(directives),
        use_model=use_model,
        stall=_detect_stall(messages),
        importance=_detect_importance(instruction, kind, frozenset(directives), est_tokens),
        ask_tokens=len(instruction) // 4,
    )


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

class Candidate:
    """One scored option. Plain class: see the Profile note."""

    __slots__ = ("profile", "capability", "cost_score", "trust", "headroom", "latency_score", "utility")

    def __init__(
        self,
        profile: Profile,
        capability: float,
        cost_score: float,
        trust: float,
        headroom: float,
        latency_score: float,
    ) -> None:
        self.profile = profile
        self.capability = capability
        self.cost_score = cost_score
        self.trust = trust
        self.headroom = headroom
        self.latency_score = latency_score
        self.utility = 0.0


def _blended_cost(profile: Profile) -> float:
    # Output tokens are the expensive half of an agent turn; weight them up.
    return profile.cost_in * 0.45 + profile.cost_out * 0.55


# HKD per USD for the spend estimate in the decision log. Rounded input the
# operator supplied (2026-09-21).
HKD_PER_USD = 7.8


def _estimate_cost_hkd(signature: Any, profile: Profile) -> float:
    """Rough pre-call spend estimate for the decision log / dashboard.

    Input side = the whole thread re-sent this turn (est_tokens). Output
    side is estimated from the marginal ask: trivial asks get short answers,
    big asks get long ones. It is an estimate for visibility, not billing -
    the true number arrives with usage data if the upstream reports it.
    """
    est_out = min(4000, max(64, signature.ask_tokens * 3))
    usd = (signature.est_tokens / 1_000_000) * profile.cost_in + (est_out / 1_000_000) * profile.cost_out
    return usd * HKD_PER_USD


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if high - low < 1e-9:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


class PolicyState:
    """In-memory session pins and the learned trust table (per worker)."""

    def __init__(self) -> None:
        self.sessions: dict[str, dict[str, Any]] = {}
        self.trust: dict[tuple[str, str], float] = {}
        # Expired-lease instances (model|deadline) already swept, so the
        # orphan cleanup runs once per lease instance instead of suppressing
        # stickiness forever for sessions whose fresh choice happens to
        # equal the expired model.
        self.cleaned_leases: set[str] = set()
        # The last live lease this process served a request under, as
        # {"model": ..., "expires_ts": ...} or None. The moment it reads as
        # expired, sessions pinned to its model are dropped (review B5).
        self.last_lease: dict[str, Any] | None = None
        # Per-session per-(model, kind) spend alarm state.
        # kind_alarms[session_key][(model, kind)] = {"spend_hkd": x, "ts": t}
        # Triggered when spend_hkd >= KIND_ALARM_HKD. A global alarm_epoch is
        # read from the control plane; stale epochs clear in-memory alarms.
        self.kind_alarms: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
        self.alarm_epoch: int = 0

    def session_spend(self, session_key: str | None) -> float:
        """Estimated cumulative spend for this session from the decision log.

        Kept in RAM because it is a running aggregate of per-turn estimates
        already written to the log; recomputing from disk on every request
        would be wasteful. Sessions are TTL-pruned, so the total memory cost
        is bounded (SESSION_CAP × SESSION_TTL).
        """
        if not session_key:
            return 0.0
        return float(self.sessions.get(session_key, {}).get("spend_hkd", 0.0))

    def session_key(self, messages: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]) -> str | None:
        explicit = metadata.get("session_id") or metadata.get("litellm_session_id")
        if isinstance(explicit, str) and explicit:
            return explicit
        first_user = next((m for m in messages if m.get("role") == "user"), None)
        if first_user is None:
            return None
        seed = _message_text(first_user.get("content"))[:400]
        return hashlib.sha256(seed.encode("utf-8", "replace")).hexdigest()[:24]

    def last_ask_of(self, session_key: str | None) -> str:
        """The newest human ask this session served, RAM-only (review B3's
        skip-path observer needs it for topical continuity; m9 limitation:
        the session key is the first-message hash, documented as weak)."""
        if not session_key:
            return ""
        record = self.sessions.get(session_key)
        if not record:
            return ""
        return str(record.get("ask") or "")

    def drop_leases(self, model: str, expires: Any = None) -> None:
        """Lease-expiry orphan cleanup (review B5). A lease pins every session
        it touches to one model; when it expires, those sessions must
        re-decide fresh instead of holding the model via stickiness for up
        to SESSION_TTL. Idempotent per (model, deadline) instance, so a new
        lease on the same model still gets its own sweep when it ends."""
        fingerprint = f"{model}|{expires}"
        if fingerprint in self.cleaned_leases:
            return
        self.cleaned_leases.add(fingerprint)
        for key, value in list(self.sessions.items()):
            if value.get("model") == model:
                self.sessions.pop(key, None)

    def prune(self) -> None:
        now = time.time()
        if len(self.sessions) > SESSION_CAP:
            for key, value in sorted(self.sessions.items(), key=lambda item: item[1]["ts"])[: len(self.sessions) - SESSION_CAP]:
                self.sessions.pop(key, None)
        for key, value in list(self.sessions.items()):
            if now - value["ts"] > SESSION_TTL:
                self.sessions.pop(key, None)
        # Alarms also TTL to bound memory; a long-stale session's alarms
        # are useless and would otherwise leak under heavy use.
        for key, value in list(self.kind_alarms.items()):
            for alarm_key, alarm in list(value.items()):
                if now - alarm["ts"] > SESSION_TTL:
                    value.pop(alarm_key, None)
            if not value:
                self.kind_alarms.pop(key, None)

    def trust_of(self, kind: str, model: str) -> float:
        return self.trust.get((kind, model), 0.0)

    def effective_trust_of(self, kind: str, model: str) -> float:
        """Runtime trust plus the persisted (learned) value, clamped once.

        Additive by construction (review M8's fix): trust = clamp(runtime +
        learned, +/-TRUST_CEILING) at the single point of use. With zero
        learned influence -- no store, learning disabled, empty tables --
        this is exactly the runtime value, which is what makes the fail-open
        contract (review B1) hold byte-for-byte: the presence of a control
        store must not itself move a decision."""
        runtime = self.trust.get((kind, model), 0.0)
        if CONTROL is None:
            return runtime
        try:
            persisted = CONTROL.persisted_trust(kind, model)
        except Exception:  # noqa: BLE001 - optional input
            return runtime
        return max(TRUST_FLOOR, min(TRUST_CEILING, runtime + persisted))

    def adjust(self, kind: str, model: str, delta: float) -> None:
        value = self.trust.get((kind, model), 0.0) + delta
        self.trust[(kind, model)] = max(TRUST_FLOOR, min(TRUST_CEILING, value))


STATE = PolicyState()


def _cascade_excluded(known: list[Profile], previous: Mapping[str, Any] | None, now: float | None = None) -> tuple[set[str], dict[str, Any]]:
    """Models to skip this turn because they failed upstream recently.

    Returns (excluded_model_set, telemetry_dict). The set is empty when the
    failure hook is absent, learning is disabled, or no recent failures are
    recorded. A previous-model failure also docks that model's trust for the
    kind of work it just failed on, so the next fresh decision naturally
    prefers a different model.
    """
    excluded: set[str] = set()
    info: dict[str, Any] = {"excluded": [], "window_s": CASCADE_FAILURE_WINDOW_S}
    if not _learn_enabled() or _failure_hook is None:
        return excluded, info
    now = time.time() if now is None else now

    # Any fleet model with a failure inside the window is temporarily avoided.
    for profile in known:
        ts = _failure_hook.last_failure_ts(profile.model)
        if ts is not None and now - ts < CASCADE_FAILURE_WINDOW_S:
            excluded.add(profile.model)
            info["excluded"].append({
                "model": profile.model,
                "failure_age_s": round(now - ts, 3),
            })

    # The previous model for this session gets an extra trust penalty on top
    # of exclusion, so even if it is the only capable model it is ranked down.
    if previous and isinstance(previous.get("model"), str) and isinstance(previous.get("kind"), str):
        ts = _failure_hook.last_failure_ts(previous["model"])
        if ts is not None and now - ts < CASCADE_FAILURE_WINDOW_S:
            STATE.adjust(previous["kind"], previous["model"], CASCADE_TRUST_PENALTY)
            info["penalized_model"] = previous["model"]
            info["penalty"] = CASCADE_TRUST_PENALTY

    return excluded, info


def _log(record: Mapping[str, Any]) -> None:
    if not LOG_PATH:
        return
    try:
        directory = os.path.dirname(LOG_PATH)
        if directory:
            os.makedirs(directory, exist_ok=True)
        if os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > LOG_MAX_BYTES:
            os.replace(LOG_PATH, LOG_PATH + ".1")
        with open(LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass  # logging must never affect a request


def decide(
    signature: Signature,
    known: list[Profile],
    trust_of,
    weights: tuple[float, float, float, float] = (W_COST, W_TRUST, W_HEADROOM, W_LATENCY),
    bar_bias: float = 0.0,
    laya_scores: Mapping[str, float] | None = None,
    laya_weight: float = 0.0,
) -> tuple[Profile | None, list[Candidate], str, float]:
    """Pure selection core. Returns (winner, scored candidates, gate, bar).

    `weights` is the (cost, trust, headroom, latency) tuple, resolved by the
    caller from env + control plane. `bar_bias` is a signed learned offset
    already clamped by the store; positive makes this kind harder.
    `laya_scores` is an optional per-kind confidence map from the shadow
    scorer; it is blended into utility when `laya_weight` > 0.
    """
    if not known:
        return None, [], "no-known-models", 0.0
    w_cost, w_trust, w_headroom, w_latency = weights
    laya_score = 0.0
    if laya_scores and laya_weight:
        laya_score = max(0.0, min(1.0, float(laya_scores.get(signature.kind, 0.0))))

    # 1. hard gates -------------------------------------------------------
    gated = [
        profile for profile in known
        if (not signature.has_image or profile.vision)
        and signature.est_tokens < profile.ctx_window * 0.9
    ]
    gate_reason = "ok"
    if not gated:
        # Nothing fits every gate: keep the closest by window, prefer vision.
        gated = sorted(
            known,
            key=lambda profile: (profile.vision if signature.has_image else True, profile.ctx_window),
            reverse=True,
        )[:1]
        gate_reason = "gate_relaxed"

    # 2. capability bar ---------------------------------------------------
    # Bar inputs are MARGINAL-ask signals: ask-scale (size of this message,
    # not the accumulated thread) and importance tier. A trivial follow-up in
    # a 130k-token thread must still clear a flash model's bar.
    ask_scale = min(MAX_SCALE, (signature.ask_tokens or 1) // 400)
    bar = BASE_BAR.get(signature.kind, 0.6) + SCALE_BAR_STEP * ask_scale + bar_bias
    bar += IMPORTANCE_OFFSETS[signature.importance]
    if "escalate" in signature.directives or signature.stall:
        bar += SCALE_BAR_STEP * 2
    if "cheap" in signature.directives:
        bar -= SCALE_BAR_STEP * 2
    bar = max(0.35, min(0.98, bar))

    capable = [profile for profile in gated if profile.cap.get(signature.kind, 0.5) >= bar]
    if not capable:
        best = max(profile.cap.get(signature.kind, 0.5) for profile in gated)
        capable = [profile for profile in gated if profile.cap.get(signature.kind, 0.5) >= best - 1e-9]
        gate_reason = "bar_relaxed" if gate_reason == "ok" else gate_reason

    # 3. utility ----------------------------------------------------------
    costs = _normalize([_blended_cost(profile) for profile in capable])
    latencies = _normalize([float(profile.latency) for profile in capable])
    scored: list[Candidate] = []
    for position, profile in enumerate(capable):
        capability = profile.cap.get(signature.kind, 0.5)
        headroom = min(1.0, max(0.0, 1.0 - signature.est_tokens / max(profile.ctx_window, 1)))
        candidate = Candidate(
            profile=profile,
            capability=capability,
            cost_score=costs[position],
            trust=trust_of(signature.kind, profile.model),
            headroom=headroom,
            latency_score=latencies[position],
        )
        candidate.utility = (
            capability
            + w_trust * candidate.trust
            + w_headroom * headroom
            - w_cost * candidate.cost_score
            - w_latency * candidate.latency_score
            + laya_weight * laya_score
        )
        scored.append(candidate)

    scored.sort(key=lambda item: (-item.utility, _blended_cost(item.profile)))
    return scored[0].profile, scored, gate_reason, bar


class CursorAutoPolicy:
    """RoutingPlugin implementation. One instance per proxy process."""

    async def run(self, context: Any) -> Any:
        try:
            return self._run(context)
        except Exception as error:  # noqa: BLE001 - fail open, never break a request
            try:
                context.signals["policy_error"] = f"{type(error).__name__}: {error}"
            except Exception:
                pass
            return context

    def _run(self, context: Any) -> Any:
        if DISABLED:
            context.signals["policy"] = "disabled"
            return context

        candidates = list(getattr(context, "candidate_models", []) or [])
        # A model string appears once per deployment (glm-5.3 is on both keys),
        # so dedupe: the policy picks a model, and litellm load-balances and
        # fails over between that model's deployments by itself.
        known = list({model: PROFILES[model] for model in candidates if model in PROFILES}.values())
        if len(known) < 2:
            # Not the policy-governed group: either an unknown model, or one of
            # the direct escape-hatch groups the caller picked on purpose.
            _observe_skip_path(context, candidates)
            context.signals["policy"] = "skipped-not-fleet-group"
            return context

        messages = list(getattr(context, "structured_messages", []) or []) or list(
            getattr(context, "raw_messages", []) or []
        )
        metadata = dict(getattr(context, "metadata", {}) or {})
        signature = build_signature(messages)
        if signature is None:
            context.signals["policy"] = "skipped-no-ask"
            return context

        STATE.prune()
        session_key = STATE.session_key(messages, metadata)

        # ---- control plane: lease (bounded pin) --------------------------
        # A live lease overrides selection for the bounded window. Gates still
        # apply: a lease never sends an image to a text-only model or an
        # oversized prompt into a small window - it fails open to normal
        # selection for that request instead, and reports lease_bypassed.
        #
        # Lease-expiry orphan cleanup (review B5) runs BEFORE the session
        # record is read: a lease pins every session it touches to one
        # model, and when it expires those sessions must re-decide fresh
        # instead of holding the model via stickiness for up to SESSION_TTL.
        # The sweep is idempotent per lease instance (model + deadline).
        if CONTROL is not None:
            lease_now = None
            try:
                lease_now = CONTROL.active_lease()
            except Exception:  # noqa: BLE001
                lease_now = None
            previous_lease = STATE.last_lease
            STATE.last_lease = dict(lease_now) if lease_now is not None else None
            if lease_now is None and isinstance(previous_lease, Mapping) \
                    and isinstance(previous_lease.get("model"), str):
                STATE.drop_leases(previous_lease["model"], previous_lease.get("expires_ts"))

        previous = STATE.sessions.get(session_key) if session_key else None

        # ---- stale-ask execution demotion --------------------------------
        # The newest human ask has driven >= EXEC_PHASE_TURNS turns with tool
        # activity: the planning decision is made, the loop is now executing
        # it. Re-signature the turn as agentic/Normal with no ask-scale so
        # flash models clear the bar. Fresh asks, directives and non-planning
        # kinds are untouched; stall/cascade re-escalate if flash struggles.
        exec_demoted = False
        if (
            previous
            and not signature.directives
            and signature.kind in EXEC_PHASE_KINDS
            and signature.tool_fanout > 0
            and int(previous.get("turns", 0) or 0) >= EXEC_PHASE_TURNS
            and previous.get("ask") == signature.text
        ):
            signature = Signature(
                kind=EXEC_PHASE_KIND,
                scale=signature.scale,
                est_tokens=signature.est_tokens,
                has_image=signature.has_image,
                tool_fanout=signature.tool_fanout,
                text=signature.text,
                directives=signature.directives,
                use_model=signature.use_model,
                stall=signature.stall,
                importance=min(signature.importance, 1),
                ask_tokens=min(signature.ask_tokens, 399),  # ask_scale 0
            )
            exec_demoted = True

        lease = None
        lease_profile: Profile | None = None
        if CONTROL is not None:
            try:
                lease = CONTROL.active_lease()
            except Exception:  # noqa: BLE001
                lease = None
        if lease is not None:
            model = lease.get("model")
            candidate_profile = PROFILES.get(model) if isinstance(model, str) else None
            if candidate_profile is not None and candidate_profile in known:
                fits = (not signature.has_image or candidate_profile.vision) and (
                    signature.est_tokens < candidate_profile.ctx_window * 0.9
                )
                if fits:
                    lease_profile = candidate_profile

        # ---- learned bar bias (phase 1/2 tables, read-only here) ----------
        # The ControlStore reads learned.json (single writer: this proxy
        # process, via the learner) and clamps at read time; the learner's
        # own flush keeps the file bounded. No bias is applied unless BOTH
        # the env switch and control.json enable learning (C8).
        bar_bias = 0.0
        if _learn_enabled():
            try:
                bar_bias = CONTROL.kind_bar_bias(signature.kind) + CONTROL.phrase_bias(signature.text)
            except Exception:  # noqa: BLE001 - biases are optional
                bar_bias = 0.0

        weights = _effective_weights()

        # ---- cascade-lite (one-turn failure escalation) -------------------
        # If the previous model (or any fleet model) failed upstream inside
        # the window, drop it from consideration this turn and dock its
        # trust. Explicit [[use:...]] and live leases override this.
        cascade_excluded, cascade_info = _cascade_excluded(known, previous)
        cascade_fallback = False

        # ---- laya shadow scorer (telemetry-only by default) ---------------
        # Enqueue the marginal ask for background scoring. The score is read
        # from a RAM cache; on a cold miss it is simply absent for this turn.
        # Weight defaults to 0, so a missing/broken scorer is byte-identical.
        laya_scores = None
        laya_weight = 0.0
        if _learn_enabled() and _laya_scorer is not None:
            try:
                _laya_scorer.maybe_enqueue(signature.text, signature.kind)
                laya_weight = _laya_weight()
                if laya_weight:
                    laya_scores = _laya_scorer.get_scores(signature.text)
            except Exception:  # noqa: BLE001
                pass

        # Feedback: the caller escalated, has re-asked, or the loop is stuck.
        if previous:
            if "escalate" in signature.directives:
                STATE.adjust(previous["kind"], previous["model"], -0.20)
            if signature.stall:
                STATE.adjust(previous["kind"], previous["model"], -0.15)
            elif previous.get("turns", 0) >= 3 and not signature.stall:
                STATE.adjust(previous["kind"], previous["model"], 0.03)

        # ---- learner hooks (phase 1): verdicts + reset intents ------------
        # Fire only when learning is enabled (C8). Every hook is wrapped and
        # counted; the request path never sees a learner failure (C1).
        _ensure_background_tasks()
        if _learn_enabled():
            try:
                learner = _get_learner()
                if learner is not None:
                    control_path = os.environ.get(
                        "CONTROL_PATH",
                        os.path.join(
                            os.environ.get("POLICY_STATE_DIR")
                            or ("/app/logs" if os.path.isdir("/app/logs") else tempfile.gettempdir()),
                            "control.json",
                        ),
                    )
                    learner.poll_reset_intents(control_path)
                    prev_record = None
                    if previous:
                        prev_record = {
                            "ask": str(previous.get("ask") or ""),
                            "directives": (),
                            "stall": 0,
                            "kind": previous["kind"],
                            "model": previous["model"],
                            "turns": previous.get("turns", 0),
                        }
                    current_record = {
                        "ask": signature.text,
                        "directives": signature.directives,
                        "stall": signature.stall,
                        "kind": signature.kind,
                        "model": "",
                        "turns": (previous.get("turns", 0) + 1) if previous else 1,
                    }
                    learner.observe_decision(prev_record, current_record, session_key)
                    # Phase 2: queue the newest human ask for extraction
                    # (RAM only, never persisted raw - plan C5).
                    learner.enqueue(signature.text, signature.kind, "",
                                    current_record["turns"])
                    learner.flush()
            except Exception:  # noqa: BLE001 - learning must never touch a request
                pass
            _record_learner_state()

        # Bar preview for the pin re-check: the same formula decide() uses
        # (base + ask-scale + importance, before learned bar_bias), computed
        # early so the pin logic can drop a held model that no longer clears
        # the CURRENT bar. Duplicating the arithmetic is deliberate: pulling
        # this out of decide() would change decide()'s contract mid-stream;
        # keeping the two formulas identical is enforced by tests.
        bar_preview = BASE_BAR.get(signature.kind, 0.6) + SCALE_BAR_STEP * min(
            MAX_SCALE, (signature.ask_tokens or 1) // 400
        ) + IMPORTANCE_OFFSETS[signature.importance]
        if "escalate" in signature.directives or signature.stall:
            bar_preview += SCALE_BAR_STEP * 2
        if "cheap" in signature.directives:
            bar_preview -= SCALE_BAR_STEP * 2

        pinned: Profile | None = None
        pin_held: "str | None" = None   # "grace" | "loyalty" when a pin survives
        if signature.use_model:
            # Explicit intent: absolute, but never past the hard gates. A
            # directive pinning a text-only model for an image turn, or a
            # small-window model for an oversized prompt, must fail open to
            # normal selection instead of shipping a request that cannot be
            # served.
            pinned = next((profile for profile in known if profile.model.endswith(signature.use_model)), None)
            if pinned is not None and not (
                (not signature.has_image or pinned.vision)
                and signature.est_tokens < pinned.ctx_window * 0.9
            ):
                pinned = None
        if pinned is None and previous and previous["kind"] == signature.kind and not signature.directives:
            held = PROFILES.get(previous["model"])
            if held in known and (not signature.has_image or held.vision) and signature.est_tokens < held.ctx_window * 0.9:
                turns_held = int(previous.get("turns", 1) or 1)
                # Stickiness cost guard: a PREMIUM model held by session
                # stickiness must not serve a Low-importance follow-up. The
                # user's principle: strong on spec/planning, cheap on
                # execution and daily asks. Breaking the pin costs one cold
                # cache read on the cheap model; keeping it costs the premium
                # rate on every trivial turn. Explicit intent ([[use:]]) and
                # the directive path are never overridden - only the implicit
                # same-kind pin is.
                if signature.importance == 0 and _blended_cost(held) > PREMIUM_COST:
                    pinned = None
                elif held.model in cascade_excluded:
                    pinned = None
                elif turns_held > PIN_PREMIUM_CEIL and _blended_cost(held) > PREMIUM_COST:
                    # Premium run ceiling: after PIN_PREMIUM_CEIL consecutive
                    # turns the held premium model must re-win a FRESH
                    # decision - grace and the loyalty bonus are void. If the
                    # bar genuinely requires a premium model, it re-wins and
                    # the run continues re-justified. If it was only winning
                    # because every cheap challenger sat below an inflated
                    # bar, the fresh decision swaps to the cheap model. Cheap
                    # pins are exempt: keeping them IS the goal, and explicit
                    # [[use:]] pins never reach this branch at all.
                    pinned = None
                elif turns_held <= PIN_GRACE_TURNS:
                    # Grace window: the first turns of a pin hold
                    # unconditionally. The cache really is warm here, and the
                    # early turns of a task are where continuity pays.
                    pinned = held
                    pin_held = "grace"
                elif held.cap.get(signature.kind, 0.5) < bar_preview:
                    # Bar re-check every turn, not only at first pick. A
                    # High-importance or spec-like turn arriving mid-pin must
                    # not be served by a model below the CURRENT bar, even
                    # inside a sticky task.
                    pinned = None
                else:
                    # After grace the pin COMPETES: a loyalty bonus decays
                    # linearly from PIN_BONUS_START toward PIN_BONUS_FLOOR,
                    # and the held model keeps the task only while its
                    # utility + bonus still beats the best challenger. This
                    # is the fix for the 44-turn premium lock-in: the pin
                    # buys cache warmth, it does not buy the session.
                    pinned = held
                    pin_held = "loyalty"

        # ---- budget cap (cost-of-last-resort guard) ----------------------
        # If a session's estimated spend has crossed the configured cap, all
        # subsequent cursor-auto turns are forced to the cheapest capable
        # model unless the caller explicitly escalates or pins. This is the
        # only control that would have stopped the user's 162 HKD hour:
        # importance routing already chose the right model per turn, but the
        # sheer volume of ~130k-token turns on premium models still added up.
        # ---- per-(model, kind) spend alarm -------------------------------
        # A single model-kind pair burning >= KIND_ALARM_HKD in one session
        # is treated like a runaway. The alarm is stored in RAM per session;
        # the control plane can broadcast an alarm_epoch bump to clear all
        # proxies' stale alarms at once (e.g. via MCP router_kind_alarm_clear).
        # The alarm only fires when the CURRENT turn is the same kind as the
        # runaway pair; a debug turn after a code_gen runaway should still be
        # routed normally so the user can investigate.
        kind_alarm_active = False
        kind_alarm_model: str | None = None
        if (
            session_key
            and previous
            and isinstance(previous.get("model"), str)
            and isinstance(previous.get("kind"), str)
            and previous["kind"] == signature.kind
        ):
            previous_pair = (previous["model"], previous["kind"])
            session_alarms = STATE.kind_alarms.get(session_key, {})
            alarm_record = session_alarms.get(previous_pair)
            if (
                alarm_record is not None
                and alarm_record.get("epoch", 0) >= STATE.alarm_epoch
                and float(alarm_record.get("spend_hkd", 0.0)) >= KIND_ALARM_HKD
                and time.time() - alarm_record["ts"] < KIND_ALARM_TTL_S
            ):
                kind_alarm_active = True
                kind_alarm_model = previous["model"]

        # Sync alarm_epoch from the control plane if it is configured. A
        # manual clear bumps the epoch; any in-memory alarm with a lower
        # epoch is immediately forgotten. No new persistent file is needed.
        if CONTROL is not None:
            try:
                live_epoch = CONTROL.alarm_epoch()
                if isinstance(live_epoch, int) and live_epoch > STATE.alarm_epoch:
                    STATE.alarm_epoch = live_epoch
            except Exception:  # noqa: BLE001
                pass

        budget = None
        if CONTROL is not None:
            try:
                budget = CONTROL.budget_hkd()
            except Exception:  # noqa: BLE001
                budget = None
        session_spend = STATE.session_spend(session_key)
        budget_breached = (
            budget is not None
            and session_spend >= budget
            and not ({"escalate", "high", "use"} & signature.directives)
            and not (lease_profile is not None and signature.use_model)
        )

        reason = "fresh"
        gate_reason = "ok"
        bar = BASE_BAR.get(signature.kind, 0.6) + SCALE_BAR_STEP * min(
            MAX_SCALE, (signature.ask_tokens or 1) // 400
        ) + IMPORTANCE_OFFSETS[signature.importance]
        scored: list[Candidate] = []
        # B5 precedence: in-band [[use:...]] beats an active lease; an active
        # lease beats session stickiness. The directive wins even when a
        # lease is live because it is the more specific, transcript-visible
        # intent for exactly this request.
        # Budget breach is a safety override: it beats pinned-cache (but NOT
        # explicit [[use:]] or a live lease), so a long session cannot keep
        # paying premium rates on every trivial follow-up once the cap is hit.
        if budget_breached:
            pinned = None
            pin_held = None
        if lease_profile is not None and signature.use_model and \
                pinned is not None and pinned is not lease_profile:
            lease_profile = None  # directive overrides the lease this turn
        if lease_profile is not None:
            winner = lease_profile
            reason = "lease"
            for profile in known:
                scored.append(
                    Candidate(profile=profile, capability=profile.cap.get(signature.kind, 0.5),
                              cost_score=0.0, trust=STATE.trust_of(signature.kind, profile.model),
                              headroom=0.0, latency_score=0.0)
                )
        elif pinned is not None and (signature.use_model or pin_held == "grace"):
            # Absolute holds: explicit [[use:]] and the pin's grace window.
            winner = pinned
            reason = "pinned-directive" if signature.use_model else "pinned-grace"
            for profile in known:
                scored.append(
                    Candidate(profile=profile, capability=profile.cap.get(signature.kind, 0.5),
                              cost_score=0.0, trust=STATE.trust_of(signature.kind, profile.model),
                              headroom=0.0, latency_score=0.0)
                )
        elif pinned is not None and pin_held == "loyalty":
            # The pin competes. Run the normal decision over the fleet, then
            # compare the challenger's utility against the held model's
            # utility + a decaying loyalty bonus. The held model stays only
            # while it is still the better deal WITH the bonus.
            known_for_decision = [profile for profile in known if profile.model not in cascade_excluded]
            if not known_for_decision:
                known_for_decision = known
                cascade_fallback = True
            winner, scored, gate_reason, bar = decide(
                signature, known_for_decision, STATE.effective_trust_of, weights=weights,
                bar_bias=bar_bias, laya_scores=laya_scores, laya_weight=laya_weight,
            )
            if winner is not None and winner is not pinned:
                held_row = next((item for item in scored if item.profile is pinned), None)
                win_row = next((item for item in scored if item.profile is winner), None)
                if held_row is not None and win_row is not None:
                    turns_held = max(1, int(previous.get("turns", 1) or 1))
                    span = max(1, PIN_GRACE_TURNS * 4)
                    decay = max(0.0, 1.0 - (turns_held - PIN_GRACE_TURNS) / span)
                    bonus = PIN_BONUS_FLOOR + (PIN_BONUS_START - PIN_BONUS_FLOOR) * decay
                    if held_row.utility + bonus >= win_row.utility:
                        # The challenger does not beat the held model by
                        # enough to pay for one cold cache. Keep the pin.
                        winner = pinned
                        reason = "pinned-loyalty"
                    else:
                        reason = "pin-swap"
            if winner is pinned and reason == "fresh":
                reason = "pinned-loyalty"
        elif budget_breached or kind_alarm_active:
            # Force the cheapest model that still clears the capability gate.
            # We keep the same gate/bar so safety is preserved; we just pick
            # by cost instead of utility. This path is shared by the global
            # session budget cap and the per-(model, kind) spend alarm.
            capable = [
                profile for profile in known
                if (not signature.has_image or profile.vision)
                and signature.est_tokens < profile.ctx_window * 0.9
                and profile.cap.get(signature.kind, 0.5) >= bar
                and profile.model not in cascade_excluded
            ]
            if not capable:
                capable = [
                    profile for profile in known
                    if (not signature.has_image or profile.vision)
                    and signature.est_tokens < profile.ctx_window * 0.9
                    and profile.model not in cascade_excluded
                ]
            if not capable:
                capable = [
                    profile for profile in known
                    if (not signature.has_image or profile.vision)
                    and signature.est_tokens < profile.ctx_window * 0.9
                ]
                cascade_fallback = True
            capable.sort(key=_blended_cost)
            winner = capable[0] if capable else known[0]
            reason = "budget-cap" if budget_breached else "kind-alarm"
            for profile in known:
                scored.append(
                    Candidate(profile=profile, capability=profile.cap.get(signature.kind, 0.5),
                              cost_score=0.0, trust=STATE.trust_of(signature.kind, profile.model),
                              headroom=0.0, latency_score=0.0)
                )
        else:
            known_for_decision = [profile for profile in known if profile.model not in cascade_excluded]
            if not known_for_decision:
                known_for_decision = known
                cascade_fallback = True
            winner, scored, gate_reason, bar = decide(
                signature, known_for_decision, STATE.effective_trust_of, weights=weights,
                bar_bias=bar_bias, laya_scores=laya_scores, laya_weight=laya_weight,
            )
            if cascade_excluded and not cascade_fallback:
                reason = "cascade-escalation"

        if winner is None:
            context.signals["policy"] = "no-winner"
            return context

        # The lease is a pure deadline read (review B5): there is no request
        # counter to decrement anywhere. When the deadline passes, the next
        # decision re-runs fresh; sessions pinned during the lease simply
        # re-decide, because stickiness below requires a matching model the
        # gates still accept and a lease that just expired is no longer the
        # session's held model.

        context.candidate_models = [winner.model]
        if session_key:
            turn_cost = _estimate_cost_hkd(signature, winner)
            if reason == "pin-swap" and previous and previous.get("model") != winner.model:
                # A swap resets the turn counter: the new model starts its
                # OWN grace window, so a marginal swap cannot ping-pong on
                # every following turn.
                turns_now = 1
            else:
                turns_now = (previous.get("turns", 0) + 1) if previous else 1
            STATE.sessions[session_key] = {
                "model": winner.model,
                "kind": signature.kind,
                "ts": time.time(),
                "turns": turns_now,
                # RAM-only (C5): the session's newest ask, for the skip-path
                # observer's topical-continuity check and the learner's B2
                # containment verdict on the NEXT turn.
                "ask": signature.text,
                "spend_hkd": session_spend + turn_cost,
            }
            # Update per-(model, kind) spend alarm. Only the previous turn's
            # model-kind pair is charged; the winner of THIS turn is charged
            # on the NEXT turn (when it becomes "previous"). This avoids
            # double-counting and keeps the alarm aligned with continuity.
            if previous and isinstance(previous.get("model"), str) and isinstance(previous.get("kind"), str):
                pair = (previous["model"], previous["kind"])
                session_alarms = STATE.kind_alarms.setdefault(session_key, {})
                alarm = session_alarms.setdefault(pair, {"spend_hkd": 0.0, "ts": time.time(), "epoch": STATE.alarm_epoch})
                alarm["spend_hkd"] += _estimate_cost_hkd(
                    signature,  # use the marginal ask being served now
                    PROFILES[previous["model"]],
                )
                alarm["ts"] = time.time()
                alarm["epoch"] = STATE.alarm_epoch
            _remember_session(session_key, signature.text)

        context.signals["policy"] = {
            "chosen": winner.model,
            "reason": reason,
            "gate": gate_reason,
            "kind": signature.kind,
            "scale": signature.scale,
            "importance": signature.importance,
            "bar": round(bar, 3),
            "est_tokens": signature.est_tokens,
            "ask_tokens": signature.ask_tokens,
            "est_cost_hkd": round(_estimate_cost_hkd(signature, winner), 4),
            "has_image": signature.has_image,
            "stall": signature.stall,
            "directives": sorted(signature.directives),
            "session": session_key,
            "spend_hkd": round(STATE.session_spend(session_key), 4),
            "budget_hkd": budget,
            "budget_alarm": budget_breached,
            "kind_alarm": kind_alarm_active,
            "kind_alarm_model": kind_alarm_model,
            "lease": lease.get("model") if lease is not None else None,
            "weights": {"cost": weights[0], "trust": weights[1],
                        "headroom": weights[2], "latency": weights[3]},
            "cascade": cascade_info if cascade_info.get("excluded") or cascade_info.get("penalized_model") else None,
            "cascade_fallback": cascade_fallback,
            "pin_held": pin_held,
            "exec_demoted": exec_demoted,
            "laya_weight": round(laya_weight, 4),
            "laya_scores": laya_scores if laya_scores else None,
            "scores": [
                {
                    "model": item.profile.model,
                    "cap": round(item.capability, 3),
                    "cost": round(item.cost_score, 3),
                    "trust": round(item.trust, 3),
                    "headroom": round(item.headroom, 3),
                    "utility": round(item.utility, 3),
                }
                for item in scored
            ],
        }
        _log(
            {
                "ts": time.time(),
                "chosen": winner.model,
                "reason": reason,
                "kind": signature.kind,
                "scale": signature.scale,
                "importance": signature.importance,
                "est_tokens": signature.est_tokens,
                "ask_tokens": signature.ask_tokens,
                "est_cost_hkd": round(_estimate_cost_hkd(signature, winner), 4),
                "spend_hkd": round(STATE.session_spend(session_key), 4),
                "budget_hkd": budget,
                "budget_alarm": budget_breached,
                "kind_alarm": kind_alarm_active,
                "kind_alarm_model": kind_alarm_model,
                "has_image": signature.has_image,
                "stall": signature.stall,
                "directives": sorted(signature.directives),
                "candidates": [item.profile.model for item in scored],
                "session": session_key,
                "lease": lease.get("model") if lease is not None else None,
                "cascade": cascade_info if cascade_info.get("excluded") or cascade_info.get("penalized_model") else None,
                "cascade_fallback": cascade_fallback,
                "pin_held": pin_held,
                "exec_demoted": exec_demoted,
                "laya_weight": round(laya_weight, 4),
                "laya_scores": laya_scores if laya_scores else None,
                # Learner telemetry piggybacked on the log (review M6): the
                # control service's /health reads it from the newest record.
                "learner": dict(OBSERVATIONS),
            }
        )
        return context


def _observe_skip_path(context: Any, candidates: list) -> None:
    """Skip-path observer (review B3): a direct escape-hatch group means the
    caller bypassed the policy on purpose. When a recent cursor-auto session
    exists, the switched-to model differs from the model that session was
    served by, and the new ask is topically continuous with that session's
    previous ask, this is negative evidence (-0.20) about the model the
    session used. Same-model picks are router bypasses, not dissatisfaction;
    brand-new chats share no session key and are invisible by construction
    (accepted limitation, review m9).

    Attribution only. The candidate list is NEVER touched - routing behaviour
    on this path is unchanged (plan task 1.4)."""
    if not _learn_enabled():
        return
    try:
        learner = _get_learner()
        if learner is None or not candidates:
            return
        new_model = next((c for c in candidates if isinstance(c, str) and c in PROFILES), None)
        if new_model is None:
            return
        prev_key = _LAST_CURSOR_AUTO_SESSION.get("key")
        if not prev_key:
            return
        # The direct request's own newest human ask, for the continuity check.
        messages = list(getattr(context, "structured_messages", []) or []) or list(
            getattr(context, "raw_messages", []) or []
        )
        ask_pair = _newest_human_ask(messages) if messages else None
        if ask_pair is None:
            return
        session_record = STATE.sessions.get(prev_key) or {}
        prev_record = {
            "ask": str(session_record.get("ask") or ""),
            "kind": str(session_record.get("kind") or ""),
            "model": str(session_record.get("model") or ""),
        }
        learner.observe_switch(prev_record, ask_pair[0], new_model, prev_key)
    except Exception:  # noqa: BLE001 - observation must never break the skip path
        pass


_LAST_CURSOR_AUTO_SESSION: dict[str, str | None] = {"key": None, "ask": ""}


def _remember_session(session_key: str | None, ask_text: str) -> None:
    """Track the most recent cursor-auto session so the skip-path observer
    can correlate a later direct-model switch with it (review B3)."""
    if session_key:
        _LAST_CURSOR_AUTO_SESSION["key"] = session_key
        _LAST_CURSOR_AUTO_SESSION["ask"] = ask_text


# The config must name THIS, not the class. litellm resolves a dotted path to
# whatever attribute it finds, and a class satisfies the RoutingPlugin Protocol
# check (it has a `run` attribute) while failing on every request with
# "run() missing 1 required positional argument: 'context'". Naming an instance
# is also what the `litellm_settings.callbacks` convention does.
router_policy = CursorAutoPolicy()
