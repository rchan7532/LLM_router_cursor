"""
Phase 3 reliability hook (REDUCED SCOPE, review M1/q4): record upstream
429/5xx failures per model, expose them, and change NOTHING about routing.
litellm's own cooldowns (allowed_fails: 2, cooldown_time: 60) already move
traffic off fast-failing deployments; this hook only makes the failures
VISIBLE for /health and /learned.

Registered via `litellm_settings.callbacks` in litellm-config.yaml - a
DIFFERENT plugin mechanism than router_settings.plugins (the routing
plugin is pre-call only and never sees status codes; verified against the
pinned litellm version: CustomLogger.async_log_failure_event fires on
upstream errors, log_success_event on success).

C1 discipline: the hook never raises, and with learning off (POLICY_LEARN=0
or control.json learning_enabled=false) it stays completely inert.

It records only {ts, model, status, kind-of-failure}: no messages, no
payloads, nothing that could carry prompt text to disk (plan C5).
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from typing import Any, Mapping

try:
    from litellm.integrations.custom_logger import CustomLogger
except ImportError:  # pragma: no cover - litellm always present in the proxy
    CustomLogger = object  # type: ignore[assignment, misc]

try:
    import learning as _learning
except ImportError:  # pragma: no cover - split for tests
    _learning = None

RELIABILITY_PATH = os.environ.get(
    "POLICY_RELIABILITY_PATH",
    os.path.join(
        os.environ.get("POLICY_STATE_DIR")
        or ("/app/logs" if os.path.isdir("/app/logs") else tempfile.gettempdir()),
        "reliability.jsonl",
    ),
)
RELIABILITY_MAX_BYTES = int(os.environ.get("POLICY_LOG_MAX_BYTES", "5000000"))

# In-RAM counters so /health can read today's state without re-parsing the
# log; the log itself is the durable record (and what the control service
# tails). Per-model: {model: {"count_429": int, "count_5xx": int,
# "count_error": int, "last_ts": float}}.
RELIABILITY: dict[str, dict[str, Any]] = {}


def _learn_on() -> bool:
    """Same C8 precedence as the policy's learning gate. The hook is part
    of the learning subsystem, so POLICY_LEARN=0 disables it too."""
    if os.environ.get("POLICY_LEARN", "1").lower() in {"0", "false", "no"}:
        return False
    try:
        import control_store as _cs
        store = _cs.default_store()
        return bool(store.learning_enabled())
    except Exception:  # noqa: BLE001 - fail inert, never fail loud
        return False


def record_failure(model: str, status: int | None, now: float | None = None) -> dict[str, Any] | None:
    """Append one failure record and update the RAM counters. Returns the
    record, or None when the hook is inert or the failure carries no model.
    The record is {ts, model, status} and nothing else - no payload fields."""
    now = time.time() if now is None else now
    if not model or not isinstance(model, str):
        return None
    record = {"ts": now, "model": model, "status": int(status) if status else None}
    try:
        directory = os.path.dirname(RELIABILITY_PATH)
        if directory:
            os.makedirs(directory, exist_ok=True)
        if os.path.exists(RELIABILITY_PATH) and os.path.getsize(RELIABILITY_PATH) > RELIABILITY_MAX_BYTES:
            os.replace(RELIABILITY_PATH, RELIABILITY_PATH + ".1")
        with open(RELIABILITY_PATH, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
    except Exception:  # noqa: BLE001 - logging must never affect a request
        pass
    counters = RELIABILITY.setdefault(model, {"count_429": 0, "count_5xx": 0,
                                              "count_error": 0, "last_ts": 0.0})
    if status == 429:
        counters["count_429"] += 1
    elif isinstance(status, int) and 500 <= status < 600:
        counters["count_5xx"] += 1
    else:
        counters["count_error"] += 1
    counters["last_ts"] = now
    return record


def snapshot() -> dict[str, dict[str, Any]]:
    """Per-model failure counters for GET /health. A copy, so callers
    cannot mutate the hook's state."""
    return {model: dict(counters) for model, counters in RELIABILITY.items()}


def last_failure_ts(model: str) -> float | None:
    """The newest recorded failure time for `model`, or None.

    Read by routing_policy's cascade-lite escalation: the policy runs
    pre-call and never sees status codes, so this RAM counter is the only
    bridge between an observed upstream failure and the next decision.
    Returns None for unknown models — a missing entry must be
    indistinguishable from 'no failures', which keeps the fail-open
    contract (a proxy without this hook routes identically)."""
    counters = RELIABILITY.get(model)
    if not counters:
        return None
    ts = counters.get("last_ts")
    if isinstance(ts, (int, float)) and ts > 0:
        return float(ts)
    return None


class ReliabilityHook(CustomLogger if CustomLogger is not object else object):  # type: ignore[misc,valid-type]
    """litellm CustomLogger that records upstream failures. Learning-off it
    does nothing at all (inert), and no method ever raises."""

    # ---- sync callbacks ----

    def log_success_event(self, kwargs, response_obj, start_time, end_time):  # noqa: D102, ANN001
        pass  # successes are not learning signals (429/5xx stay neutral, spec table)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):  # noqa: D102, ANN001
        try:
            if not _learn_on():
                return
            model = self._model_of(kwargs)
            record_failure(model, self._status_of(kwargs, response_obj))
        except Exception:  # noqa: BLE001 - litellm wraps callbacks; assert anyway
            pass

    # ---- async callbacks ----

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):  # noqa: D102, ANN001
        pass

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):  # noqa: D102, ANN001
        try:
            if not _learn_on():
                return
            model = self._model_of(kwargs)
            record_failure(model, self._status_of(kwargs, response_obj))
        except Exception:  # noqa: BLE001
            pass

    async def async_post_call_failure_hook(self, request_data, original_exception,
                                           user_api_key_dict, traceback_str=None):  # noqa: ANN001, D102
        try:
            if _learn_on():
                model = str((request_data or {}).get("model") or "")
                status = getattr(original_exception, "status_code", None)
                record_failure(model, status)
        except Exception:  # noqa: BLE001
            pass
        return None  # never transform the error response

    # ---- helpers ----

    @staticmethod
    def _model_of(kwargs: Any) -> str:
        if not isinstance(kwargs, Mapping):
            return ""
        model = kwargs.get("model")
        if isinstance(model, str) and model:
            return model
        litellm_params = kwargs.get("litellm_params") or {}
        if isinstance(litellm_params, Mapping):
            model = litellm_params.get("model")
            if isinstance(model, str):
                return model
        return ""

    @staticmethod
    def _status_of(kwargs: Any, response_obj: Any) -> int | None:
        """Best-effort status extraction. litellm failure kwargs carry the
        exception info; 429/5xx are the shapes that matter, anything else
        counts as a generic error."""
        if isinstance(kwargs, Mapping):
            exception = kwargs.get("exception") or kwargs.get("original_exception")
            status = getattr(exception, "status_code", None)
            if isinstance(status, int):
                return status
            litellm_params = kwargs.get("litellm_params") or {}
            if isinstance(litellm_params, Mapping):
                status = litellm_params.get("status_code")
                if isinstance(status, int):
                    return status
        status = getattr(response_obj, "status_code", None)
        return status if isinstance(status, int) else None


# The config must name THIS instance (same convention as routing_policy).
reliability_hook = ReliabilityHook()
