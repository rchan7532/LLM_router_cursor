"""
Post-call usage receipt (measurement plumbing).

Registered in litellm-config.yaml under `litellm_settings.callbacks` next to
failure_hook.reliability_hook. Where failure_hook records only failures, this
records the BILLING RECEIPT of every successful cursor-auto call:

  provider-usage (prompt / completion / cached tokens)
    -> usage_state.real_cost_hkd (cached reads priced at the discount rate)
    -> usage_state.record_receipt  (routing_usage.jsonl + usage_state.json)

Why it exists: the decision log's est_cost_hkd assumes the WHOLE thread is
re-sent uncached every turn - a worst-case visibility number. The provider
bills cached prompt reads at ~1/10 the input rate, so on a warm agent loop
the dashboard total (~70 HKD) drifted far from the actual bill (~15 HKD).
Receipt-based accounting closes that gap and lets the router's per-(model,
kind) spend alarm fire on real money, not estimates.

How it knows WHICH decision a receipt belongs to: litellm stashes the
routing plugin's `context.signals` into
`kwargs["litellm_params"]["metadata"]["routing_plugin_signals"]` before the
call, and the same object arrives here on success (verified against the
pinned proxy version with a live call). `signals["policy"]` carries
{session, kind, chosen, client, est_cost_hkd}, which is the join key plus the
estimate to compare the receipt against. The plugin and this hook are
SEPARATE module instances in litellm's loader, so no RAM may be shared
between them - the on-disk aggregate in usage_state is the only bridge.

Discipline (same C1 contract as failure_hook): nothing here may raise into a
request. Every method is wrapped. If the policy signal is absent (direct
escape-hatch call, plugin disabled, foreign client) the receipt is still
recorded with kind=None and client from the request's own headers where
visible - spend accounting must not silently go missing. Streaming responses
accumulate usage in the final chunk; litellm hands the assembled
ModelResponse to log_success_event, so one code path covers both shapes.
"""

from __future__ import annotations

import os
import time
from typing import Any, Mapping

try:
    from litellm.integrations.custom_logger import CustomLogger
except ImportError:  # pragma: no cover - litellm always present in the proxy
    CustomLogger = object  # type: ignore[assignment, misc]

try:
    import usage_state as _usage
except ImportError:  # pragma: no cover - deployed together, split for tests
    _usage = None  # type: ignore[assignment]

# Kill switch: POLICY_USAGE_RECORD=0 turns receipts off entirely (the hook
# stays loaded but inert). Separate from POLICY_LEARN: billing visibility is
# an operator-facing feature, not a learning subsystem.
RECORDING_ON = os.environ.get("POLICY_USAGE_RECORD", "1").lower() not in {
    "0", "false", "no",
}


def _get(mapping: Any, *keys: str) -> Any:
    """Dotted-safe dict walk: _get(d, "a", "b") == d["a"]["b"] if every hop
    is a Mapping, else None."""
    current = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _num(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _duration_seconds(start_time: Any, end_time: Any) -> float | None:
    """Wall-clock seconds between litellm's callback timestamps. Accepts
    floats (unix seconds) or datetimes; returns None when not derivable or
    nonsensical (never invents a number)."""
    try:
        if isinstance(start_time, bool) or isinstance(end_time, bool):
            return None
        if isinstance(start_time, (int, float)) and isinstance(end_time, (int, float)):
            delta = float(end_time) - float(start_time)
        elif hasattr(start_time, "timestamp") and hasattr(end_time, "timestamp"):
            delta = float(end_time.timestamp()) - float(start_time.timestamp())
        else:
            return None
        return delta if 0.0 <= delta < 3600.0 else None
    except Exception:  # noqa: BLE001
        return None


def extract_receipt_parts(kwargs: Any, response_obj: Any) -> dict[str, Any] | None:
    """Pull everything needed for one receipt out of litellm's callback
    payload. Returns None when the response carries no usage at all (nothing
    to bill). Pure function - unit-testable without litellm running."""
    if not isinstance(kwargs, Mapping):
        return None

    signals = _get(kwargs, "litellm_params", "metadata", "routing_plugin_signals")
    policy = signals.get("policy") if isinstance(signals, Mapping) else None
    if not isinstance(policy, Mapping):
        policy = {}
    # Skip-path annotations (escape-hatch groups, e.g. mimo-v2.6-pro) arrive
    # as TOP-LEVEL signal keys, not under "policy": the policy dict is never
    # built there. Prefer the dict, fall back to the annotation so a direct
    # call still gets session/task/client on its receipt.
    def _signal(name: str) -> Any:
        value = policy.get(name)
        if isinstance(value, str) and value:
            return value
        fallback = signals.get("annotator_" + name) if isinstance(signals, Mapping) else None
        return fallback if isinstance(fallback, str) and fallback else None

    usage = getattr(response_obj, "usage", None)
    if usage is None and isinstance(response_obj, Mapping):
        usage = response_obj.get("usage")
    if usage is None:
        return None
    prompt_tokens = _num(getattr(usage, "prompt_tokens", None)
                         if not isinstance(usage, Mapping) else usage.get("prompt_tokens"))
    completion_tokens = _num(getattr(usage, "completion_tokens", None)
                             if not isinstance(usage, Mapping) else usage.get("completion_tokens"))
    details = getattr(usage, "prompt_tokens_details", None)
    if isinstance(usage, Mapping):
        details = usage.get("prompt_tokens_details")
    cached_tokens = _num(getattr(details, "cached_tokens", None)
                         if not isinstance(details, Mapping)
                         else details.get("cached_tokens"))
    if prompt_tokens <= 0 and completion_tokens <= 0:
        return None

    # The model that SERVED (and was billed), not just the one chosen: on a
    # retry/failover these can differ. kwargs["model"] is the deployment
    # model (provider-prefixed or not, depending on litellm version); fall
    # back to the policy's pick.
    model = kwargs.get("model") or kwargs.get("litellm_model") or ""
    if not isinstance(model, str) or not model:
        model = str(policy.get("chosen") or "")
    model = _normalise_model(model)

    return {
        "model": model,
        "session": _signal("session"),
        "kind": policy.get("kind") if isinstance(policy.get("kind"), str) else None,
        "client": _signal("client") or _client_from_headers(kwargs),
        "task": _signal("task"),
        "switch_back_from": (policy.get("switch_back_from")
                             if isinstance(policy.get("switch_back_from"), str) else None),
        "est_cost_hkd": policy.get("est_cost_hkd"),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
    }


_MODEL_PREFIXES = ("openai/", "azure/", "anthropic/", "google/")


def _normalise_model(model: str) -> str:
    """Accept either the profile key ("openai/glm-5.3") or the bare name
    ("glm-5.3") and return whichever usage_state.PRICING knows about."""
    if _usage is None:
        return model
    if model in _usage.PRICING:
        return model
    bare = model
    for prefix in _MODEL_PREFIXES:
        if bare.startswith(prefix):
            bare = bare[len(prefix):]
            break
    for key in _usage.PRICING:
        if key.endswith("/" + bare):
            return key
    return model


def _client_from_headers(kwargs: Any) -> str:
    """Fallback client tag when no policy signal exists (direct escape-hatch
    group, plugin disabled): read the caller's identity off the request
    metadata the proxy injects. X-Client-Type wins (nginx stamps it per
    entry path, so it reflects the route the caller used, not whatever the
    client library puts in User-Agent - aider's UA string does not contain
    "aider"). User-Agent is the secondary signal. Same vocabulary as
    routing_policy._detect_client but deliberately not shared code (separate
    module instances, and the plugin must keep working without this file and
    vice versa)."""
    headers = _get(kwargs, "metadata", "headers") or _get(kwargs, "litellm_params", "metadata", "headers")
    if not isinstance(headers, Mapping):
        return "unknown"
    explicit = headers.get("x-client-type") or headers.get("X-Client-Type") or ""
    text = str(explicit).strip().lower()
    if text in {"aider", "cursor", "other"}:
        return text
    agent = headers.get("user-agent") or headers.get("User-Agent") or ""
    lowered = str(agent).lower()
    if "aider" in lowered:
        return "aider"
    if "cursor" in lowered:
        return "cursor"
    return "other" if agent else "unknown"


class UsageRecorder(CustomLogger if CustomLogger is not object else object):  # type: ignore[misc,valid-type]
    """Records real-usage receipts for successful calls. Never raises, never
    changes routing, inert when POLICY_USAGE_RECORD=0 or usage_state is
    missing."""

    def _record(self, kwargs: Any, response_obj: Any,
                start_time: Any = None, end_time: Any = None) -> None:
        if not RECORDING_ON or _usage is None:
            return
        parts = extract_receipt_parts(kwargs, response_obj)
        if parts is None:
            return
        _usage.record_receipt(
            session=parts["session"],
            model=parts["model"],
            kind=parts["kind"],
            client=parts["client"],
            task=parts["task"],
            switch_back_from=parts.get("switch_back_from"),
            prompt_tokens=parts["prompt_tokens"],
            completion_tokens=parts["completion_tokens"],
            cached_tokens=parts["cached_tokens"],
            est_cost_hkd=parts["est_cost_hkd"],
            duration_s=_duration_seconds(start_time, end_time),
            ts=time.time(),
        )

    # ---- sync + async success paths ----

    def log_success_event(self, kwargs, response_obj, start_time, end_time):  # noqa: D102, ANN001
        try:
            self._record(kwargs, response_obj, start_time, end_time)
        except Exception:  # noqa: BLE001 - a broken receipt must not eat a response
            pass

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):  # noqa: D102, ANN001
        try:
            self._record(kwargs, response_obj, start_time, end_time)
        except Exception:  # noqa: BLE001
            pass

    # Failed calls consume tokens on some providers but the usage object is
    # usually absent there; failure_hook already owns the failure record.
    # Nothing to bill without a receipt, so failures stay silent here.


# The config must name THIS instance (same convention as routing_policy).
usage_recorder = UsageRecorder()
