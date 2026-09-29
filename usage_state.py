"""
Usage accounting (measurement plumbing).

Two jobs, both deliberately dependency-free (no import of routing_policy, so
this module can be loaded by BOTH the routing plugin and the post-call
callback as separate litellm file-path module instances without a circular
import):

  1. real cost math. `real_cost_hkd(model, prompt_tokens, completion_tokens,
     cached_tokens)` prices a completed call from the provider's reported
     usage receipt. Input is split into cached-read (heavily discounted) and
     fresh (full rate) tokens; output is the full rate. This is the number
     the operator's bill tracks, as opposed to routing_policy's pre-call
     `est_cost_hkd`, which assumes the WHOLE thread is re-sent uncached every
     turn. On a warm agent loop the two diverge by ~5x (the 70-vs-15 HKD
     complaint): the estimate is a worst-case visibility signal, the receipt
     is what you actually pay.

  2. a cross-module spend bridge. The plugin that decides routing and the
     callback that sees the usage are different module instances in the same
     process (litellm loads each via importlib file-path, never sys.modules).
     They cannot share RAM, so this module keeps a small JSON aggregate on
     disk (single writer: the callback; reader: the plugin, mtime-cached)
     recording real spend per session and per (model, kind). The plugin's
     per-(model, kind) spend ALARM can then fire on REAL spend, not only on
     the inflated pre-call estimate.

PRICING mirrors routing_policy.PROFILES (cost_in/cost_out) and adds the
cache-read rate that only lives in litellm-config.yaml's model_info today. A
test in tests/test_usage_state.py asserts the in/out halves match PROFILES so
the two tables can never silently drift. cache_in is 1/10 of cost_in for every
entry, matching the config. HKD/USD matches routing_policy.HKD_PER_USD.

Nothing here raises into a request: `record_receipt` is wrapped by the caller
(usage_hook) and `load_totals` returns an empty view on any read error.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from typing import Any

HKD_PER_USD = 7.8

# model (== routing_policy.PROFILES key == litellm deployment
# `litellm_params.model`) -> USD per 1M tokens (input, output, cache-read).
# in/out copied from PROFILES; cache_in = in/10 (matches litellm-config.yaml
# cache_read_input_token_cost for every deployment).
PRICING: dict[str, tuple[float, float, float]] = {
    "openai/qwen3-vl-flash": (0.016, 0.18, 0.0016),
    "openai/glm-5.3-flash": (0.10, 0.34, 0.01),
    "openai/qwen3.8-flash": (0.12, 0.38, 0.012),
    "openai/deepseek-v4.1-flash": (0.30, 1.21, 0.03),
    "openai/kimi-k2.7-code": (0.77, 3.22, 0.077),
    "openai/glm-5.3": (1.13, 3.54, 0.113),
    "openai/claude-haiku-4-5": (1.01, 5.03, 0.101),
    # Escape hatches (not in PROFILES, never auto-picked). Prices from the
    # operators' published lists, 2026-09-29:
    #   glm-5.3-flashx: HKD list (2.355 in / 8.2425 out per 1M) at 7.8 HKD/USD.
    #   mimo-v2.6-pro: USD list; cache READ is 0.0018 (NOT in/10 - the
    #     provider prices cache writes separately at the full input rate).
    "openai/glm-5.3-flashx": (0.302, 1.057, 0.0302),
    "openai/mimo-v2.6-pro": (0.2175, 0.435, 0.0018),
}


def _default_state_dir() -> str:
    if os.name != "nt" and os.path.isdir("/app/logs"):
        return "/app/logs"
    return tempfile.gettempdir()


STATE_DIR = os.environ.get("POLICY_STATE_DIR") or _default_state_dir()
# Per-call receipts (durable record) and the tiny live aggregate (RAM-backed
# bridge the plugin reads). Both on the proxy's own volume; the control
# service mounts the same volume read-only, so it can serve /usage too.
USAGE_LOG_PATH = os.environ.get(
    "POLICY_USAGE_LOG", os.path.join(STATE_DIR, "routing_usage.jsonl")
)
USAGE_STATE_PATH = os.environ.get(
    "POLICY_USAGE_STATE", os.path.join(STATE_DIR, "usage_state.json")
)
USAGE_LOG_MAX_BYTES = int(os.environ.get("POLICY_LOG_MAX_BYTES", "5000000"))
# How many sessions the on-disk aggregate keeps before dropping the oldest.
# A session is a decision-loop, so a few hundred is plenty and bounds the
# file at well under a megabyte.
USAGE_STATE_MAX_SESSIONS = int(os.environ.get("POLICY_USAGE_STATE_CAP", "512"))
USAGE_STATE_TTL_S = float(os.environ.get("POLICY_USAGE_STATE_TTL", "3600"))


def _int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


def real_cost_hkd(model: str, prompt_tokens: int, completion_tokens: int,
                  cached_tokens: int = 0) -> float:
    """Billed HKD for one completed call, from the usage receipt.

    cached_tokens (a subset of prompt_tokens that the provider served from
    its prompt cache at the discount rate) is priced at cache_in; the rest of
    the prompt at the full input rate. Output at the output rate. A model with
    no PRICING entry (e.g. a direct escape-hatch name) returns 0.0 so an
    unknown never inflates an alarm - the decision log still records est_cost.
    """
    pricing = PRICING.get(model)
    if pricing is None:
        return 0.0
    cost_in, cost_out, cache_in = pricing
    cached = max(0, min(_int(cached_tokens), _int(prompt_tokens)))
    fresh = max(0, _int(prompt_tokens) - cached)
    usd = (
        fresh / 1_000_000 * cost_in
        + cached / 1_000_000 * cache_in
        + _int(completion_tokens) / 1_000_000 * cost_out
    )
    return usd * HKD_PER_USD


# --------------------------------------------------------------------------
# Writer (the post-call callback owns this; single writer per process)
# --------------------------------------------------------------------------

_WRITE_LOCK = threading.Lock()
# In-process mirror of the on-disk aggregate, so record_receipt does not have
# to read the file back on every call. Keyed by session; each entry is
# {"ts": float, "real_hkd": float, "kinds": {"model\\u0000kind": float},
#  "clients": {"aider|cursor|other": float}} (kinds/clients values are HKD).
_AGG: dict[str, dict[str, Any]] = {}


def _pair_key(model: str, kind: str) -> str:
    return f"{model}\u0000{kind}"


def _prune_locked(now: float) -> None:
    stale = [s for s, e in _AGG.items() if now - float(e.get("ts", 0)) > USAGE_STATE_TTL_S]
    for s in stale:
        _AGG.pop(s, None)
    if len(_AGG) <= USAGE_STATE_MAX_SESSIONS:
        return
    # Drop oldest-by-ts down to the cap.
    ordered = sorted(_AGG.items(), key=lambda kv: float(kv[1].get("ts", 0)))
    for s, _ in ordered[: len(_AGG) - USAGE_STATE_MAX_SESSIONS]:
        _AGG.pop(s, None)


def _flush_locked() -> None:
    """Atomically persist the aggregate. Torn-write safe: temp + os.replace."""
    payload = {"ts": time.time(), "sessions": _AGG}
    try:
        directory = os.path.dirname(USAGE_STATE_PATH)
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp = USAGE_STATE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, default=str)
        os.replace(tmp, USAGE_STATE_PATH)
    except Exception:  # noqa: BLE001 - accounting must never break a response
        pass


def _append_receipt_locked(record: dict[str, Any]) -> None:
    try:
        directory = os.path.dirname(USAGE_LOG_PATH)
        if directory:
            os.makedirs(directory, exist_ok=True)
        if os.path.exists(USAGE_LOG_PATH) and os.path.getsize(USAGE_LOG_PATH) > USAGE_LOG_MAX_BYTES:
            os.replace(USAGE_LOG_PATH, USAGE_LOG_PATH + ".1")
        with open(USAGE_LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
    except Exception:  # noqa: BLE001
        pass


def record_receipt(*, session: str | None, model: str, kind: str | None,
                   client: str | None, task: str | None = None,
                   prompt_tokens: int, completion_tokens: int,
                   cached_tokens: int = 0, est_cost_hkd: float | None = None,
                   duration_s: float | None = None,
                   ts: float | None = None) -> float:
    """Record one usage receipt. Returns the real HKD for this call (0.0 for
    an unpriced model). Updates the durable log, the RAM aggregate, and the
    on-disk aggregate the plugin reads. Never raises to the caller's caller
    beyond what json/os can throw, which the hook wraps anyway.

    task is a measurement-only label from routing_policy's [[task:<slug>]]
    directive. It never affects routing and defaults to None for all existing
    callers (fail-open).

    duration_s is the upstream wall-clock latency in seconds (litellm passes
    start/end times to the success callback). It is the latency-vs-cost
    comparison signal: e.g. "glm-5.3-flash is cheaper per task but slower".
    None when the timing is not derivable; never inferred.
    """
    now = time.time() if ts is None else ts
    real = real_cost_hkd(model, prompt_tokens, completion_tokens, cached_tokens)
    task_value = task if isinstance(task, str) and task else None
    duration_value: float | None = None
    if isinstance(duration_s, (int, float)) and not isinstance(duration_s, bool):
        duration_value = round(float(duration_s), 3)
        if duration_value < 0:
            duration_value = None
    record = {
        "ts": now,
        "session": session,
        "model": model,
        "kind": kind,
        "client": client,
        "task": task_value,
        "prompt_tokens": _int(prompt_tokens),
        "completion_tokens": _int(completion_tokens),
        "cached_tokens": _int(cached_tokens),
        "real_cost_hkd": round(real, 6),
        "est_cost_hkd": round(float(est_cost_hkd), 6) if est_cost_hkd is not None else None,
        "duration_s": duration_value,
    }
    with _WRITE_LOCK:
        _append_receipt_locked(record)
        if session:
            entry = _AGG.setdefault(
                session, {"ts": now, "real_hkd": 0.0, "kinds": {}, "clients": {}, "tasks": {}}
            )
            entry["ts"] = now
            entry["real_hkd"] = round(float(entry.get("real_hkd", 0.0)) + real, 6)
            kinds = entry.setdefault("kinds", {})
            if kind:
                pk = _pair_key(model, kind)
                kinds[pk] = round(float(kinds.get(pk, 0.0)) + real, 6)
            if client:
                clients = entry.setdefault("clients", {})
                clients[client] = round(float(clients.get(client, 0.0)) + real, 6)
            if task_value and client:
                tasks = entry.setdefault("tasks", {})
                tk = f"{task_value}\x00{client}"
                tasks[tk] = round(float(tasks.get(tk, 0.0)) + real, 6)
            _prune_locked(now)
            _flush_locked()
    return real


# --------------------------------------------------------------------------
# Reader (the routing plugin owns this; mtime-cached)
# --------------------------------------------------------------------------

class UsageView:
    """A point-in-time snapshot of the on-disk aggregate. Plain class (no
    dataclass) to match the module-loading discipline used across this repo."""

    __slots__ = ("real_hkd", "kind_hkd", "client_hkd", "task_hkd")

    def __init__(self) -> None:
        self.real_hkd = 0.0
        self.kind_hkd: dict[tuple[str, str], float] = {}
        self.client_hkd: dict[str, float] = {}
        self.task_hkd: dict[tuple[str, str], float] = {}


_EMPTY = UsageView()
_READ_LOCK = threading.Lock()
_READ_STAMP: tuple[int, int] | None = None
_READ_CACHE: dict[str, UsageView] = {}


def _parse_entry(entry: Any) -> UsageView:
    view = UsageView()
    if not isinstance(entry, dict):
        return view
    view.real_hkd = float(entry.get("real_hkd", 0.0) or 0.0)
    for pk, val in (entry.get("kinds") or {}).items():
        if isinstance(pk, str) and "\u0000" in pk:
            model, kind = pk.split("\u0000", 1)
            view.kind_hkd[(model, kind)] = float(val or 0.0)
    for cl, val in (entry.get("clients") or {}).items():
        view.client_hkd[str(cl)] = float(val or 0.0)
    for tk, val in (entry.get("tasks") or {}).items():
        if isinstance(tk, str) and "\u0000" in tk:
            slug, client = tk.split("\u0000", 1)
            view.task_hkd[(slug, client)] = float(val or 0.0)
    return view


def load_view(session: str | None) -> UsageView:
    """Real-spend view for one session from usage_state.json. Returns an
    empty view if the file is missing/corrupt or session is None. mtime-cached
    so a steady stream of requests only pays the read when the callback
    actually wrote (same discipline as ControlStore reading control.json)."""
    global _READ_STAMP
    if not session:
        return _EMPTY
    try:
        stat = os.stat(USAGE_STATE_PATH)
    except OSError:
        return _EMPTY
    stamp = (stat.st_mtime_ns, stat.st_size)
    with _READ_LOCK:
        if stamp != _READ_STAMP:
            try:
                with open(USAGE_STATE_PATH, encoding="utf-8") as handle:
                    payload = json.load(handle)
                sessions = payload.get("sessions") if isinstance(payload, dict) else None
                _READ_CACHE.clear()
                if isinstance(sessions, dict):
                    for key, entry in sessions.items():
                        _READ_CACHE[str(key)] = _parse_entry(entry)
                _READ_STAMP = stamp
            except Exception:  # noqa: BLE001 - corrupt/partial read keeps last-good
                return _READ_CACHE.get(session, _EMPTY)
        return _READ_CACHE.get(session, _EMPTY)
