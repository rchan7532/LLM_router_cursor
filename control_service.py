"""
Control service for the llm-router.

A tiny HTTP service that owns writes to control.json (and, in phase 2, drains
the observation queue into learned.json). It runs in the same compose as the
litellm proxy: control.json lives on a volume the policy mounts read-only,
so this service is the single writer by construction (review B4).

Why a separate service: the policy plugin must gain no HTTP client and no
write path - it stays a pure file reader with mtime caching, so a control-plane
outage or bug cannot touch the request path.

Every request must carry `Authorization: Bearer $CONTROL_TOKEN`. The nginx
path token is not authentication - it appears in nginx's access log - so the
service authenticates callers itself (review m3; DEPLOY.md Step 8b). When
CONTROL_TOKEN is unset the service refuses every write and serves reads, so
a misconfigured deploy fails closed on the write path instead of standing up
an unauthenticated control plane.

Run standalone (dev):  python control_service.py            (port 4010)
Not designed to be exposed directly; put it behind nginx like the proxy.

Endpoints (all JSON):
  GET  /health          proxy-independent status + active lease + revision
  GET  /decisions?n=50  tail of routing.jsonl (parsed, newest last)
  GET  /learned         parsed learned.json (or {"present": false})
  GET  /usage?n=2000    receipt-based spend rollup (real vs estimate) + sessions
  GET  /receipts?n=50   tail of routing_usage.jsonl (newest last)
  POST /lease           {model, seconds} -> deadline lease (review B5)
  POST /lease/clear     drop any active lease
  POST /weights         {cost|trust|headroom|latency: 0..1}   (partial ok)
  POST /learning        {enabled: bool} or {reset: "phrases"|"trust"|"bars"|"all"}
  POST /reset/learned   physical delete of learned.json (both files recreated)

Lease semantics (review B5): seconds-primary and stateless. The stored lease
is a single `expires_ts` deadline in epoch seconds, evaluated by the policy
on read without any in-proxy state, so restarts cannot re-arm or lose it.
A request-count lease is converted to that deadline at issue time
(~LEASE_SECONDS_PER_REQUEST, clamped to the max window); the small overshoot
versus an exact count is documented and accepted. One active lease at a
time: a new POST replaces the old. Bounds are clamped here, loudly.
"""

from __future__ import annotations

import hmac
import json
import os
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping
from urllib.parse import parse_qs, urlparse

STATE_DIR = os.environ.get("POLICY_STATE_DIR") or (
    "/app/logs" if os.path.isdir("/app/logs") else tempfile.gettempdir()
)
# Paths are env-overridable so compose can point control.json at its own
# volume and learned.json/routing.jsonl at the read-only log mount (B4).
CONTROL_PATH = os.environ.get("CONTROL_PATH", os.path.join(STATE_DIR, "control.json"))
LEARNED_PATH = os.environ.get("LEARNED_PATH", os.path.join(STATE_DIR, "learned.json"))
DECISIONS_PATH = os.environ.get("POLICY_LOG", os.path.join(STATE_DIR, "routing.jsonl"))
# Phase 3: the failure hook's log, on the same read-only mount (review q4).
RELIABILITY_PATH = os.environ.get("POLICY_RELIABILITY_PATH", os.path.join(STATE_DIR, "reliability.jsonl"))
# Measurement plumbing: the usage_hook's receipts log and the tiny live
# aggregate written alongside it (usage_state.py). Read-only here, like the
# decision log; both live on the proxy-owned router-logs volume.
USAGE_LOG_PATH = os.environ.get("POLICY_USAGE_LOG", os.path.join(STATE_DIR, "routing_usage.jsonl"))
USAGE_STATE_PATH = os.environ.get("POLICY_USAGE_STATE", os.path.join(STATE_DIR, "usage_state.json"))
PORT = int(os.environ.get("CONTROL_PORT", "4010"))
# Bearer token the service itself requires. Distinct from the nginx path
# token and from LITELLM_MASTER_KEY: a leaked MCP config must not be able to
# replay against /v1 for completions (review m3).
CONTROL_TOKEN = os.environ.get("CONTROL_TOKEN", "")

# Valid weight keys mirror routing_policy.W_* plus the laya shadow weight;
# a value outside [0,1] is a 400, not a silent clamp.
WEIGHT_KEYS = ("cost", "trust", "headroom", "latency", "laya")
BUDGET_MIN = 1.0
BUDGET_MAX = 10000.0
# Lease bounds (review B5). A lease with no deadline would be a persistent
# pin, which is exactly the session-identity race this design refuses.
LEASE_MAX_SECONDS = 900.0
LEASE_MAX_REQUESTS = 200
LEASE_SECONDS_PER_REQUEST = 30.0  # request-count -> deadline conversion

_WRITE_LOCK = threading.Lock()


def _load_control() -> dict[str, Any]:
    try:
        with open(CONTROL_PATH, encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return data
    except (OSError, ValueError):
        pass
    return {"lease": None, "weights": None, "learning_enabled": True, "revision": 0}


def _atomic_write_json(path: str, payload: Mapping[str, Any]) -> None:
    """tmp+fsync+rename so a reader never sees a half-written file and a
    crash cannot leave the path itself torn (review M5 / plan C11)."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _save_control(control: dict[str, Any]) -> dict[str, Any]:
    control["revision"] = int(control.get("revision", 0)) + 1
    _atomic_write_json(CONTROL_PATH, control)
    return control


def set_lease(model: str, requests: int | None, seconds: float | None, set_by: str) -> dict[str, Any]:
    """Store one lease as a stateless deadline (review B5).

    Exactly one bound is accepted. A seconds bound is stored as-is (clamped
    to LEASE_MAX_SECONDS). A requests bound is converted to a deadline at
    issue time: `now + requests * LEASE_SECONDS_PER_REQUEST`, clamped to the
    same max. The policy evaluates the deadline on every read with no
    in-proxy counter, so a restart can neither re-arm nor lose the lease;
    the conversion makes the count approximate, which is the documented
    overshoot the seconds bound exists to backstop.
    """
    if requests is None and seconds is None:
        raise ValueError("lease needs exactly one bound: requests or seconds")
    if requests is not None and seconds is not None:
        raise ValueError("lease accepts only one bound: requests or seconds, not both")
    now = time.time()
    if requests is not None:
        if not isinstance(requests, int) or requests < 1:
            raise ValueError("requests must be a positive integer")
        # Clamp at the service (review B5): an oversized count lands on the
        # max window rather than being rejected, so a call meant as "about
        # 500 requests" still produces a bounded lease.
        expires = now + min(requests * LEASE_SECONDS_PER_REQUEST, LEASE_MAX_SECONDS)
    else:
        if not isinstance(seconds, (int, float)) or seconds <= 0:
            raise ValueError("seconds must be a positive number")
        expires = now + min(float(seconds), LEASE_MAX_SECONDS)
    lease: dict[str, Any] = {
        "model": model,
        "expires_ts": expires,
        "set_by": set_by,
        "set_ts": now,
    }
    if requests is not None:
        lease["requested_requests"] = int(requests)  # audit only; never evaluated
    with _WRITE_LOCK:
        control = _load_control()  # one active lease: a new POST replaces the old
        control["lease"] = lease
        return _save_control(control)


def clear_lease() -> dict[str, Any]:
    with _WRITE_LOCK:
        control = _load_control()
        control["lease"] = None
        return _save_control(control)


def set_weights(overrides: Mapping[str, Any]) -> dict[str, Any]:
    clean: dict[str, float] = {}
    for key, value in overrides.items():
        if key not in WEIGHT_KEYS:
            raise ValueError(f"unknown weight {key!r}; valid: {', '.join(WEIGHT_KEYS)}")
        if not isinstance(value, (int, float)) or not (0.0 <= float(value) <= 1.0):
            raise ValueError(f"weight {key} must be a number in [0,1]")
        clean[key] = round(float(value), 4)
    if not clean:
        raise ValueError("no valid weights supplied")
    with _WRITE_LOCK:
        control = _load_control()
        merged = dict(control.get("weights") or {})
        merged.update(clean)
        control["weights"] = merged
        return _save_control(control)


def set_learning(enabled: bool) -> dict[str, Any]:
    with _WRITE_LOCK:
        control = _load_control()
        control["learning_enabled"] = bool(enabled)
        return _save_control(control)


def set_budget(hkd: float) -> dict[str, Any]:
    if not isinstance(hkd, (int, float)) or not (BUDGET_MIN <= float(hkd) <= BUDGET_MAX):
        raise ValueError(f"budget must be a number in [{BUDGET_MIN}, {BUDGET_MAX}]")
    with _WRITE_LOCK:
        control = _load_control()
        control["budget_hkd"] = round(float(hkd), 4)
        return _save_control(control)


def clear_budget() -> dict[str, Any]:
    with _WRITE_LOCK:
        control = _load_control()
        control["budget_hkd"] = None
        return _save_control(control)


def bump_alarm_epoch() -> dict[str, Any]:
    """Increment the global alarm epoch. Every proxy drops stale per-(model,
    kind) alarms on its next control.json read. This is the control-plane
    clear for the kind-alarm fail-safe."""
    with _WRITE_LOCK:
        control = _load_control()
        control["alarm_epoch"] = int(control.get("alarm_epoch", 0)) + 1
        return _save_control(control)


def _load_learned() -> dict[str, Any]:
    try:
        with open(LEARNED_PATH, encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return data
    except (OSError, ValueError):
        pass
    return {}


def reset_learned(scope: str) -> dict[str, Any]:
    """Reset one learned table. 'all' deletes the file so the learner starts
    from a pristine slate. Never raises on a missing table."""
    if scope not in {"phrases", "trust", "bars", "all"}:
        raise ValueError("reset scope must be one of phrases, trust, bars, all")
    with _WRITE_LOCK:
        if scope == "all":
            try:
                os.remove(LEARNED_PATH)
            except OSError:
                pass
            return {"reset": scope, "present": False}
        learned = _load_learned()
        if scope == "phrases":
            learned.pop("phrase_kind", None)
            learned.pop("key_points", None)
        elif scope == "trust":
            learned.pop("model_trust", None)
        elif scope == "bars":
            learned.pop("kind_bar_bias", None)
        _atomic_write_json(LEARNED_PATH, learned)
        return {"reset": scope, "learned": learned}


def tail_decisions(limit: int) -> list[dict[str, Any]]:
    """Newest `limit` decisions from routing.jsonl.

    Robust to torn writes: reads the whole tail and parses line-by-line,
    skipping anything that is not valid JSON (a half-written record, a merged
    fragment) instead of silently dropping a good record glued to a bad one,
    which readlines() would do when the last line lacks a newline.
    """
    try:
        with open(DECISIONS_PATH, encoding="utf-8") as handle:
            content = handle.read()
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for line in content.splitlines()[-limit * 2:]:  # window: skip torn prefix cheaply
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            out.append(record)
    return out[-limit:]


def health() -> dict[str, Any]:
    control = _load_control()
    lease = control.get("lease")
    active = None
    if isinstance(lease, Mapping):
        expires = lease.get("expires_ts")
        if isinstance(expires, (int, float)) and time.time() < float(expires):
            active = dict(lease)
    return {
        "status": "ok",
        "ts": time.time(),
        "learning_enabled": bool(control.get("learning_enabled", True)),
        "lease": active,
        "weights": control.get("weights"),
        "revision": int(control.get("revision", 0)),
        "budget_hkd": control.get("budget_hkd"),
        "alarm_epoch": int(control.get("alarm_epoch", 0)),
        "laya_weight": (control.get("weights") or {}).get("laya"),
        "decisions_present": os.path.exists(DECISIONS_PATH),
        "learned_present": os.path.exists(LEARNED_PATH),
        # Learner telemetry, piggybacked on the decision log (review M6: no
        # second writable file; the proxy stamps queue depth / errors /
        # extraction count into every decision record).
        "learner": _learner_stats(),
        # Phase 3 reliability (review q4: log-and-expose only). Per-model
        # 429/5xx counters from reliability.jsonl; litellm's cooldowns do
        # the routing-side work, this only makes failures visible.
        "reliability": _reliability_stats(),
    }


def _learner_stats() -> dict[str, Any]:
    """Queue depth / error count from the newest decision-log record."""
    records = tail_decisions(1)
    if not records:
        return {"queue_depth": None, "errors": None, "extractions": None}
    last = records[-1]
    learner = last.get("learner")
    if not isinstance(learner, Mapping):
        return {"queue_depth": None, "errors": None, "extractions": None}
    return {
        "queue_depth": learner.get("queue_depth"),
        "errors": learner.get("errors"),
        "extractions": learner.get("extractions"),
    }


def _reliability_stats() -> dict[str, dict[str, int]]:
    """Per-model failure counts from reliability.jsonl (last ROTATE_WINDOW
    records are enough for a per-user router)."""
    try:
        with open(RELIABILITY_PATH, encoding="utf-8") as handle:
            lines = handle.readlines()[-2000:]
    except OSError:
        return {}
    stats: dict[str, dict[str, int]] = {}
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        model = record.get("model")
        if not isinstance(model, str) or not model:
            continue
        counters = stats.setdefault(model, {"count_429": 0, "count_5xx": 0,
                                            "count_error": 0})
        status = record.get("status")
        if status == 429:
            counters["count_429"] += 1
        elif isinstance(status, int) and 500 <= status < 600:
            counters["count_5xx"] += 1
        else:
            counters["count_error"] += 1
    return stats


def tail_usage(limit: int) -> list[dict[str, Any]]:
    """Newest `limit` usage receipts from routing_usage.jsonl.

    Same torn-write robustness as tail_decisions: parse line-by-line, skip
    anything that is not a valid dict, so a half-written final line cannot
    hide the good records before it."""
    try:
        with open(USAGE_LOG_PATH, encoding="utf-8") as handle:
            content = handle.read()
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for line in content.splitlines()[-limit * 2:]:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            out.append(record)
    return out[-limit:]


def _load_usage_state() -> dict[str, Any]:
    try:
        with open(USAGE_STATE_PATH, encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def usage_summary(limit: int = 2000) -> dict[str, Any]:
    """Receipt-based spend rollup for the dashboard.

    Returns the real billed HKD vs the pre-call estimate, broken down per
    model, per client (aider/cursor/...), per task kind, and per task slug
    (with a redo flag when the same slug appears under multiple clients),
    over the newest `limit` receipts. Each by_model row also carries prompt/
    completion/cached token totals and summed upstream latency (duration_s
    over duration_calls), so unit economics and latency are readable without
    raw-log math. `by_task_model` is the sibling of `by_task` keyed by model
    instead of client: the model-replacement comparison must not merge two
    models that ran the same task under the same client. This is what closes
    the dashboard-vs-bill gap: the
    estimate column sums est_cost_hkd (worst-case, uncached); the actual
    column sums real_cost_hkd (cached reads at the discount rate). If the
    receipts log is absent the totals read as zero and `present` is false,
    so the dashboard shows the estimate it always showed rather than a
    misleading empty bill.
    """
    records = tail_usage(limit)
    totals = {
        "present": bool(records),
        "count": len(records),
        "real_hkd": 0.0,
        "est_hkd": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "by_model": {},
        "by_client": {},
        "by_kind": {},
        "by_task": {},
        "by_task_model": {},
        "task_redo": [],
        # One-sided trial signal: {model: count} of fleet receipts flagged as
        # returning from that direct model mid-task (see _switch_back_from).
        # Frequent switches = the trial model is being abandoned = quality or
        # latency complaint. Absent key reads as zero.
        "switch_backs": {},
    }
    by_task_flat: dict[tuple[str, str], dict[str, Any]] = {}
    # Per-(task, model) rollup: the model-replacement comparison ("same task
    # on mimo-v2.6-pro vs glm-5.3-flash") must not merge two models that ran
    # under the same client. Sibling of by_task, not a replacement.
    by_task_model_flat: dict[tuple[str, str], dict[str, Any]] = {}
    task_clients: dict[str, set[str]] = {}

    def _bump(table: dict[Any, dict[str, Any]], key: Any, real: float, est: float) -> None:
        row = table.setdefault(key, {"real_hkd": 0.0, "est_hkd": 0.0, "calls": 0})
        row["real_hkd"] = round(row["real_hkd"] + real, 6)
        row["est_hkd"] = round(row["est_hkd"] + est, 6)
        row["calls"] += 1

    for record in records:
        model = record.get("model")
        if not isinstance(model, str) or not model:
            continue
        real = float(record.get("real_cost_hkd") or 0.0)
        raw_est = record.get("est_cost_hkd")
        est = float(raw_est) if isinstance(raw_est, (int, float)) else 0.0
        client = record.get("client") if isinstance(record.get("client"), str) else "unknown"
        kind = record.get("kind") if isinstance(record.get("kind"), str) else "unknown"
        task = record.get("task") if isinstance(record.get("task"), str) else None
        prompt = int(record.get("prompt_tokens") or 0)
        completion = int(record.get("completion_tokens") or 0)
        cached = int(record.get("cached_tokens") or 0)
        raw_duration = record.get("duration_s")
        duration = (float(raw_duration)
                    if isinstance(raw_duration, (int, float))
                    and not isinstance(raw_duration, bool) else None)
        totals["real_hkd"] = round(totals["real_hkd"] + real, 6)
        totals["est_hkd"] = round(totals["est_hkd"] + est, 6)
        totals["prompt_tokens"] += prompt
        totals["completion_tokens"] += completion
        totals["cached_tokens"] += cached
        _bump(totals["by_model"], model, real, est)
        # Per-model unit economics (cost per 1M tokens is read off these) and
        # upstream latency: the replacement question "is mimo-v2.6-pro worth
        # it against glm-5.3-flash?" needs both, per model, side by side.
        row = totals["by_model"][model]
        row["prompt_tokens"] = int(row.get("prompt_tokens", 0)) + prompt
        row["completion_tokens"] = int(row.get("completion_tokens", 0)) + completion
        row["cached_tokens"] = int(row.get("cached_tokens", 0)) + cached
        if duration is not None:
            row["duration_s"] = round(float(row.get("duration_s", 0.0)) + duration, 3)
            row["duration_calls"] = int(row.get("duration_calls", 0)) + 1
        _bump(totals["by_client"], client, real, est)
        _bump(totals["by_kind"], kind, real, est)
        if task is not None:
            _bump(by_task_flat, (task, client), real, est)
            _bump(by_task_model_flat, (task, model), real, est)
            task_clients.setdefault(task, set()).add(client)
        switched = record.get("switch_back_from")
        if isinstance(switched, str) and switched:
            totals["switch_backs"][switched] = int(totals["switch_backs"].get(switched, 0)) + 1
    for (task, client), row in by_task_flat.items():
        totals["by_task"].setdefault(task, {})[client] = row
    for (task, model), row in by_task_model_flat.items():
        totals["by_task_model"].setdefault(task, {})[model] = row
    totals["task_redo"] = sorted(task for task, clients in task_clients.items() if len(clients) >= 2)
    return totals


class Handler(BaseHTTPRequestHandler):
    server_version = "llm-router-control/1"

    def _send(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            return data if isinstance(data, dict) else {}
        except (ValueError, UnicodeDecodeError):
            return {}

    def log_message(self, fmt: str, *args: Any) -> None:  # keep logs quiet
        pass

    def _authorized(self) -> bool:
        """Self-authentication, independent of the nginx path token (m3).

        With CONTROL_TOKEN configured, EVERY request needs
        `Authorization: Bearer $CONTROL_TOKEN`. With no token configured the
        service fails closed on the write path: reads are served (the same
        data the proxy already writes to a shared volume), but every write
        is refused, so a misconfigured deploy cannot stand up an
        unauthenticated control plane.
        """
        header = self.headers.get("Authorization") or ""
        if not CONTROL_TOKEN:
            return True  # read-only mode; do_POST applies its own refusal
        expected = f"Bearer {CONTROL_TOKEN}"
        return hmac.compare_digest(header, expected)

    def do_GET(self) -> None:
        if not self._authorized():
            self._send(401, {"error": "unauthorized"})
            return
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send(200, health())
            return
        if parsed.path == "/decisions":
            params = parse_qs(parsed.query)
            try:
                limit = max(1, min(500, int(params.get("n", ["50"])[0])))
            except ValueError:
                limit = 50
            self._send(200, {"decisions": tail_decisions(limit)})
            return
        if parsed.path == "/learned":
            self._send(200, {"present": os.path.exists(LEARNED_PATH), "learned": _load_learned()})
            return
        if parsed.path == "/usage":
            # Receipt-based spend rollup (real vs estimate, per model/client/
            # kind). Also exposes the live per-session aggregate so the
            # dashboard can join a decision row to its session's real spend.
            params = parse_qs(parsed.query)
            try:
                limit = max(1, min(20000, int(params.get("n", ["2000"])[0])))
            except ValueError:
                limit = 2000
            self._send(200, {"summary": usage_summary(limit),
                             "sessions": _load_usage_state().get("sessions", {})})
            return
        if parsed.path == "/receipts":
            params = parse_qs(parsed.query)
            try:
                limit = max(1, min(500, int(params.get("n", ["50"])[0])))
            except ValueError:
                limit = 50
            self._send(200, {"receipts": tail_usage(limit)})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not CONTROL_TOKEN or not self._authorized():
            # No token configured = read-only mode: every write is refused
            # so a misconfigured deploy fails closed (review m3).
            self._send(401, {"error": "unauthorized"})
            return
        parsed = urlparse(self.path)
        body = self._body()
        try:
            if parsed.path == "/lease":
                model = body.get("model")
                if not isinstance(model, str) or not model:
                    raise ValueError("model is required")
                result = set_lease(
                    model,
                    body.get("requests"),
                    body.get("seconds"),
                    str(body.get("set_by") or "mcp:lease"),
                )
                self._send(200, result)
                return
            if parsed.path == "/lease/clear":
                self._send(200, clear_lease())
                return
            if parsed.path == "/weights":
                self._send(200, set_weights(body))
                return
            if parsed.path == "/learning":
                if "enabled" in body:
                    self._send(200, set_learning(bool(body["enabled"])))
                    return
                if "reset" in body:
                    self._send(200, reset_learned(str(body["reset"])))
                    return
                raise ValueError("learning needs {enabled} or {reset}")
            if parsed.path == "/budget":
                if "hkd" in body:
                    self._send(200, set_budget(body["hkd"]))
                    return
                if body.get("clear"):
                    self._send(200, clear_budget())
                    return
                raise ValueError("budget needs {hkd} or {clear: true}")
            if parsed.path == "/reset/learned":
                self._send(200, reset_learned(str(body.get("scope", "all"))))
                return
            if parsed.path == "/alarm/clear":
                self._send(200, bump_alarm_epoch())
                return
            self._send(404, {"error": "not found"})
        except ValueError as error:
            self._send(400, {"error": str(error)})
        except Exception as error:  # noqa: BLE001 - report, never crash the service
            self._send(500, {"error": f"{type(error).__name__}: {error}"})


def main() -> None:
    os.makedirs(STATE_DIR, exist_ok=True)
    # Bind host is container-relevant only: inside the Docker network the
    # port publish (127.0.0.1:4010 on the HOST) is what keeps this private.
    # Binding 127.0.0.1 in-container makes the published port unreachable
    # from the host loopback, so compose overrides it to 0.0.0.0.
    host = os.environ.get("CONTROL_HOST", "127.0.0.1")
    server = ThreadingHTTPServer((host, PORT), Handler)
    print(f"control service on {host}:{PORT}, state dir {STATE_DIR}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
