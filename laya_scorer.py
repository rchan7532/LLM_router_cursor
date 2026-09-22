"""
Local shadow scorer using convaiinnovations/laya.

This module is entirely off the request path. routing_policy.py may enqueue
a signature text here; a background thread scores it against the task-kind
taxonomy and writes the result to an in-RAM cache and a telemetry log. By
default the control-plane laya weight is 0, so the scores influence nothing
until the operator explicitly enables them.

The scorer is deliberately optional: if the `laya` package is not installed,
if the checkpoint download fails, or if inference raises, routing continues
identically and the only effect is that no shadow log line is written.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import tempfile
import threading
import time
from typing import Any

try:
    import laya
except ImportError:  # pragma: no cover - optional dependency
    laya = None  # type: ignore[assignment]

try:
    from control_store import _default_state_dir
except Exception:  # pragma: no cover - keep scorer self-contained
    def _default_state_dir() -> str:  # type: ignore[misc]
        if os.name != "nt" and os.path.isdir("/app/logs"):
            return "/app/logs"
        return tempfile.gettempdir()


LAYA_SHADOW_PATH = os.environ.get(
    "LAYA_SHADOW_PATH",
    os.path.join(
        os.environ.get("POLICY_STATE_DIR") or _default_state_dir(),
        "laya_shadow.jsonl",
    ),
)
LAYA_MAX_TEXT = 400  # keep well inside the English checkpoint's 512-token budget
LAYA_KINDS = (
    "code_edit", "code_gen", "refactor", "debug", "review", "design",
    "explain", "bulk", "writing", "factual", "agentic",
)

_agent: Any = None
_lock = threading.Lock()
_queue: queue.Queue[str] = queue.Queue()
_cache: dict[str, dict[str, float]] = {}
_thread: threading.Thread | None = None
_stop = threading.Event()


def _key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:24]


def _load_agent() -> Any:
    """Lazy load the laya checkpoint; returns None on any failure."""
    global _agent
    if _agent is not None:
        return _agent
    if laya is None:
        return None
    with _lock:
        if _agent is not None:
            return _agent
        try:
            _agent = laya.load("convaiinnovations/laya")
        except Exception:  # noqa: BLE001 - fail open, keep routing untouched
            _agent = None
    return _agent


def _score(text: str) -> dict[str, float]:
    """Score one text against every task kind, returning 0..1 per kind."""
    agent = _load_agent()
    if agent is None:
        return {}

    truncated = text[:LAYA_MAX_TEXT]
    state = {"text": truncated}
    questions: dict[str, Any] = {}
    for kind in LAYA_KINDS:
        questions[kind] = {
            "type": "score",
            "instructions": f"How strongly is this request about {kind.replace('_', ' ')}?",
            "criteria": ["not at all", "slightly", "strongly"],
        }

    try:
        result = agent.predict(state, questions)
    except Exception:  # noqa: BLE001
        return {}

    answers = result.get("answers") if isinstance(result, dict) else None
    if not isinstance(answers, dict):
        return {}

    scores: dict[str, float] = {}
    for kind in LAYA_KINDS:
        ans = answers.get(kind)
        if isinstance(ans, dict):
            score = ans.get("score")
        else:
            score = None
        if isinstance(score, (int, float)):
            # criteria has 3 items -> score range is 0..2
            scores[kind] = max(0.0, min(1.0, float(score) / 2.0))
        else:
            scores[kind] = 0.0
    return scores


def _write_log(text: str, scores: dict[str, float]) -> None:
    """Append a shadow-scorer telemetry line. No raw prompt text is stored."""
    try:
        directory = os.path.dirname(LAYA_SHADOW_PATH)
        if directory:
            os.makedirs(directory, exist_ok=True)
        record = {
            "ts": time.time(),
            "key": _key(text),
            "kinds": list(scores.keys()),
            "scores": {kind: round(value, 4) for kind, value in scores.items()},
        }
        with open(LAYA_SHADOW_PATH, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
    except Exception:  # noqa: BLE001 - telemetry must not break routing
        pass


def _worker() -> None:
    """Background thread that drains the scoring queue."""
    while not _stop.is_set():
        try:
            text = _queue.get(timeout=1.0)
        except queue.Empty:
            continue
        try:
            scores = _score(text)
            if scores:
                _cache[_key(text)] = scores
                _write_log(text, scores)
        except Exception:  # noqa: BLE001 - worker must survive every item
            pass


def _ensure_worker() -> None:
    """Start the daemon worker once."""
    global _thread
    if _thread is not None and _thread.is_alive():
        return
    with _lock:
        if _thread is not None and _thread.is_alive():
            return
        _thread = threading.Thread(target=_worker, daemon=True)
        _thread.start()


def maybe_enqueue(text: str, kind: str) -> None:
    """Queue a signature text for background scoring if not already cached.

    `kind` is accepted for API symmetry but is not persisted; only the
    derived scores are stored.
    """
    if laya is None:
        return
    key = _key(text)
    if key in _cache:
        return
    _ensure_worker()
    try:
        _queue.put(text, block=False)
    except queue.Full:  # pragma: no cover - unbounded queue never raises this
        pass


def get_scores(text: str) -> dict[str, float] | None:
    """Read cached scores for a signature text, or None on a cache miss."""
    return _cache.get(_key(text))


def shutdown() -> None:
    """Best-effort stop signal for the background worker (tests only)."""
    _stop.set()
