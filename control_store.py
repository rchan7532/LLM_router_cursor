"""
Shared control-store contract for the llm-router.

Three parties touch control.json:

  - routing_policy.py   reads it (mtime-cached) before every decision
  - control_service.py  owns writes (leases, weight overrides, learning switch)
  - mcp_server.py       talks to the control service over HTTP

The policy NEVER writes and never does HTTP: it reads the file the control
service owns, and only when the (mtime, size) stamp moved. An absent file
means "no overrides" - routing is identical to a build without this module.
A corrupt file is different: the last-good snapshot stays in force (see M5
below). Both properties are load-bearing and test-enforced.

Schema (all keys optional, unknown keys ignored):

    {
      "lease": {
        "model": "openai/kimi-k2.7-code",
        "expires_ts": 1789701882.0,      # epoch seconds; the ONLY bound
        "set_by": "mcp:lease",           # audit string, shown in health
        "set_ts": 1789701802.0
      },
      "weights": {                       # overrides the W_* module constants
        "cost": 0.35, "trust": 0.30, "headroom": 0.10, "latency": 0.10
      },
      "learning_enabled": true,
      "revision": 12                     # bumped on every accepted write
    }

A lease carries one deadline in epoch seconds and nothing else (review B5).
A request-count lease is converted to that deadline by the control service at
issue time (~30 s per request, clamped to its max window) -- there is no
counter anywhere, so a container restart can neither re-arm nor lose a lease,
and the policy stays read-only on this file.

Fail-open with memory (review M5): a corrupt or torn control.json falls back
to the LAST-GOOD snapshot -- kept in RAM and persisted atomically to
`<control.json>.lastgood` on every successful read -- never to defaults,
because pristine defaults would silently re-enable learning the operator
disabled. Only an absent file reads as defaults. A failed parse records
nothing, so every following request retries the read and a repaired file
heals at once. learned.json is deliberately different: corrupt learned
tables read as zero influence (pristine), which is the safe direction for
learned state (the mode matrix in review B1).

Learning persistence (learned.json) shares the same fail-open read discipline:
  phrase_kind  -> {"phrase": [count, value, last_ts]}   phase-B row format:
                 {"phrase": {"kind": ..., "count": ..., "bias": ...}}   legacy
  kind_bar_bias -> {"debug": [count, value, last_ts]}   phase-B row format
                 {"debug": 0.05}                        legacy (clamp only)
  model_trust  -> {"debug|openai/kimi-k2.7-code": [count, value, last_ts]}
                 {"debug|openai/kimi-k2.7-code": -0.12}  legacy (clamp only)
  phrase_kinds -> {"phrase": "debug"}                   phrase -> kind labels
  key_points   -> [{"ts": ..., "kind": ..., "text": "..."}]   distilled, no raw text

Phase-B rows carry their evidence: count (observations), value (the signed
accumulated influence, already clamped by the writer), last_ts (epoch). The
reader applies the 90-day half-life DECAY to the count and the MIN_EVIDENCE
gate at read time (plan C3): effective = count * 0.5**(age_days/90); below
the gate a row applies at zero strength. Legacy bare-number entries predate
the gate and keep the clamp-only behaviour so hand-written old files still
load (the writer always emits rows now).
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from typing import Any, Mapping


def _default_state_dir() -> str:
    """Container default /app/logs when it exists AND was not mounted as a
    Windows drive path (on Windows, isdir("/app/logs") checks C:\\app\\logs,
    so a dev machine with that directory would otherwise silently receive
    learning state). A temp dir everywhere else. Override with POLICY_STATE_DIR."""
    if os.name != "nt" and os.path.isdir("/app/logs"):
        return "/app/logs"
    return tempfile.gettempdir()

# Influence bounds. Learned values nudge; they can never replace hand-tuned
# config. A reader that clamps here cannot be fed an out-of-range value even
# by a hand-edited file.
PHRASE_BAR_NUDGE = 0.10      # max |bar shift| from one learned phrase
KIND_BAR_BIAS_MAX = 0.15     # max |bar shift| from kind_bar_bias
TRUST_PERSIST_MAX = 0.40     # learned trust magnitude cap (same as runtime)
MIN_EVIDENCE = 8             # observations before a learned row applies
PHRASE_EVIDENCE = 5          # observations before a phrase nudges
KEY_POINTS_KEEP = 200        # newest N distilled key points retained
DECAY_HALF_LIFE_DAYS = 90.0  # evidence decay half-life (open item 1)
GATE_EPSILON = 1e-4          # a row exactly ON the gate still applies

EMPTY_CONTROL: dict[str, Any] = {
    "lease": None,
    "weights": None,
    "learning_enabled": True,
    "budget_hkd": None,
    "revision": 0,
}


class ControlStore:
    """Read-only, mtime-cached view of control.json + learned.json.

    One instance per process. All methods return safe values on any failure:
    a missing file, a permissions error, a truncated JSON write mid-crash, or
    a schema that drifted. The caller cannot distinguish "no overrides" from
    "overrides unreadable", which is the point: routing must not change
    because the control plane hiccuped.
    """

    __slots__ = ("_control_path", "_learned_path", "_control_stamp", "_learned_stamp",
                 "_control", "_learned", "_errors", "_control_ever_good")

    def __init__(self, control_path: str, learned_path: str) -> None:
        self._control_path = control_path
        self._learned_path = learned_path
        # (st_mtime_ns, st_size) of the last successful read, or None. A
        # corrupt read never records a stamp, so the next request retries the
        # read instead of sticking stale (review M5).
        self._control_stamp: tuple[int, int] | None = None
        self._learned_stamp: tuple[int, int] | None = None
        self._control: dict[str, Any] = dict(EMPTY_CONTROL)
        self._learned: dict[str, Any] = {}
        self._errors = 0
        # True once a successful control.json read has ever been seen (in RAM
        # or recovered from the .lastgood sidecar). Until then a corrupt file
        # has no last-good to fall back to.
        self._control_ever_good = False

    @property
    def errors(self) -> int:
        return self._errors

    # -- internal ----------------------------------------------------------

    def _reload_if_changed(self, path: str, stamp: tuple[int, int] | None,
                           fallback: dict[str, Any]) -> tuple[dict[str, Any], tuple[int, int] | None, str]:
        """Read `path` unless its (mtime_ns, size) stamp is unchanged.

        Returns (snapshot, stamp, status). status is "ok" (fresh successful
        read), "unchanged" (stamp matches; snapshot is the caller's fallback),
        "absent" (file missing or unreadable), or "corrupt" (readable but not
        valid JSON / not an object). A corrupt read records NO stamp, so the
        next call retries instead of sticking stale (review M5)."""
        try:
            stat = os.stat(path)
        except OSError:
            # Missing/unreadable is the steady state before first use.
            return fallback, (stamp if stamp is not None else None), "absent"
        new_stamp = (stat.st_mtime_ns, stat.st_size)
        if stamp is not None and new_stamp == stamp:
            return fallback, stamp, "unchanged"
        try:
            with open(path, encoding="utf-8") as handle:
                loaded = json.load(handle)
            if not isinstance(loaded, dict):
                raise ValueError("root is not an object")
        except Exception:
            # Truncated write, corrupt JSON, wrong schema. The caller keeps
            # the last good value rather than half-applying garbage.
            self._errors += 1
            return fallback, None, "corrupt"
        return loaded, new_stamp, "ok"

    def _load_lastgood(self, control_path: str) -> tuple[dict[str, Any], bool]:
        """Try to load the persisted last-good snapshot for a corrupt file.

        Returns (snapshot, ok). `ok` is False when there is none: file absent,
        unreadable, or itself corrupt. Used only on the very first successful
        read after a corrupt first sight, when RAM holds no good snapshot yet."""
        lastgood_path = control_path + ".lastgood"
        try:
            with open(lastgood_path, encoding="utf-8") as handle:
                loaded = json.load(handle)
        except Exception:
            return {}, False
        if not isinstance(loaded, dict):
            return {}, False
        return loaded, True

    def _persist_lastgood(self, control_path: str, payload: Mapping[str, Any]) -> None:
        """Atomically record a known-good control.json (review M5)."""
        lastgood_path = control_path + ".lastgood"
        try:
            tmp = lastgood_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(dict(payload), handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, lastgood_path)
        except Exception:
            pass  # best-effort durability; RAM still holds the snapshot

    def _refresh_control(self) -> None:
        loaded, stamp, status = self._reload_if_changed(self._control_path, self._control_stamp, self._control)
        if status == "ok":
            self._control = loaded
            if not isinstance(self._control.get("learning_enabled"), bool):
                # A valid file may simply not express the flag; the
                # documented default is enabled. (Only the corrupt-fallback
                # path below forces it off - review M5.)
                self._control["learning_enabled"] = True
            # Every successful read atomically refreshes the persisted
            # last-good snapshot the next cold start will trust (review M5).
            self._persist_lastgood(self._control_path, self._control)
            self._control_ever_good = True
            self._control_stamp = stamp
            return
        if status == "corrupt":
            # Keep the last-good snapshot in RAM. On the very first read
            # there is no RAM snapshot yet, so recover from the persisted
            # last-good sidecar; only if that is missing too does the store
            # fall back to defaults — with learning_enabled forced False, so
            # corruption can never silently re-enable disabled learning
            # (review M5).
            if not self._control_ever_good:
                snapshot, ok = self._load_lastgood(self._control_path)
                if ok:
                    self._control = snapshot
                    if not isinstance(self._control.get("learning_enabled"), bool):
                        self._control["learning_enabled"] = False
                    self._control_ever_good = True
                else:
                    self._control = dict(EMPTY_CONTROL)
                    self._control["learning_enabled"] = False
            return
        # "absent": missing file reads as defaults — a deliberate toggle-off
        # is expressed in the file, not by deleting it. Stamp stays put so an
        # ephemeral miss (e.g. a replace racing the stat) does not force a
        # re-read of an unchanged file.
        if status == "unchanged":
            self._control_ever_good = True

    def _refresh_learned(self) -> None:
        loaded, stamp, status = self._reload_if_changed(self._learned_path, self._learned_stamp, self._learned)
        if status == "ok":
            self._learned = loaded
            self._learned_stamp = stamp

    # -- control.json ------------------------------------------------------

    def active_lease(self, now: float | None = None) -> Mapping[str, Any] | None:
        """The lease if its deadline is still in the future, else None.

        Stateless by construction (review B5): the only bound is `expires_ts`
        in epoch seconds, evaluated fresh on every call against the clock.
        No counter is decremented anywhere, so a proxy restart can neither
        re-arm nor lose a lease, and the policy stays read-only on this
        file. A lease without a parseable future deadline is dead."""
        self._refresh_control()
        lease = self._control.get("lease")
        if not isinstance(lease, Mapping):
            return None
        expires = lease.get("expires_ts")
        if not isinstance(expires, (int, float)):
            return None
        if (now or time.time()) >= float(expires):
            return None
        return lease

    def weights(self) -> Mapping[str, Any] | None:
        self._refresh_control()
        weights = self._control.get("weights")
        return weights if isinstance(weights, Mapping) else None

    def learning_enabled(self) -> bool:
        self._refresh_control()
        value = self._control.get("learning_enabled", True)
        # bool(...) would coerce truthy garbage to enabled; without a good
        # read there is nothing to enable, so a corrupt file fails closed
        # here too (review M5).
        return value if isinstance(value, bool) else False

    def budget_hkd(self) -> float | None:
        """Per-session spend cap in HKD. None means no cap."""
        self._refresh_control()
        value = self._control.get("budget_hkd")
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
        return None

    def laya_weight(self) -> float | None:
        """Laya shadow scorer weight, 0..1. None means not configured."""
        self._refresh_control()
        weights = self._control.get("weights")
        if isinstance(weights, Mapping):
            value = weights.get("laya")
            if isinstance(value, (int, float)):
                return float(value)
        return None

    def alarm_epoch(self) -> int | None:
        """Global alarm epoch. Bumping this in control.json makes every proxy
        drop its stale per-(model, kind) alarms on the next read."""
        self._refresh_control()
        value = self._control.get("alarm_epoch")
        return int(value) if isinstance(value, int) else None

    def control_snapshot(self) -> dict[str, Any]:
        """Raw view for diagnostics; never used for decisions."""
        self._refresh_control()
        return dict(self._control)

    # -- learned.json ------------------------------------------------------

    def learned_snapshot(self) -> dict[str, Any]:
        self._refresh_learned()
        return dict(self._learned)

    @staticmethod
    def _decayed(row_count: float, last_ts: float, now: float) -> float:
        age_days = max(0.0, (now - float(last_ts))) / 86400.0
        return round(float(row_count) * 0.5 ** (age_days / DECAY_HALF_LIFE_DAYS), 9)

    def _row(self, entry: Any) -> tuple[float, float] | None:
        """(effective_count, value) for a learned table entry, or None when
        the entry is unusable. Rows carry [count, value, last_ts] and pass
        the evidence gate after 90-day decay; legacy bare numbers bypass the
        gate (count None) and keep the clamp-only behaviour."""
        if isinstance(entry, (list, tuple)) and len(entry) >= 3 \
                and all(isinstance(part, (int, float)) for part in entry[:3]):
            count, value, last_ts = entry[0], entry[1], entry[2]
            effective = self._decayed(count, last_ts, time.time())
            if effective < MIN_EVIDENCE - GATE_EPSILON:
                return None
            return effective, float(value)
        if isinstance(entry, (int, float)):
            return None, float(entry)   # legacy: no evidence metadata
        return None

    def phrase_bias(self, ask_text: str) -> float:
        """Bar nudge from learned phrases present in this ask.

        Scans learned phrases; a phrase hit contributes its signed bias
        within +/-PHRASE_BAR_NUDGE once its evidence clears the phrase gate
        (PHRASE_EVIDENCE observations, 90-day decayed). Positive raises the
        bar (harder work than the words suggest), negative lowers it."""
        self._refresh_learned()
        phrases = self._learned.get("phrase_kind")
        if not isinstance(phrases, Mapping) or not ask_text:
            return 0.0
        lowered = ask_text.lower()
        total = 0.0
        for phrase, entry in phrases.items():
            if not isinstance(phrase, str) or phrase.lower() not in lowered:
                continue
            if isinstance(entry, Mapping):
                # Legacy dict shape: {"kind", "count", "bias"}
                count = entry.get("count", 0)
                if not isinstance(count, (int, float)) or count < PHRASE_EVIDENCE:
                    continue
                bias = entry.get("bias", 0.0)
                if isinstance(bias, (int, float)):
                    total += float(bias)
            elif isinstance(entry, (list, tuple)) and len(entry) >= 3 \
                    and all(isinstance(part, (int, float)) for part in entry[:3]):
                # Phase-B row: [count, value, last_ts] with decay
                effective = self._decayed(entry[0], entry[2], time.time())
                if effective < PHRASE_EVIDENCE - GATE_EPSILON:
                    continue
                total += float(entry[1])
            elif isinstance(entry, (int, float)):
                total += float(entry)
        return max(-PHRASE_BAR_NUDGE, min(PHRASE_BAR_NUDGE, total))

    def kind_bar_bias(self, kind: str) -> float:
        """Learned bar offset for a task kind, clamped to +/-KIND_BAR_BIAS_MAX.

        Phase-B rows must ALSO clear the evidence gate (8 observations,
        90-day decayed - plan C3); a row below the gate applies at zero
        strength. Legacy bare floats keep the clamp-only behaviour."""
        self._refresh_learned()
        bias = self._learned.get("kind_bar_bias", {})
        if not isinstance(bias, Mapping):
            return 0.0
        entry = bias.get(kind)
        if isinstance(entry, (list, tuple)):
            resolved = self._row(entry)
            if resolved is None:
                return 0.0
            return max(-KIND_BAR_BIAS_MAX, min(KIND_BAR_BIAS_MAX, float(resolved[1])))
        if isinstance(entry, (int, float)):
            return max(-KIND_BAR_BIAS_MAX, min(KIND_BAR_BIAS_MAX, float(entry)))
        return 0.0

    def persisted_trust(self, kind: str, model: str) -> float:
        """Learned trust for (kind, model), clamped to +/-TRUST_PERSIST_MAX.

        Phase-B rows must clear the evidence gate (8 observations, 90-day
        decayed - plan C3); below the gate the learned value is zero, which
        is what keeps the combined runtime+learned clamp (review M8) from
        ever being fed an ungated value."""
        self._refresh_learned()
        table = self._learned.get("model_trust", {})
        if not isinstance(table, Mapping):
            return 0.0
        key = f"{kind}|{model}"
        entry = table.get(key)
        if isinstance(entry, (list, tuple)):
            resolved = self._row(entry)
            if resolved is None:
                return 0.0
            return max(-TRUST_PERSIST_MAX, min(TRUST_PERSIST_MAX, float(resolved[1])))
        if isinstance(entry, (int, float)):
            return max(-TRUST_PERSIST_MAX, min(TRUST_PERSIST_MAX, float(entry)))
        return 0.0


def default_store() -> ControlStore:
    """The store the policy uses. Paths are env-overridable so tests and the
    local runner can point at temp dirs without touching production files.

    In the Docker compose the proxy reads control.json from a read-only
    mount (/control) while it owns learned.json and the decision log on
    /app/logs. CONTROL_PATH takes precedence for control.json so the two
    files can live on different volumes (review B4)."""
    base = os.environ.get("POLICY_STATE_DIR") or _default_state_dir()
    control_path = os.environ.get("CONTROL_PATH") or os.path.join(base, "control.json")
    return ControlStore(
        control_path=control_path,
        learned_path=os.path.join(base, "learned.json"),
    )
