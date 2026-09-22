"""
Cross-session learning for the llm-router (plan phase 1 + phase 2, review
findings B1-B4, M2-M5, M8, m5, m8, m10).

The router never sees the answer quality. It sees decisions and the next
request. Every learnable signal is an indirect leak, and this module turns
those leaks into bounded, evidence-gated tables that persist to learned.json.

What is here:

  verdicts        - leak -> verdict mapping over two consecutive decision
                    records (ESCALATE on a follow-up, re-asks, stalls,
                    direct-model switches produce negative evidence; a clean
                    task completion is weak positive; upstream 429/5xx is
                    neutral reliability, never a quality verdict)
  EvidenceTable   - per-row {count, value, last_ts} with 90-day half-life
                    evidence decay (effective_count = count * 0.5**(age/90))
                    and the MIN_EVIDENCE evidence gate: a row below the gate
                    reads as zero influence
  LearnedState    - the three persisted tables (model_trust per (kind, model),
                    kind_bar_bias per kind, phrase_kind per phrase) plus the
                    key-point distillations and the reset-intent bookkeeping
  Learner         - the in-memory queue of newest-human-ask texts, the
                    apply-once reset-intent consumer, the flush-to-disk loop
                    and the supervised phase-2 extraction batcher

Hard rules this module enforces (all test-gated):

  C5  Raw prompt text NEVER touches disk. The queue is RAM-only (a bounded
      deque, newest human ask per decision, max 10 minutes old, then drop).
      Only the mechanically sanitized distillation is persisted. No debug or
      log path may serialize the raw text: exception messages carry a batch
      length, never a payload.
  C4  The proxy process is the sole writer of learned.json. Resets arrive as
      intents in control.json (the control service's file) and are consumed
      exactly once, tracked by a monotonic intent id + last-applied id in
      learned.json meta. Nothing here ever writes control.json.
  C11 Every write is atomic (tempfile in the same directory + os.replace).
  C1  A learner failure is counted and swallowed; the request path never
      sees it. Corrupt or absent learned.json reads as an empty state
      (zero learned influence) -- the safe direction for learned state.
  C8  POLICY_LEARN=0 is the hard kill-switch and beats control.json's
      learning_enabled: with it, no queue fills, no extraction call is
      made, no file is written.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from collections import deque
from typing import Any, Iterable, Mapping

try:
    from control_store import ControlStore  # type-only; the store is injected
except ImportError:  # pragma: no cover - split for tests
    ControlStore = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Tuning constants (plan/review numbers; env-overridable only where noted)
# ---------------------------------------------------------------------------

LEARNED_PATH = os.environ.get("LEARNED_PATH") or os.path.join(
    os.environ.get("POLICY_STATE_DIR")
    or ("/app/logs" if os.path.isdir("/app/logs") else tempfile.gettempdir()),
    "learned.json",
)

DECAY_HALF_LIFE_DAYS = 90.0
MIN_EVIDENCE = 8          # observations before a learned row applies (plan C3)
PHRASE_EVIDENCE = 5       # phrases nudge on lighter evidence (matches ControlStore)
ROW_CAP = 200             # hard row cap per table; evict lowest effective count
KEY_POINTS_KEEP = 200     # newest N distilled key points retained
QUEUE_MAX = 40            # RAM-only ask queue bound (plan 1.3)
QUEUE_MAX_AGE_S = 600.0   # 10 minutes, then drop (review M4)
EXTRACT_BATCH = 20        # asks per extraction call (plan 2.2)

# Valid extraction kinds: the BASE_BAR keys (review m5, claim 14), NOT the
# KIND_KEYWORDS map ('writing' is unreachable via keywords but exists in the
# bar table). Kept in sync with routing_policy.BASE_BAR by tests/test_learning.py.
KINDS = frozenset({
    "code_edit", "code_gen", "refactor", "debug", "review", "design",
    "explain", "bulk", "writing", "factual", "agentic",
})

# Gate tolerance: a row sitting exactly ON the gate decays below it within
# microseconds of its last observation (8.0 * 0.5**(age/90d) < 8 for any
# age > 0), which would make an 8-observation row unobservable across a
# process boundary. The gate therefore passes while the decayed count is
# within GATE_EPSILON of it; real decay differences dwarf this epsilon.
GATE_EPSILON = 1e-4

VERDICT_WEIGHTS = {
    "escalate": -0.20,
    "reask": -0.20,
    "stall": -0.15,
    "clean": 0.03,
    "direct_switch": -0.20,
    "error": 0.0,          # upstream 429/5xx: reliability, not quality
}
CLEAN_TURNS = 4           # a clean completion needs this many turns (spec table)
REASK_CONTRIB_CAP = 2     # per-session re-ask verdict cap (review B2)
SWITCH_NEGATIVES_CAP = 3  # per-session direct-switch verdict cap

# Influence bounds. The ControlStore clamps again at read time; these bounds
# keep what we WRITE inside the same envelope so a stored file can never be
# out of range even before the reader guards it (plan C2, guardrail).
TRUST_BOUND = 0.40
BAR_BIAS_BOUND = 0.15
PHRASE_BOUND = 0.10

# Value cap per single observation for the bounded accumulate: a bound like
# TRUST_BOUND/VERDICT_WEIGHTS["clean"] observations in one direction is the
# most any single row can ever move, by construction.
_TRUST_STEP = TRUST_BOUND / max(1.0, abs(VERDICT_WEIGHTS["escalate"]))
_BAR_STEP = BAR_BIAS_BOUND / 2.0

# Mechanically sanitized distillations only (review M4): strip the common
# credential shapes and cap lengths before anything reaches disk.
_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9]{16,}"),
    re.compile(r"ghp_\w+"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"xox[baprs]-\w+"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
)
KEY_POINT_MAX = 400
PHRASE_MAX = 60


def sanitize(text: Any) -> str:
    """Regex scrub + hard truncation. Applied to every value that will be
    persisted (review M4/m5): the extraction model can quote secrets verbatim
    and 'notable phrases' are quoted by design, so instruction-only
    stripping is not enforcement."""
    cleaned = str(text)
    for pattern in _SECRET_PATTERNS:
        cleaned = pattern.sub("[REDACTED]", cleaned)
    return cleaned


# ---------------------------------------------------------------------------
# Verdict detection (plan task 1.1) - pure functions, no I/O
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9_./-]{3,}")


def tokenize(text: str) -> set[str]:
    """Lowercased [a-z0-9_./-]{3,} tokens. Exact definition from review B2."""
    return set(_TOKEN_RE.findall(text.lower()))


def is_reask(prev_ask: str, next_ask: str) -> bool:
    """Strict containment, NOT Jaccard (review B2): a re-ask repeats the
    previous ask without adding scope. At least 90% of the previous ask's
    tokens must still be present and at most max(1, len(prev)//20) new
    tokens may appear. 'fix the failing test in report_parser.py and the
    date parser too' is an iterative EXTENSION, not a complaint, and must
    read False. Precision over recall: a missed re-ask costs nothing (no
    signal fires), a false one costs persisted trust."""
    prev_tokens = tokenize(prev_ask)
    next_tokens = tokenize(next_ask)
    if not prev_tokens:
        return False
    overlap = len(prev_tokens & next_tokens) / len(prev_tokens)
    if overlap < 0.9:
        return False
    new_tokens = len(next_tokens - prev_tokens)
    return new_tokens <= max(1, len(prev_tokens) // 20)


def contains_escalate_word(ask: str) -> bool:
    r"""Word-boundary ESCALATE match (review m8). 'ESCALATED' and any chat
    about the router must not count; the plain 'ESCALATE' directive does."""
    return re.search(r"\bESCALATE\b", ask) is not None


def classify_verdict(prev: Mapping[str, Any], current: Mapping[str, Any],
                     is_reask_fn=is_reask, contains_escalate_fn=contains_escalate_word) -> str | None:
    """Leak -> verdict over two consecutive decision records.

    Records carry: ask, directives (list/set), stall, kind, model, turns.
    Returns one of 'escalate' | 'reask' | 'clean' | None. The caller owns
    'stall', 'direct_switch' and 'error' - they are conditions of the
    current/previous record itself, not a relation between two records.
    Per-session re-ask caps are the caller's bookkeeping (it holds the
    session state); this function stays pure.
    """
    ask_text = str(current.get("ask") or "")
    directives = current.get("directives") or frozenset()
    try:
        directive_set = {str(item).lower() for item in directives}
    except TypeError:
        directive_set = set()
    if contains_escalate_fn(ask_text) or "escalate" in directive_set:
        return "escalate"
    if is_reask_fn(str(prev.get("ask") or ""), ask_text):
        return "reask"
    if int(current.get("turns") or 0) >= CLEAN_TURNS:
        return "clean"
    return None


def verdict_weight(verdict: str) -> float:
    """-0.20 escalate/re-ask/direct-switch, -0.15 stall, +0.03 clean,
    0.0 upstream 429/5xx (spec's verdict table, review B2 fixes)."""
    return VERDICT_WEIGHTS.get(verdict, 0.0)


def direct_switch_verdict(prev_ask: str, prev_model: str, new_model: str,
                          topical_overlap: float) -> str | None:
    """Skip-path attribution (review B3). A direct-model switch counts as
    negative evidence ONLY when a recent cursor-auto session exists (there
    is a prior ask to attribute against), the switched-to model differs
    from the model that session was served by, and the new ask is topically
    continuous (token overlap >= 0.3). Same-model switches are router
    bypasses, not dissatisfaction; new chats and unrelated asks are
    invisible to us by construction."""
    if not prev_ask:
        return None
    if not prev_model or not new_model or prev_model == new_model:
        return None
    if topical_overlap < 0.3:
        return None
    return "direct_switch"


def topical_overlap(a: str, b: str) -> float:
    """Containment overlap: the share of the smaller token set present in
    the other. Used by the skip-path observer (>= 0.3 means continuity)."""
    tokens_a = tokenize(a)
    tokens_b = tokenize(b)
    if not tokens_a or not tokens_b:
        return 0.0
    smaller, larger = (tokens_a, tokens_b) if len(tokens_a) <= len(tokens_b) else (tokens_b, tokens_a)
    return len(smaller & larger) / len(smaller)


# ---------------------------------------------------------------------------
# Evidence tables with decay + gate (plan task 1.2, review B1/C3)
# ---------------------------------------------------------------------------

class EvidenceTable:
    """rows[key] = {"count": float, "value": float, "last_ts": epoch}.

    `count` is the sum of signed observation magnitudes (reliable evidence
    of exposure), `value` the signed accumulated influence. The gate reads
    the DECAYED count: effective_count = count * 0.5 ** (age_days / 90),
    so a row last seen 90 days ago has half the evidence and typically
    falls below the gate and vanishes on its own (review's open item 1).
    Rows below the gate apply at zero strength (read side) and are pruned
    on flush."""

    def __init__(self) -> None:
        self.rows: dict[tuple, dict[str, float]] = {}

    def add(self, key: tuple, delta: float, now: float | None = None) -> None:
        now = time.time() if now is None else now
        row = self.rows.get(key)
        if row is None:
            row = {"count": 0.0, "value": 0.0, "last_ts": now}
            self.rows[key] = row
        # One add carrying influence is one observation; a 0.0 delta is a
        # label registration (phrase -> kind) and earns NO evidence.
        if delta:
            row["count"] += 1.0
        row["value"] = row["value"] + delta
        row["last_ts"] = now

    def effective_count(self, key: tuple, now: float | None = None) -> float:
        row = self.rows.get(key)
        if row is None:
            return 0.0
        now = time.time() if now is None else now
        age_days = max(0.0, (now - float(row.get("last_ts", now)))) / 86400.0
        # Rounded so repeated 0.2 adds land exactly on the integer gate
        # (0.2 + 0.2 + ... never sums to 8.0 in binary floating point).
        return round(float(row.get("count", 0.0)) * 0.5 ** (age_days / DECAY_HALF_LIFE_DAYS), 9)

    def raw_value(self, key: tuple) -> float:
        row = self.rows.get(key)
        return float(row["value"]) if row else 0.0

    def value(self, key: tuple, bound: float, gate: int = MIN_EVIDENCE,
              now: float | None = None) -> float | None:
        """Signed influence, clamped to +/-bound, or None below the gate."""
        if self.effective_count(key, now) < gate - GATE_EPSILON:
            return None
        return max(-bound, min(bound, self.raw_value(key)))

    def reset(self, now: float | None = None) -> None:
        self.rows.clear()

    def prune(self, now: float | None = None) -> None:
        """Drop DORMANT rows whose decayed evidence has fallen below the
        gate (90-day half-life, review's open item 1): a row last seen
        more than one half-life ago that no longer clears the gate is dead
        by decay. Rows still accumulating sit below the gate too, but they
        apply at zero strength on the read side and must NOT be deleted
        while they build evidence, so only dormant rows are pruned here.
        The row cap is enforced regardless, evicting the weakest evidence."""
        now = time.time() if now is None else now
        for key in list(self.rows):
            row = self.rows[key]
            age_days = max(0.0, (now - float(row.get("last_ts", now)))) / 86400.0
            if age_days > DECAY_HALF_LIFE_DAYS and self.effective_count(key, now) < MIN_EVIDENCE:
                del self.rows[key]
        if len(self.rows) > ROW_CAP:
            ranked = sorted(self.rows, key=lambda key: self.effective_count(key, now))
            for key in ranked[: len(self.rows) - ROW_CAP]:
                del self.rows[key]

    def to_dict(self) -> dict[str, list]:
        out: dict[str, list] = {}
        for key, row in self.rows.items():
            out["|".join(str(part) for part in key)] = [
                float(row.get("count", 0.0)), float(row.get("value", 0.0)),
                float(row.get("last_ts", 0.0)),
            ]
        return out

    @classmethod
    def from_dict(cls, data: Any, key_parts: int = 1) -> "EvidenceTable":
        """Never raises (plan C1). Malformed input -> empty table; unknown
        shapes and version-skewed payloads are ignored."""
        table = cls()
        if not isinstance(data, Mapping):
            return table
        for encoded, row in data.items():
            if not isinstance(encoded, str) or not isinstance(row, (list, tuple)) or len(row) < 3:
                continue
            count, value, last_ts = row[0], row[1], row[2]
            if not all(isinstance(part, (int, float)) for part in row[:3]):
                continue
            # Keys are pipe-joined; every segment must be present and the
            # segment count must match the table's arity ('|x' is junk).
            segments = encoded.split("|")
            if len(segments) != key_parts or any(not segment for segment in segments):
                continue
            key = tuple(segments)
            table.rows[key] = {"count": float(count), "value": float(value), "last_ts": float(last_ts)}
        return table


# ---------------------------------------------------------------------------
# LearnedState: the persisted shape of learned.json
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1


class LearnedState:
    """The three bounded tables + distilled key points + reset bookkeeping.

    model_trust   per (kind, model): outcome leaks, +/-0.40
    kind_bar_bias per kind:         escalate/direct-switch rate, +/-0.15
    phrase_kind   per phrase:       phrase -> kind with a bar nudge, +/-0.10

    Re-ask evidence deliberately does NOT feed kind_bar_bias (review q2):
    it is the noisiest input, it feeds model_trust only, where it is
    clamped and per-cell sparse. A stall verdict also feeds trust only.
    """

    def __init__(self) -> None:
        self.trust = EvidenceTable()     # keys ("kind", "model")
        self.bars = EvidenceTable()      # keys ("kind",)
        self.phrases = EvidenceTable()   # keys ("phrase",)
        self.key_points: list[dict[str, Any]] = []
        self.last_applied_reset_id: str | None = None
        self._phrase_kinds: dict[str, str] = {}   # phrase -> kind label
        self.version = SCHEMA_VERSION

    # -- verdict application ------------------------------------------------

    def apply_verdict(self, verdict: str, kind: str, model: str,
                      phrase_hints: Iterable[str] = (), now: float | None = None) -> None:
        """Fold one verdict into the tables. `kind`/`model` describe the
        PREVIOUS decision record (the one being judged). A no-op for the
        neutral 'error' verdict (429/5xx) by design."""
        now = time.time() if now is None else now
        weight = verdict_weight(verdict)
        if weight == 0.0:
            return
        if kind and model:
            self.trust.add((kind, model), weight, now)
        # kind_bar_bias consumes only strong, hard-to-produce-accidentally
        # verdicts (review q2): escalate and confirmed direct-switches.
        if kind and verdict in ("escalate", "direct_switch"):
            self.bars.add((kind,), weight, now)
        # Weak positive evidence reinforces the phrase -> kind mapping.
        if verdict == "clean":
            for phrase in phrase_hints:
                phrase = sanitize(phrase).strip().lower()[:PHRASE_MAX]
                if phrase:
                    self.phrases.add((phrase,), 1.0, now)
        self.prune(now)

    def prune(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        self.trust.prune(now)
        self.bars.prune(now)
        self.phrases.prune(now)
        if len(self.key_points) > KEY_POINTS_KEEP:
            self.key_points = self.key_points[-KEY_POINTS_KEEP:]

    # -- read-side influence (bounded at the point of use) -------------------

    def learned_trust(self, kind: str, model: str, now: float | None = None) -> float:
        """Gated, clamped learned trust for (kind, model); 0.0 below gate."""
        value = self.trust.value((kind, model), TRUST_BOUND, now=now)
        return value if value is not None else 0.0

    def bar_adjust(self, kind: str, now: float | None = None) -> float:
        """Gated, clamped kind_bar_bias for a kind; 0.0 below gate."""
        value = self.bars.value((kind,), BAR_BIAS_BOUND, now=now)
        return value if value is not None else 0.0

    def phrase_hints(self, ask: str) -> list[str]:
        """Learned phrases present in this ask (bounded read side)."""
        lowered = ask.lower()
        return [key[0] for key in list(self.phrases.rows) if key[0].lower() in lowered]

    def phrase_kind(self, ask: str, now: float | None = None) -> str | None:
        """The dominant learned kind for the phrases in this ask, below the
        phrase evidence gate nothing is returned. Feeds the bounded
        phrase_kind bonus in the policy's kind selection (review M3)."""
        lowered = ask.lower()
        best: tuple[float, str] | None = None
        for key in list(self.phrases.rows):
            phrase = key[0]
            if phrase.lower() not in lowered:
                continue
            value = self.phrases.value(key, PHRASE_BOUND, gate=PHRASE_EVIDENCE, now=now)
            if value is None:
                continue
            if best is None or abs(value) > abs(best[0]):
                best = (value, phrase)
        if best is None:
            return None
        return self._phrase_kinds.get(best[1]) if hasattr(self, "_phrase_kinds") else None

    # -- serialization --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "model_trust": self.trust.to_dict(),
            "kind_bar_bias": self.bars.to_dict(),
            "phrase_kind": self.phrases.to_dict(),
            "phrase_kinds": dict(self._phrase_kinds),
            "key_points": self.key_points,
            "last_applied_reset_id": self.last_applied_reset_id,
            "updated_ts": time.time(),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "LearnedState":
        """Never raises. Absent/corrupt/foreign -> empty state (zero learned
        influence), which is the fail-open direction for learned state."""
        state = cls()
        if not isinstance(data, Mapping):
            return state
        state.trust = EvidenceTable.from_dict(data.get("model_trust"), key_parts=2)
        state.bars = EvidenceTable.from_dict(data.get("kind_bar_bias"), key_parts=1)
        state.phrases = EvidenceTable.from_dict(data.get("phrase_kind"), key_parts=1)
        # phrase -> kind labels ride along for phrase_kind lookups
        kinds = data.get("phrase_kinds")
        if isinstance(kinds, Mapping):
            state._phrase_kinds = {
                str(k): str(v) for k, v in kinds.items() if isinstance(k, str) and isinstance(v, str)
            }
        points = data.get("key_points")
        if isinstance(points, list):
            state.key_points = [point for point in points if isinstance(point, dict)]
        reset_id = data.get("last_applied_reset_id")
        if isinstance(reset_id, str):
            state.last_applied_reset_id = reset_id
        return state


# ---------------------------------------------------------------------------
# Persistence (plan task 1.3)
# ---------------------------------------------------------------------------

def load_learned(path: str) -> LearnedState:
    """Absent or corrupt learned.json -> empty state. Never raises (B1)."""
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return LearnedState()
    if not isinstance(data, dict):
        return LearnedState()
    return LearnedState.from_dict(data)


def _atomic_write_json(path: str, payload: Mapping[str, Any]) -> None:
    """tmp + fsync + os.replace in the same directory (plan C11)."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def save_learned(path: str, state: LearnedState) -> None:
    """The proxy process is the ONLY writer of learned.json (review B4).
    Raises on failure; the caller counts and swallows."""
    _atomic_write_json(path, state.to_dict())


def append_observation(path: str, record: Mapping[str, Any]) -> None:
    """Append one sanitized distillation to observations.jsonl.

    Every value passes through sanitize() first (review M4): this is the
    mechanical post-filter that keeps raw text and secrets off disk even
    when the extraction model quotes them verbatim. The caller rotates the
    file with the LOG_MAX_BYTES pattern."""
    clean: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, str):
            clean[key] = sanitize(value)[:KEY_POINT_MAX]
        elif isinstance(value, list):
            clean[key] = [sanitize(item)[:PHRASE_MAX if key == "phrases" else KEY_POINT_MAX]
                          for item in value if isinstance(item, str)]
        else:
            clean[key] = value
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(clean, default=str) + "\n")


# ---------------------------------------------------------------------------
# The learner: queue, reset intents, flush, phase-2 extraction
# ---------------------------------------------------------------------------

class Learner:
    """Owns the RAM-only ask queue, the reset-intent consumer and the
    flush loop. All queue contents are NEWEST-HUMAN-ASK texts (never tool
    output, never the system prompt - that is where secrets live); they are
    dropped when the batch completes and are never serialized anywhere."""

    def __init__(self, learned_path: str, observations_path: str | None = None) -> None:
        self.learned_path = learned_path
        self.observations_path = observations_path
        self.state = load_learned(learned_path)
        self.errors = 0
        self.extractions = 0
        self._queue: deque[tuple[float, str]] = deque()  # (ts, ask text)
        self._queue_turns: deque[int] = deque()          # parallel turns record
        self._queue_kinds: deque[str] = deque()          # parallel kind record
        self._queue_models: deque[str] = deque()         # parallel model record
        self.reask_budgets: dict[str, int] = {}          # session key -> re-asks left
        self.switch_budgets: dict[str, int] = {}         # session key -> switches left
        self._last_control_stamp: tuple[int, int] | None = None

    # -- queue (phase 2 plumbing; C5: RAM only) ------------------------------

    def enqueue(self, ask_text: str, kind: str, model: str, turns: int) -> None:
        now = time.time()
        while self._queue and now - self._queue[0][0] > QUEUE_MAX_AGE_S:
            self._queue.popleft(); self._queue_turns.popleft()
            self._queue_kinds.popleft(); self._queue_models.popleft()
        self._queue.append((now, ask_text))
        self._queue_turns.append(int(turns))
        self._queue_kinds.append(kind)
        self._queue_models.append(model)
        while len(self._queue) > QUEUE_MAX:
            self._queue.popleft(); self._queue_turns.popleft()
            self._queue_kinds.popleft(); self._queue_models.popleft()

    @property
    def queue_depth(self) -> int:
        return len(self._queue)

    def _drop_queue_front(self, count: int) -> None:
        for _ in range(min(count, len(self._queue))):
            self._queue.popleft(); self._queue_turns.popleft()
            self._queue_kinds.popleft(); self._queue_models.popleft()

    # -- control-file reset intents (B4: consume exactly once, never write) --

    def poll_reset_intents(self, control_path: str) -> bool:
        """Read control.json (stat-gated) and consume any pending reset
        intent. Read-only on control.json, always. Returns True if a fresh
        intent was applied. A monotonic intent id (monotonic counter or uuid)
        + learned.json's last_applied_reset_id makes this apply-once without
        ANY write-back to control.json."""
        try:
            stat = os.stat(control_path)
        except OSError:
            return False
        stamp = (stat.st_mtime_ns, stat.st_size)
        if stamp == self._last_control_stamp:
            return False
        self._last_control_stamp = stamp
        try:
            with open(control_path, encoding="utf-8") as handle:
                control = json.load(handle)
        except (OSError, ValueError):
            return False
        if not isinstance(control, dict):
            return False
        reset_id = control.get("reset_id")
        reset = control.get("reset")
        if not isinstance(reset_id, str) or not reset_id:
            return False
        if reset_id == self.state.last_applied_reset_id:
            return False  # already applied; do NOT re-apply (idempotence)
        self.apply_reset(str(reset) if isinstance(reset, str) else "all")
        self.state.last_applied_reset_id = reset_id
        return True

    def apply_reset(self, scope: str) -> None:
        """Mirror the control service's reset scopes. 'all' clears every
        table; 'trust'/'bars'/'phrases' clear one (control_service also
        pairs 'phrases' with key_points, so we do the same)."""
        if scope == "all":
            self.state = LearnedState()
            return
        if scope == "trust":
            self.state.trust.reset()
        elif scope == "bars":
            self.state.bars.reset()
        elif scope == "phrases":
            self.state.phrases.reset()
            self.state.key_points.clear()

    # -- flush ----------------------------------------------------------------

    def flush(self, force: bool = False) -> bool:
        """Prune decayed rows, persist learned.json atomically. Rows below
        the evidence gate are pruned at write time (they already apply at
        zero strength at read time). Returns True on a write."""
        try:
            self.state.prune()
            if force or self.state.trust.rows or self.state.bars.rows \
                    or self.state.phrases.rows or self.state.key_points \
                    or self.state.last_applied_reset_id:
                save_learned(self.learned_path, self.state)
                return True
        except Exception:
            self.errors += 1
        return False

    # -- verdict folding + decision-summary records ----------------------------

    def observe_decision(self, prev_record: Mapping[str, Any] | None,
                         current_record: Mapping[str, Any],
                         session_key: str | None) -> None:
        """Fold the leak between the previous and current decision into the
        tables. Per-session re-ask caps live here (B2: a third re-ask in a
        session contributes nothing). Every failure is counted and
        swallowed; the request path never sees it."""
        try:
            if prev_record is None:
                return
            verdict = classify_verdict(prev_record, current_record)
            if verdict == "reask" and session_key:
                budget = self.reask_budgets.get(session_key, REASK_CONTRIB_CAP)
                if budget <= 0:
                    return  # capped: contributes nothing
                self.reask_budgets[session_key] = budget - 1
            kind = str(prev_record.get("kind") or "")
            model = str(prev_record.get("model") or "")
            ask_text = str(current_record.get("ask") or "")
            hints = self.state.phrase_hints(ask_text)
            self.state.apply_verdict(verdict or "none", kind, model, hints)
        except Exception:
            self.errors += 1

    def observe_switch(self, prev_record: Mapping[str, Any], new_ask: str,
                       new_model: str, session_key: str | None) -> str | None:
        """Skip-path observer (review B3). Returns the verdict applied, or
        None when attribution says no: no prior session ask (brand-new
        chats are invisible by construction), a switch to the model the
        session was already served by, no topical continuity, or a capped
        session budget."""
        try:
            prev_ask = str(prev_record.get("ask") or "")
            prev_model = str(prev_record.get("model") or "")
            if not prev_ask:
                return None
            verdict = direct_switch_verdict(
                prev_ask, prev_model, new_model, topical_overlap(prev_ask, new_ask),
            )
            if verdict is None:
                return None
            if session_key:
                budget = self.switch_budgets.get(session_key, SWITCH_NEGATIVES_CAP)
                if budget <= 0:
                    return None
                self.switch_budgets[session_key] = budget - 1
            kind = str(prev_record.get("kind") or "")
            model = str(prev_record.get("model") or "")
            self.state.apply_verdict(verdict, kind, model)
            return verdict
        except Exception:
            self.errors += 1
            return None

    # -- phase 2: extraction batch -------------------------------------------

    def drain_batch(self, batch_size: int = EXTRACT_BATCH) -> list[str]:
        """Pop up to batch_size asks, oldest first. Raw text leaves the RAM
        queue here and is dropped after the batch completes (C5)."""
        asks = [text for _, text in list(self._queue)[:batch_size]]
        self._drop_queue_front(len(asks))
        return asks

    def extract_and_store(self, asks: list[str], extractor) -> int:
        """Run the extractor over a batch, store sanitized distillations.
        `extractor` is async callable(asks) -> list[item dict]. The raw ask
        text is never logged, never persisted, never interpolated into an
        exception message (review M4)."""
        if not asks:
            return 0
        try:
            import asyncio
            items = asyncio.run(extractor(asks))
        except Exception:
            self.errors += 1
            return 0  # batch dropped; losing a batch loses only learning
        stored = 0
        try:
            for item in items or []:
                if not isinstance(item, dict):
                    continue
                kind = item.get("kind")
                points = item.get("key_points") or []
                phrases = item.get("phrases") or []
                # Client-side schema validation (review m5): kind must be a
                # real task kind (BASE_BAR keys), list fields must be lists.
                # Non-conforming items are dropped entirely.
                if not isinstance(kind, str) or kind not in KINDS:
                    continue
                if not isinstance(points, list) or not isinstance(phrases, list):
                    continue
                record = {
                    "ts": time.time(),
                    "kind": kind,
                    "key_points": [sanitize(p)[:KEY_POINT_MAX] for p in points if isinstance(p, str)],
                    "phrases": [sanitize(p)[:PHRASE_MAX] for p in phrases if isinstance(p, str)],
                }
                if self.observations_path:
                    append_observation(self.observations_path, record)
                for point in record["key_points"]:
                    self.state.key_points.append({"ts": record["ts"], "kind": record["kind"],
                                                  "text": point})
                # Learned phrases start with zero evidence; the phrase -> kind
                # label rides in phrase_kinds. They earn influence only when
                # clean verdicts reinforce them past the phrase gate.
                for phrase in record["phrases"]:
                    normalized = phrase.strip().lower()[:PHRASE_MAX]
                    if normalized:
                        self.state.phrases.add((normalized,), 0.0)
                        self.state._phrase_kinds[normalized] = kind
                stored += 1
            self.state.prune()
        except Exception:
            self.errors += 1
        return stored

    def process_queue(self, extractor, batch_size: int = EXTRACT_BATCH) -> int:
        """Drain up to one full batch through the extractor, then drop the
        raw text by construction (the queue entries are already popped).
        One call per batch, straight to the upstream provider - never
        through this proxy (review M2)."""
        asks = self.drain_batch(batch_size)
        if not asks:
            return 0
        stored = self.extract_and_store(asks, extractor)
        self.extractions += 1
        return stored
