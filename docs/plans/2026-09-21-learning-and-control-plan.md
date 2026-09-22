# Learning + Control Surface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **No git in this repo.** Ignore every commit step in the sub-skills. The gate after each task is the test command shown — run it and see it pass; that replaces the commit.
>
> **Front door change (2026-09-21):** the VPS front door is **nginx**, not Caddy. The `Caddyfile` no longer exists; the server block lives at `deploy/nginx-router.ifmphk.com.conf`. Where a task below says Caddy, edit that nginx config instead — the `/control` route already ships in it, so Task A3's Caddy step is a no-op.
>
> **Fleet change (2026-09-21):** the 4-model fleet in the examples below was expanded to 8 live-probed models with real prices (`kimi-k2.7-code` and `deepseek-v4.1-flash` are the corrected upstream ids; text-only models are now `deepseek-v4.1-flash` AND `qwen3.8-flash`). Authoritative source: `PROFILES` in `routing_policy.py` + `litellm-config.yaml`.

**Goal:** Add silent cross-session learning (persisted verdict tables) and a control surface (control-service in compose, local MCP server, control-file-driven policy reads) to the llm-router, with a hard fail-open guarantee: with learning off, corrupt, or absent, routing is byte-identical to today.

**Architecture:** A `control-service` container owns `control.json` (leases, weight overrides, learning toggles, reset intents) on a named volume the policy mounts read-only; the policy reads it via an atomic-write-aware mtime+size cache, never over HTTP. The learner inside the policy process persists verdict-derived tables to `learned.json` (single writer: the proxy), guarded by evidence gates and read-time influence clamps. A local stdio MCP server proxies read/write endpoints through Caddy to the control-service. A golden-replay test locks the fail-open guarantee.

**Tech Stack:** Python 3.11 stdlib (json, os, re, time, hashlib, threading), FastAPI or stdlib http.server for the control-service (stdlib chosen below — zero new deps in the proxy image path), pytest, Docker Compose, Caddy, MCP stdio server (Python `mcp` SDK, local machine only).

## Global Constraints

These apply to every task; each task's requirements implicitly include this section.

- **C1 Fail-open is absolute:** any learner or control-file error inside the request path is caught, counted, and swallowed; the candidate list is never left empty; no exception escapes `run()`. With `POLICY_LEARN=0`, corrupt/absent `learned.json`, or learning disabled in `control.json`, decisions must equal today's for any deterministic request sequence (the mode matrix in the review, finding B1).
- **C2 Influence bounds, enforced at read time:** phrase kind-bonus ≤ +0.10; `kind_bar_bias` ∈ ±0.15; combined trust (in-RAM + learned) clamped to ±0.40 at the point of use (review M8).
- **C3 Evidence gate:** a learned row applies only when its decayed evidence count ≥ 8. Decay: `effective = count * 0.5 ** (age_days / 90)` using `last_ts` (review §3, open item 1).
- **C4 Single-writer files:** the proxy process is the only writer of `learned.json` and `routing.jsonl`; the control-service is the only writer of `control.json`. Enforced by crossed read-only compose mounts (review B4).
- **C5 Raw prompt text never written to disk by the router; distillations mechanically sanitized before persistence** (regex key/token scrub + length caps: key point ≤ 400 chars, phrase ≤ 60). Queue max age 10 min, then drop (review M4).
- **C6 No new files in the litellm request path:** `control.json` is read by stat+read from a mounted volume; no HTTP calls from the policy (spec Part 2).
- **C7 Single proxy process assumed** (module-level `STATE`, learner queue, single-writer files). Document; assert at plugin init when detectable (review m6).
- **C8 Precedence of switches:** `POLICY_DISABLE=1` > `POLICY_LEARN=0` > `control.json learning.enabled=false` > learning on. Lease precedence: `[[use:...]]` directive > active lease > session stickiness > fresh decision (review B5).
- **C9 No git repo. Do not run git commands.** Test gates replace commits.
- **C10 New policy code must not assume `sys.modules` registration** — no dataclasses with stringized annotations, NamedTuple/plain classes only (README "Two failure modes"; existing tests cover the loader path).
- **C11 Every write in the system is atomic:** tempfile in the same directory + `os.replace` (control-service for `control.json`; learner for `learned.json`) (review M5).
- **C12 Test-pyramid rule:** unit tests run without network, without LLM calls, without Docker (except `tests/smoke_proxy.ps1`, unchanged in that respect).

---

## Phase A — Control surface (in flight; this plan's tasks A1–A8 define the target state)

**Depends on:** nothing. **Note:** another worker is implementing this phase in this directory right now, and its file names may differ from the ones below (at time of writing it has created `control_store.py`, `control_service.py`, `mcp_server.py` and modified `routing_policy.py`, `docker-compose.yml`, `Caddyfile`). The **interface contracts** in this plan (function names, schemas, env-var names, the crossed-mount single-writer rule, the fail-open semantics) are authoritative; the file names are not — if the worker's artifacts already provide a contract, keep its name and reconcile this plan's references. Do not duplicate or delete in-progress files; verify each task against what exists first.

### File structure for this phase

```
llm-router/
  control_service.py        # NEW  - the compose service: HTTP endpoints, atomic control.json writes
  mcp_server.py             # NEW  - local stdio MCP server (runs on the Windows box, not in compose)
  control_file.py           # NEW  - shared read/validate/cache logic for control.json (imported by BOTH
                            #      routing_policy.py and control_service.py; no litellm imports)
  routing_policy.py         # MOD  - read control.json (mtime+size cache), lease + weight override application
  docker-compose.yml        # MOD  - add control-service, control-data volume, crossed mounts
  Caddyfile                 # MOD  - add /<token>/control/* route before the v1 catch-all
  .env.example              # MOD  - document CONTROL_TOKEN / control precedence knobs
  .cursor/rules/router.md   # NEW  - the always-on rule (spec Part 3)
  tests/test_control.py     # NEW  - control-file parse, cache, lease expiry, torn write, precedence
  tests/test_control_service.py  # NEW - endpoint behaviour against a temp dir (no Docker needed)
  docs/plans/…              # (this file; no other doc edits in phase A)
```

### Task A1: `control_file.py` — schema, validation, cached reader

**Files:**
- Create: `control_file.py`
- Test: `tests/test_control.py`

**Interfaces (produces — all later tasks use these names):**
- `DEFAULT_CONTROL: dict` — `{"learning_enabled": True, "weights": {}, "lease": None, "reset": None, "reset_id": None, "consumed_reset_id": None}`
- `load_control(path: str) -> tuple[dict | None, str]` — returns `(parsed_dict, error_reason)`; `(None, reason)` on any read/parse/validation failure. Pure: no caching.
- `class ControlCache:` — `__init__(self, path: str)`, `get(self) -> dict` (never raises; returns last-good snapshot, or `DEFAULT_CONTROL` only when the file is absent and no good snapshot exists), `refresh(self) -> None`, `invalidate_for_tests(self) -> None`, attribute `last_error: str | None`.
- `validate_control(data: Any) -> tuple[dict | None, str]` — shape check: `learning_enabled: bool`, `weights: dict[str, float]` with keys in `{POLICY_W_COST, POLICY_W_TRUST, POLICY_W_HEADROOM, POLICY_W_LATENCY}` and values in `[-2.0, 2.0]`, `lease: None | {"model": str, "until": float, "max_requests": int | None}` with `model` in the known fleet, `until` in the past..+900 s, `max_requests` in `1..200` or None, `reset: None | "phrases"|"trust"|"bars"|"all"`, `reset_id: str | None` (uuid hex), `consumed_reset_id: str | None`.
- `atomic_write_json(path: str, data: dict) -> None` — tempfile in same dir + `os.replace`, fsync.

**Steps:**

- [ ] **Step 1: Write the failing tests.** In `tests/test_control.py` (test doubles use `tmp_path` fixtures; no real filesystem-timing assumptions — force cache state via `invalidate_for_tests`, never by waiting):

```python
import json, os, time, pytest, control_file

FLEET = ("openai/glm-5.3-flash", "openai/glm-5.3", "openai/kimi-2.7-code", "openai/deepseek-4.1-flash")

def write(path, data):
    control_file.atomic_write_json(str(path), data)

def test_valid_full_document_parses(tmp_path):
    p = tmp_path / "control.json"
    write(p, {"learning_enabled": False, "weights": {"POLICY_W_COST": 0.5},
              "lease": {"model": "openai/glm-5.3", "until": time.time() + 60, "max_requests": 5},
              "reset": None, "reset_id": None, "consumed_reset_id": None})
    data, err = control_file.load_control(str(p))
    assert data is not None and err == ""
    assert data["learning_enabled"] is False

def test_absent_file_returns_none_and_default_via_cache(tmp_path):
    cache = control_file.ControlCache(str(tmp_path / "none.json"))
    got = cache.get()
    assert got == control_file.DEFAULT_CONTROL

def test_corrupt_file_keeps_last_good(tmp_path):
    p = tmp_path / "control.json"
    write(p, {"learning_enabled": False})
    cache = control_file.ControlCache(str(p))
    assert cache.get()["learning_enabled"] is False
    p.write_text("{ truncated", encoding="utf-8")   # simulate torn write
    cache.refresh()
    assert cache.get()["learning_enabled"] is False   # last good, NOT defaults (review M5)
    assert cache.last_error

def test_corrupt_then_repaired_recovers_on_next_refresh(tmp_path):
    p = tmp_path / "control.json"
    write(p, {"learning_enabled": False})
    cache = control_file.ControlCache(str(p))
    p.write_text("{ truncated", encoding="utf-8")
    cache.refresh()
    write(p, {"learning_enabled": True})
    cache.refresh()
    assert cache.get()["learning_enabled"] is True

def test_size_change_without_mtime_change_invalidates(tmp_path):
    # (mtime_ns, size) pair compared by inequality, not mtime > last (review M5)
    p = tmp_path / "control.json"
    write(p, {"learning_enabled": False})
    cache = control_file.ControlCache(str(p))
    st = p.stat()
    with open(p, "w", encoding="utf-8") as h:  # rewrite with same coarse mtime, new size
        json.dump({"learning_enabled": True, "weights": {"POLICY_W_COST": 0.1}}, h)
    os.utime(p, (st.st_atime, st.st_mtime))
    cache.refresh()
    assert cache.get()["learning_enabled"] is True

def test_validation_rejects_unknown_weight_and_bad_lease(tmp_path):
    _, err = control_file.validate_control({"weights": {"POLICY_W_NOPE": 0.5}})
    assert err
    _, err = control_file.validate_control({"lease": {"model": "gpt-9", "until": time.time() + 60}})
    assert err
    _, err = control_file.validate_control({"lease": {"model": FLEET[0], "until": time.time() + 10_000}})
    assert err  # beyond 900 s clamp (review B5)

def test_lease_expired_when_until_in_past(tmp_path):
    p = tmp_path / "control.json"
    write(p, {"lease": {"model": FLEET[0], "until": time.time() - 1, "max_requests": None}})
    data, err = control_file.load_control(str(p))
    # load_control round-trips the raw lease; expiry evaluation lives in the
    # policy's lease application (Task A4). Here we only assert validation
    # accepts an already-expired lease (it must — the file can be stale) and
    # the field survives:
    assert data is not None and err == ""
    assert data["lease"]["model"] == FLEET[0]
    assert data["lease"]["until"] < time.time()
```

- [ ] **Step 2: Run them, see them fail.**

```
python -m pytest tests/test_control.py -v
```
Expected: collection error / failures — `control_file` does not exist.

- [ ] **Step 3: Implement `control_file.py`** (stdlib only; `FLEET_MODELS` imported lazily from `routing_policy.PROFILES` via a parameter — keep this module import-safe with no litellm imports so the MCP server and control-service can both use it):

```python
"""Shared control-file schema, validation, atomic writes, cached reads.

Imported by routing_policy.py (in the proxy), control_service.py (in compose)
and mcp_server.py (local). Must stay stdlib-only and litellm-free.
"""
from __future__ import annotations
import json, os, tempfile, time
from typing import Any

WEIGHT_KEYS = {"POLICY_W_COST", "POLICY_W_TRUST", "POLICY_W_HEADROOM", "POLICY_W_LATENCY"}
RESET_KINDS = {"phrases", "trust", "bars", "all"}
LEASE_MAX_SECONDS = 900.0
LEASE_MAX_REQUESTS = 200

DEFAULT_CONTROL: dict = {
    "learning_enabled": True, "weights": {}, "lease": None,
    "reset": None, "reset_id": None, "consumed_reset_id": None,
}

def validate_control(data: Any) -> tuple[dict | None, str]:
    if not isinstance(data, dict):
        return None, "not an object"
    out = dict(DEFAULT_CONTROL)
    if "learning_enabled" in data:
        if not isinstance(data["learning_enabled"], bool):
            return None, "learning_enabled not bool"
        out["learning_enabled"] = data["learning_enabled"]
    if "weights" in data:
        w = data["weights"]
        if not isinstance(w, dict):
            return None, "weights not object"
        for k, v in w.items():
            if k not in WEIGHT_KEYS:
                return None, f"unknown weight {k}"
            if not isinstance(v, (int, float)) or not -2.0 <= float(v) <= 2.0:
                return None, f"weight {k} out of range"
        out["weights"] = dict(w)
    if "lease" in data and data["lease"] is not None:
        lease = data["lease"]
        if not isinstance(lease, dict):
            return None, "lease not object"
        if "model" not in lease or "until" not in lease:
            return None, "lease missing model/until"
        if lease["until"] - time.time() > LEASE_MAX_SECONDS:
            return None, "lease beyond max duration"
        mr = lease.get("max_requests")
        if mr is not None and (not isinstance(mr, int) or not 1 <= mr <= LEASE_MAX_REQUESTS):
            return None, "lease max_requests out of range"
        out["lease"] = {"model": lease["model"], "until": float(lease["until"]),
                        "max_requests": mr}
    if "reset" in data and data["reset"] is not None:
        if data["reset"] not in RESET_KINDS:
            return None, "reset kind unknown"
        out["reset"] = data["reset"]
    for key in ("reset_id", "consumed_reset_id"):
        if key in data and data[key] is not None:
            if not isinstance(data[key], str) or len(data[key]) > 64:
                return None, f"{key} malformed"
            out[key] = data[key]
    return out, ""

def load_control(path: str) -> tuple[dict | None, str]:
    try:
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
    except FileNotFoundError:
        return None, "absent"
    except Exception as error:
        return None, f"read/parse: {type(error).__name__}"
    return validate_control(raw)

def atomic_write_json(path: str, data: dict) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".control-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

class ControlCache:
    """(st_mtime_ns, st_size)-inequality cache; last-good on failure (review M5)."""
    def __init__(self, path: str) -> None:
        self.path = path
        self._stamp: tuple[int, int] | None = None
        self._good: dict = dict(DEFAULT_CONTROL)
        self.last_error: str | None = None

    def get(self) -> dict:
        try:
            stat = os.stat(self.path)
        except FileNotFoundError:
            self.last_error = "absent"
            return self._good if self._stamp is not None else dict(DEFAULT_CONTROL)
        except OSError as error:
            self.last_error = type(error).__name__
            return self._good
        stamp = (stat.st_mtime_ns, stat.st_size)
        if stamp != self._stamp:
            self.refresh()
        return self._good

    def refresh(self) -> None:
        data, err = load_control(self.path)
        if data is None:
            self.last_error = err          # keep last good; do NOT update _stamp
            return
        try:
            stat = os.stat(self.path)
            self._stamp = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            pass
        self._good = data
        self.last_error = None

    def invalidate_for_tests(self) -> None:
        self._stamp = None
```

Note the deliberate detail: a failed parse never updates `_stamp`, so the next `get()` re-attempts the read — a torn write heals on the next request (review M5).

- [ ] **Step 4: Run the tests, see them pass.**

```
python -m pytest tests/test_control.py -v
```
Expected: all PASS. If `test_size_change_without_mtime_change_invalidates` is flaky on this machine's mtime granularity, it has surfaced the real reason production must use a named volume (review m7) — keep the test, use `os.utime` explicitly as shown.

**Gate:** `python -m pytest tests/test_control.py -v` green.

### Task A2: `control_service.py` — HTTP endpoints owning `control.json`

**Files:**
- Create: `control_service.py`
- Test: `tests/test_control_service.py`

**Interfaces:**
- Consumes: `control_file.load_control`, `atomic_write_json`, `validate_control`, `DEFAULT_CONTROL` (Task A1).
- Produces: HTTP service on `CONTROL_PORT` (default 4100), endpoints (all require `Authorization: Bearer $CONTROL_TOKEN`):
  - `GET /health` → `{"service": "ok", "control": <control dict>, "proxy_up": bool|null, "last_decision_age_s": float|null, "learner": {"queue_depth": int|null, "errors": int|null}}` (learner fields via last decision-log record, review M6; `proxy_up` via probe of `http://litellm:4000/health/liveliness`, outside the request path).
  - `GET /decisions?n=50` → last `n` parsed records of `routing.jsonl` (shared volume), newest first.
  - `GET /learned` → parsed `learned.json` (read-only, proxy-owned file), plus error field if unparseable.
  - `POST /lease` → body `{"model": str, "seconds": float} | {"model": str, "requests": int}`; writes `lease = {model, until: now+min(seconds,900), max_requests: min(requests,200) or None}`; one active lease at a time (a new POST replaces it, review B5 clamps).
  - `POST /weights` → merge-override `weights`.
  - `POST /learning` → `{"enabled": bool}` or `{"reset": "phrases"|"trust"|"bars"|"all"}`; reset writes an *intent* (`reset` + fresh `reset_id`) that the policy consumes (review B4).
- Writes: only `control.json` under its data dir (C4, C11).

**Steps:**

- [ ] **Step 1: Failing tests** (`tests/test_control_service.py`) — drive the service in-process with `http.client` against a temp data dir; each test boots it on port 0. Cover: auth rejects missing/wrong bearer; valid control round-trip via `GET /health`; `POST /lease` clamps seconds > 900 and requests > 200; `POST /learning {"reset": "trust"}` sets `reset` and a non-empty `reset_id`; `GET /decisions` reads a fabricated `routing.jsonl`; `GET /learned` returns `{"error": ...}` (not 500) when `learned.json` is corrupt.

```python
import json, time, threading, http.client, pytest
import control_service

class Harness:
    def __init__(self, tmp_path, monkeypatch):
        self.data = tmp_path
        monkeypatch.setattr(control_service, "CONTROL_PATH", str(tmp_path / "control.json"))
        monkeypatch.setattr(control_service, "ROUTING_LOG", str(tmp_path / "routing.jsonl"))
        monkeypatch.setattr(control_service, "LEARNED_PATH", str(tmp_path / "learned.json"))
        monkeypatch.setattr(control_service, "PROBE_URL", None)  # skip proxy probe in tests
        self.port, self.srv = control_service.start(0)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
    def req(self, method, path, body=None, token="t0ps3cret"):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        c.request(method, path, json.dumps(body) if body is not None else None, headers)
        r = c.getresponse(); payload = r.read().decode()
        c.close()
        return r.status, (json.loads(payload) if payload else {})

@pytest.fixture
def svc(tmp_path, monkeypatch):
    return Harness(tmp_path, monkeypatch)

def test_health_without_auth_401(svc):
    assert svc.req("GET", "/health", token=None)[0] == 401

def test_lease_roundtrip_and_clamp(svc):
    status, _ = svc.req("POST", "/lease", {"model": "openai/glm-5.3", "seconds": 10_000})
    assert status == 200
    _, body = svc.req("GET", "/health")
    assert body["control"]["lease"]["until"] <= time.time() + 900 + 5

def test_reset_writes_intent_not_direct_file_change(svc):
    status, body = svc.req("POST", "/learning", {"reset": "trust"})
    assert status == 200 and body["reset_id"]

def test_learned_corrupt_is_error_not_500(svc, tmp_path):
    (tmp_path / "learned.json").write_text("{oops", encoding="utf-8")
    status, body = svc.req("GET", "/learned")
    assert status == 200 and "error" in body

def test_decisions_reads_log(svc, tmp_path):
    with open(tmp_path / "routing.jsonl", "a", encoding="utf-8") as h:
        h.write(json.dumps({"ts": time.time(), "chosen": "openai/glm-5.3", "kind": "debug"}) + "\n")
    status, body = svc.req("GET", "/decisions?n=10")
    assert status == 200 and body["decisions"][0]["chosen"] == "openai/glm-5.3"
```

- [ ] **Step 2:** Run: `python -m pytest tests/test_control_service.py -v` — expect import failure.
- [ ] **Step 3: Implement** with `http.server.ThreadingHTTPServer` + a small router function. Requirements baked in: every handler wraps in try/except returning `{"error": ...}` with 200/4xx, never a traceback; reads under `threading.Lock`; `CONTROL_TOKEN` from env, compared with `hmac.compare_digest`; fleet-model validation for leases delegates to `control_file.validate_control` (pass the fleet list in from env or config). Ship it as a plain module runnable both in-container (`python control_service.py`) and in tests (`start(port)`).
- [ ] **Step 4:** Run tests; all green.

**Gate:** `python -m pytest tests/test_control_service.py tests/test_control.py -v` green.

### Task A3: Compose + Caddy wiring

**Files:**
- Modify: `docker-compose.yml`, `Caddyfile`, `.env.example`

**Steps (no tests here; the gate is A8's smoke run against compose-config):**

- [ ] **Step 1:** Add to `docker-compose.yml` (crossed mounts make C4 structural, review B4):

```yaml
  control:
    image: python:3.12-slim
    container_name: llm-router-control
    restart: unless-stopped
    command: ["python", "/app/control_service.py"]
    working_dir: /app
    volumes:
      - ./control_service.py:/app/control_service.py:ro
      - ./control_file.py:/app/control_file.py:ro
      - ./routing_policy.py:/app/routing_policy.py:ro   # fleet list for lease validation
      - control-data:/control                        # rw: the ONLY writer of control.json
      - router-logs:/logs:ro                          # ro: reads routing.jsonl only
    env_file: [.env]
    environment:
      CONTROL_PORT: "4100"
      CONTROL_PATH: /control/control.json
      ROUTING_LOG: /logs/routing.jsonl
      LEARNED_PATH: /logs/learned.json              # learned.json lives on router-logs, see below
    depends_on: [litellm]
    ports:
      - "127.0.0.1:4100:4100"

volumes:
  router-logs:
  control-data:
```

  And extend the litellm service mounts (the learner writes `learned.json`; the policy reads `control.json` read-only):

```yaml
      - control-data:/control:ro      # policy READS control.json; never writes it (C4)
      - router-logs:/app/logs         # existing; learner's learned.json lives here, NOT in control-data
```

  Resolve the `learned.json` location once, in one place: **learner state file is `/app/logs/learned.json`** (router-logs volume, written by the proxy, read-only-mounted into control at `/logs/learned.json`; set `LEARNED_PATH=/logs/learned.json` on control). `control-data` holds only `control.json`. This makes single-writer physically enforced by mount modes.

- [ ] **Step 2:** `Caddyfile` — add the control route *before* the v1 catch-all (longer prefix wins, review m3):

```
	handle_path /CHANGEME_TOKEN/control/* {
		reverse_proxy 127.0.0.1:4100
	}
```

- [ ] **Step 3:** `.env.example` — add `CONTROL_TOKEN=<openssl rand -hex 24, distinct from LITELLM_MASTER_KEY>` with a comment: separate token so MCP access cannot be replayed against `/v1` for completions (review m3). Also add commented `# POLICY_LEARN=0` with the precedence note (C8).

**Gate:** `docker compose config` parses (needs Docker; if unavailable on the dev box, `python -c "import yaml,sys; yaml.safe_load(open('docker-compose.yml'))"` at minimum).

### Task A4: Policy reads `control.json` — weights, lease, learning flag

**Files:**
- Modify: `routing_policy.py`
- Test: `tests/test_control.py` (add a policy-integration section), `tests/test_routing_policy.py` (golden)

**Interfaces:**
- Consumes: `control_file.ControlCache` (Task A1).
- Produces (for later tasks and the log): decision reason `"pinned-lease"`; signal/log field `"lease": true` when the lease decided; weights override application; `CONTROL_PATH` env (default `/control/control.json`, test: tmp path via env); module-level `_CONTROL_CACHE: ControlCache | None` (lazy — must not stat at import time, or unit tests that never set the env var would touch `/control`).

**Steps:**

- [ ] **Step 1: Failing tests** in `tests/test_control.py`:

```python
import asyncio, time, pytest, control_file, routing_policy as policy
# helpers `Context`/`ask` are defined in tests/test_routing_policy.py; import
# or inline them here (simplest: a tiny tests/_helpers.py both files import)

def test_env_control_path_none_means_disabled(monkeypatch):
    monkeypatch.setattr(policy, "_CONTROL_CACHE", None)
    ctx = Context(ask("fix the failing test in report_parser.py"))
    asyncio.run(policy.CursorAutoPolicy().run(ctx))
    assert ctx.candidate_models == ["openai/kimi-2.7-code"]

def test_lease_pins_and_logs_reason(tmp_path, monkeypatch):
    p = tmp_path / "control.json"
    control_file.atomic_write_json(str(p), {"lease": {"model": "openai/glm-5.3",
                                                      "until": time.time() + 120, "max_requests": None}})
    monkeypatch.setenv("CONTROL_PATH", str(p))
    policy._CONTROL_CACHE = None
    ctx = Context(ask("fix the failing test in report_parser.py"))
    asyncio.run(policy.CursorAutoPolicy().run(ctx))
    assert ctx.candidate_models == ["openai/glm-5.3"]
    assert ctx.signals["policy"]["reason"] == "pinned-lease"

def test_expired_lease_does_not_pin(tmp_path, monkeypatch):
    p = tmp_path / "control.json"
    control_file.atomic_write_json(str(p), {"lease": {"model": "openai/glm-5.3",
                                                      "until": time.time() - 5, "max_requests": None}})
    monkeypatch.setenv("CONTROL_PATH", str(p))
    policy._CONTROL_CACHE = None
    ctx = Context(ask("fix the failing test in report_parser.py"))
    asyncio.run(policy.CursorAutoPolicy().run(ctx))
    assert ctx.candidate_models == ["openai/kimi-2.7-code"]          # fresh decision
    assert ctx.signals["policy"]["reason"] != "pinned-lease"

def test_use_directive_beats_lease(tmp_path, monkeypatch):
    p = tmp_path / "control.json"
    control_file.atomic_write_json(str(p), {"lease": {"model": "openai/glm-5.3",
                                                      "until": time.time() + 120, "max_requests": None}})
    monkeypatch.setenv("CONTROL_PATH", str(p))
    policy._CONTROL_CACHE = None
    ctx = Context(ask("quick one [[use:glm-5.3-flash]]"))
    asyncio.run(policy.CursorAutoPolicy().run(ctx))
    assert ctx.candidate_models == ["openai/glm-5.3-flash"]          # C8 precedence
    assert ctx.signals["policy"]["reason"] == "pinned-directive"
```

  Plus one lease-expiry-orphan test: after a lease expires, a session that was pinned *during* the lease must re-decide fresh (drop session entries whose model came from the lease — review B5). Drive two requests: first under the active lease, then with the lease expired; assert the second is not `pinned-cache`.

- [ ] **Step 2:** Run; expect failures (`pinned-lease` unknown, control not read).
- [ ] **Step 3: Implement** in `_run`, after the group guard, before stickiness:
  - lazy `_get_control() -> dict` — builds `ControlCache(os.environ.get("CONTROL_PATH", ""))` once; empty path → `DEFAULT_CONTROL` with no stat.
  - weights: `w = {**env_defaults, **control["weights"]}` computed per decision into a dict passed to `decide()` (change `decide`'s signature to accept `weights: Mapping[str, float]`; keep module-level `W_COST` etc. as the defaults so existing tests still pass).
  - lease: active iff `until > time.time()` and model in `known` and `max_requests` budget not exhausted (in-memory counter, best-effort, documented overshoot, review B5). If active: winner = that model, reason `pinned-lease`, `gate_reason "ok"`, record `lease: true` in signals/log; bypass session-stickiness. If it *just* expired: delete session entries pointing at the leased model (orphan cleanup).
  - `learning_enabled` consumed by phase 1 (store on the control dict now; no behavioural effect yet).
  - all inside the existing try/except; control errors count into a module-level error counter, never raise (C1).
- [ ] **Step 4:** Run `python -m pytest tests/test_control.py tests/test_routing_policy.py -v` — all green, including the untouched 19 policy tests (no control file present → identical behaviour; that is the fail-open default).

**Gate:** `tests/test_control.py` + `tests/test_routing_policy.py` green with `CONTROL_PATH` unset (proves no-op when absent).

### Task A5: Behavior golden — lock today's routing

**Files:**
- Create: `tests/golden_routing.json`
- Test: `tests/test_golden.py`

**Steps:**

- [ ] **Step 1:** Create `tests/test_golden.py` that replays a fixed corpus of ~25 conversations (the spec's canonical asks plus: image turn, oversized prompt, stall loop, ESCALATE, `[[cheap]]`, `[[use:...]]`, reminder-only turn, multi-turn stickiness sequence, second-session interleaving, escape-hatch group) through `CursorAutoPolicy` with `POLICY_LOG=""`, fresh `STATE` per case, and asserts the full decision tuple (chosen, reason, gate, kind, bar rounded) matches `tests/golden_routing.json`.
- [ ] **Step 2:** Generate: `python tests/test_golden.py --update` writes the golden from current behaviour (pre-learning, post-A4). Commit-free by design (C9) — the golden is a checked-in artifact regenerated only deliberately.
- [ ] **Step 3:** Run `python -m pytest tests/test_golden.py -v` — green.

**Why this exists:** the golden is the *operationalization* of the fail-open promise (review B1). Phase 1 tasks must run the golden in all three degraded modes: control file absent, control `learning_enabled=false`, `POLICY_LEARN=0` — expecting identical output each time.

**Gate:** golden test green; `tests/test_routing_policy.py` still green.

### Task A6: MCP server (`mcp_server.py`, local machine)

**Files:**
- Create: `mcp_server.py`
- Test: `tests/test_mcp_server.py`

**Interfaces:**
- Consumes: the control-service endpoints over HTTPS through Caddy (base URL + `CONTROL_TOKEN` from env or a small config block); `control_file`-level validation for early client-side checks (model names etc.).
- Produces: MCP tools `health`, `decisions(n)`, `learned`, `lease(model, seconds|requests)`, `weights(**overrides)`, `learning(enabled|reset)`. Stdio transport. Runs on the Windows box; never in compose (spec Part 2; review q3 keeps it local).

**Steps:**

- [ ] **Step 1: Failing tests** with the MCP SDK's in-memory client (skip gracefully if `mcp` not importable, like `test_resolves_through_litellms_own_resolver` does): server object exposes the six tools; `lease` round-trips against a stubbed HTTP layer (monkeypatch the transport function to return canned responses); tool errors surface as MCP errors, not crashes.
- [ ] **Step 2:** Run; expect failure.
- [ ] **Step 3: Implement** using the `mcp` Python SDK, stdio server, one `httpx`-style client function `_call(method, path, body)` (stdlib `urllib.request` is fine — zero deps) pointed at `https://router.ifmphk.com/<token>/control/...` from env `CONTROL_BASE_URL`. Every tool wraps `_call` in try/except → returns `{"error": ...}` text.
- [ ] **Step 4:** Tests green.

**Gate:** `python -m pytest tests/test_mcp_server.py -v` green (or SKIP-all when SDK absent — then a manual `echo`-driven stdio smoke in the task notes must be performed once on the VPS-deployed stack).

### Task A7: The Cursor rule

**Files:**
- Create: `.cursor/rules/router.md` (content from the spec verbatim, the three lines: `model: inherit`, copy `[[escalate]]` on ESCALATE turns, never image work to `deepseek-4.1-flash`)
- Modify: `~/.cursor/skills/router-hygiene/SKILL.md` — only if phase A added vocabulary the skill must reflect (lease + control tools); keep the edit to a short "Control plane" section (don't touch this file otherwise).

**Steps:** write the rule file; add the control-plane paragraph to the skill; done. No test (Cursor-side artifact). Cross-check the image line still matches the escape-hatch groups in `litellm-config.yaml` (it does today — claims 3/4 in the review).

**Gate:** manual: rule file exists, is 10 lines or fewer, and its model names match the config.

### Task A8: Smoke test extension + phase A done

**Files:**
- Modify: `tests/smoke_proxy.ps1`

**Steps:**

- [ ] **Step 1:** Extend the smoke script: after the existing debug-ask assertion, set `POLICY_LOG` as today, boot the proxy with `CONTROL_PATH` pointing at a local temp `control.json` (the runner already controls env), write a lease via `control_file.atomic_write_json` from a small inline `python -c` block, send one more completion request, assert the decision log's second record shows `pinned-lease`. Then corrupt `control.json` (torn write: raw string, not atomic), send a third request, assert the decision still logs and is NOT `pinned-lease` (last-good + fresh decision), and stderr shows no `policy_error`.
- [ ] **Step 2:** Run `.\tests\smoke_proxy.ps1` locally — green.
- [ ] **Step 3:** Full phase A gate:

```
python -m pytest tests/test_routing_policy.py tests/test_control.py tests/test_control_service.py tests/test_mcp_server.py tests/test_golden.py -v
python tests\validate_config.py
.\tests\smoke_proxy.ps1
```

All green = phase A done.

---

## Phase 1 — Learning bookkeeping (no LLM calls, no prompt text)

**Depends on:** phase A (control flag + reset intent plumbing; golden test from A5).

### File structure for this phase

```
learning.py            # NEW  - verdict detection, evidence tables, decay, gates, persistence (litellm-free)
routing_policy.py      # MOD  - learner hooks: enqueue on decision, verdict on next decision, reset consumption
tests/test_learning.py # NEW
tests/test_golden.py   # MOD  - run the corpus in the three degraded modes
```

### Task 1.1: Verdict detection (pure functions)

**Files:**
- Create: `learning.py`
- Test: `tests/test_learning.py`

**Interfaces (produces):**
- `tokenize(text: str) -> set[str]` — lowercased `[a-z0-9_./-]{3,}` (exact, review B2).
- `is_reask(prev_ask: str, next_ask: str) -> bool` — strict containment: ≥ 0.9 of prev's tokens present in next AND `len(next_tokens - prev_tokens) <= max(1, len(prev_tokens) // 20)` (review B2).
- `classify_verdict(prev, current) -> str | None` — returns one of `"escalate" | "reask" | "clean" | None given two consecutive decision records (`{"ask": str, "directives": list, "stall": int, "kind": str, "model": str}`); escalate = `ESCALATE`/`[[escalate]]` in current's newest-ask with word-boundary match `\bESCALATE\b` (review m8) or `escalate` directive; reask = `is_reask` on the two asks; clean = current turn count ≥ 4 without either; else None.
- `verdict_weight(verdict: str) -> float` — escalate −0.20, reask −0.20, stall −0.15, clean +0.03, direct-switch −0.20 (handled separately in 1.4), 429/5xx 0.0.

**Steps:**

- [ ] **Step 1: Failing tests** — the decisive ones:

```python
import learning

def test_extension_is_not_reask():
    prev = "fix the failing test in report_parser.py"
    nxt  = "fix the failing test in report_parser.py and the date parser too"
    assert learning.is_reask(prev, nxt) is False     # review B2: extension, not complaint

def test_verbatim_resend_is_reask():
    prev = "fix the failing test in report_parser.py"
    assert learning.is_reask(prev, prev + " please") is True

def test_substring_escalate_word_not_verdict():
    prev = {"ask": "fix the bug", "directives": [], "stall": 0, "kind": "debug", "model": "m"}
    cur  = {"ask": "why did the test ESCALATED into a failure", "directives": [], "stall": 0, "kind": "debug", "model": "m"}
    assert learning.classify_verdict(prev, cur) is None   # review m8
```

  Plus: reask caps (per-session reask contribution cap of 2 — test that a third reask in a session returns None); clean requires `turns >= 4`.
- [ ] **Step 2:** Run; fail. **Step 3:** Implement pure functions, no I/O. **Step 4:** Green.

**Gate:** `python -m pytest tests/test_learning.py -v` green.

### Task 1.2: Evidence tables, decay, gates, clamped reads

**Files:**
- Modify: `learning.py`
- Test: `tests/test_learning.py`

**Interfaces (produces):**
- `class EvidenceTable:` — `add(key: tuple, delta: float)`, `effective_count(key) -> float` (decay `count * 0.5 ** (age_days/90)`, C3), `value(key) -> float | None` (None below gate 8), `to_dict()`, `from_dict(data) -> EvidenceTable` (malformed input → empty table), `reset(kinds: set[str])`.
- `class LearnedState:` — `phrases: EvidenceTable`, `bars: EvidenceTable` (per kind), `trust: EvidenceTable` (per (kind, model)); `apply_verdict(...)`; `to_dict()/from_dict()`; `sanitize(text) -> str` (regex scrub: `sk-[A-Za-z0-9]{16,}`, `ghp_\w+`, `AKIA[0-9A-Z]{16}`, `xox[baprs]-\w+`, `-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----`, plus hard truncation 400/60 chars, review M4/m5).
- Read-time influence application (used by `routing_policy`): `kind_bonus(ask: str) -> float` (≤ +0.10, near-tie only, review M3), `bar_adjust(kind) -> float` (±0.15), `learned_trust(kind, model) -> float` (0.0 below gate).

**Steps:**

- [ ] **Step 1: Failing tests:** evidence gate (7 observations → no value; 8 → value applies); influence bounds (a phrase repeated 100 times still moves kind bonus ≤ 0.10; 100 escalates → bar adjust within ±0.15; trust within ±0.40 *including* the combined-clamp path with in-RAM trust at +0.30 and learned at +0.20 → clamped 0.40, review M8); decay (90-day-old evidence halves effective count, falls below gate); `from_dict` on garbage → empty table; sanitize strips each secret shape and truncates.
- [ ] **Step 2:** Run; fail. **Step 3:** Implement. Note `from_dict` must never raise (C1) and must ignore unknown keys/version-skewed shapes. **Step 4:** Green.

**Gate:** `python -m pytest tests/test_learning.py -v` green.

### Task 1.3: Persistence + fail-open golden in degraded modes

**Files:**
- Modify: `learning.py` (load/save via `atomic_write_json`), `routing_policy.py` (learner wiring: `POLICY_LEARN` env → module flag; control `learning_enabled`; queue of recent decision summaries in RAM; verdict application on next decision; reset-intent consumption writing `consumed_reset_id` back through the learned-file meta)
- Test: `tests/test_learning.py`, `tests/test_golden.py`

**Interfaces:**
- `LEARNED_PATH` env (default `/app/logs/learned.json`); `load_learned(path) -> LearnedState` (absent/corrupt → empty state, never raises, review B1 matrix last row); `save_learned(path, state)`.
- Policy integration contract: learner hooks fire only when learning enabled (C8 precedence: `POLICY_LEARN=0` wins over control); every hook call wrapped, counted, swallowed (C1).

**Steps:**

- [ ] **Step 1: Failing tests:**
  - corrupt `learned.json` → `load_learned` returns empty state, no raise;
  - end-to-end fail-open: run the golden corpus (A5) in three modes — (a) `POLICY_LEARN=0`, (b) control `learning_enabled=false`, (c) `learned.json` corrupt on disk with learning on — asserting byte-identical decisions to the golden in all three (this is the spec's Verification table's fail-open row, made executable);
  - reset intent: write `control.json` with `{"reset": "trust", "reset_id": "abc"}` → next decision consumes it (trust table empty, `consumed_reset_id == "abc"` in the saved learned file), and a *second* decision with the same intent does not re-apply (idempotence);
  - learner exception (monkeypatch `apply_verdict` to raise) → request still routed, error counter incremented.
- [ ] **Step 2:** Run; fail. **Step 3:** Implement. Keep the learner queue as decision *summaries* (ask text kept in RAM only, bounded deque of 40, max age 10 min — C5; summaries dropped, not flushed, on age). **Step 4:** Green, and the full suite green.

**Gate:** `python -m pytest tests/ -v` green; golden identical across all degraded modes.

### Task 1.4: Direct-model-switch leak (skip-path observer)

**Files:**
- Modify: `routing_policy.py` (skip path), `learning.py` (attribution)
- Test: `tests/test_learning.py`

**Steps:**

- [ ] **Step 1: Failing tests** (review B3's rules): switch in an existing session, to a *different* model than the session was served by, with topical continuity → −0.20 to the previous model for that kind; switch to the *same* model the session already used → no verdict; brand-new session (no prior cursor-auto session key) → no verdict; no topical continuity (token overlap < 0.3) → no verdict.
- [ ] **Step 2:** Run; fail. **Step 3:** Implement on the `skipped-not-fleet-group` path: look up `STATE.sessions` by session key; apply `learning.classify_switch(...)`. **Step 4:** Green.

**Gate:** `python -m pytest tests/ -v` green; golden unchanged (the skip path never narrows candidates).

### Task 1.5: `/learned` endpoint data + phase 1 gate

**Steps:** `control_service.py` already reads `learned.json` (A2/A3 paths). Verify `GET /learned` shows the three tables + recent key points placeholder (empty until phase 2). Run the full gate:

```
python -m pytest tests/ -v
python tests\validate_config.py
.\tests\smoke_proxy.ps1
```

**Phase 1 done when:** all green, and a manual run with learning on over ~20 scripted requests produces a `learned.json` whose tables stay within bounds (spot-check with `python -c "import json;print(json.load(open('logs/learned.json')))"`).

---

## Phase 2 — LLM extraction (`phrase_kind` + key points)

**Depends on:** phase 1 (evidence tables, gates, persistence, sanitize pipeline).

### Task 2.1: Extraction client (direct-to-upstream, never via the proxy)

**Files:**
- Create: `extractor.py`
- Test: `tests/test_extractor.py`

**Interfaces:**
- `async extract(asks: list[str]) -> list[dict]` — calls `ROUTER_KEY_DEFAULT_BASE` `/chat/completions` with model `glm-5.3-flash` **directly** (env vars already in the container; review M2), prompt requests JSON: `{items: [{key_points: [≤3, ≤400 chars], phrases: [≤3, ≤60 chars], kind: one of BASE_BAR keys}]}`. Client-side validation: drop non-conforming items entirely (review m5); `kind` must be in `BASE_BAR` keys (not `KIND_KEYWORDS` — review claim 14). Timeout 30 s, one retry, then the batch is dropped (safe: losing a batch loses only learning, C1). Never interpolates payloads into exceptions (review M4).

**Steps:**

- [ ] **Step 1: Failing tests** with a stubbed HTTP layer (monkeypatch the transport): valid batch parses; malformed LLM output → all items dropped, no raise; secret in a returned key point → `sanitize` strips it before it can reach `EvidenceTable.add` (feed the raw output through the persistence path and assert the stored artifact is clean — C5); timeout → batch dropped, error counted.
- [ ] **Step 2:** Run; fail. **Step 3:** Implement (stdlib `urllib` in a thread, or litellm's own client — either, but no proxy-group traversal). **Step 4:** Green.

**Gate:** `python -m pytest tests/test_extractor.py -v` green.

### Task 2.2: Learner queue + batch pipeline

**Files:**
- Modify: `learning.py`, `routing_policy.py`
- Test: `tests/test_learning.py`

**Steps:**

- [ ] **Step 1: Failing tests:** queue enqueues the newest human ask per decision (and nothing else — no tool output, no system prompt, no assistant text, review M4); batch triggers at 20; queue max age 10 min drops stale items; raw text is absent from every persisted artifact after a flush (scan `learned.json` + `observations.jsonl` for a planted canary string — the C5 test); crash-path test: exception mid-batch leaves queue drained (or aged out), nothing persisted raw, counter incremented.
- [ ] **Step 2:** Run; fail. **Step 3:** Implement: `observations.jsonl` append-only via atomic-per-line writes, rotated with the `LOG_MAX_BYTES` pattern (review M4); extraction results flow into `phrases`/`bars` tables through `sanitize` + validation; queue held only in RAM. **Step 4:** Green.

**Gate:** `python -m pytest tests/ -v` green, including the canary test.

### Task 2.3: `phrase_kind` influence on classification (near-tie bounded bonus)

**Files:**
- Modify: `routing_policy.py` (`_detect_kind` or its call site)
- Test: `tests/test_learning.py`, `tests/test_golden.py`

**Steps:**

- [ ] **Step 1: Failing tests:** learned phrase present in ask and base classification is a near-tie (margin ≤ 0.10) → phrase's kind wins, bonus ≤ 0.10; margin > 0.10 → base winner stands even with 100×-repeated phrase (review M3); no learned phrases (learning off / below gate) → classification identical to today (golden mode run).
- [ ] **Step 2:** Run; fail. **Step 3:** Implement as a bounded score bonus inside the kind-selection step, reading `LearnedState` only when learning enabled. **Step 4:** Green, golden unchanged in all degraded modes (phrases absent → zero bonus).

**Gate:** full suite green; golden identical in degraded modes.

### Task 2.4: Phase 2 gate + live tuning

- [ ] Run full local gate (`python -m pytest tests/ -v`, `validate_config`, smoke).
- [ ] Deploy to VPS; run ~1 day of real traffic; then `GET /learned` (via MCP `learned` tool or curl) and check: phrase count ≤ 200 (row cap, review §3), no phrase longer than 60 chars, no key point longer than 400, every `kind` in `BASE_BAR` keys. Extraction cadence ≈ 1 flash call per 20 turns (check spend or a counter in `/health` learner fields if added per review M6).

**Phase 2 done when:** the above checks pass on live data.

---

## Phase 3 — Reliability signal (429/5xx) — REDUCED SCOPE

**Depends on:** phase 1. **Decision per review M1/q4 (recommendation adopted): log-and-expose only, no routing effect.** If instead the full version is wanted, a separate spec addendum is required first (post-call hook mechanism, a fourth table `model_reliability` with its own bound, utility interaction) — do not build it inside this plan.

### Task 3.1: Failure logging hook

**Files:**
- Create: `failure_hook.py` — a litellm `CustomLogger`-style callback module registered via `litellm_settings.callbacks` (this is a *different* plugin mechanism than `router_settings.plugins`; verify the exact registration key against the pinned litellm version in a spike before writing tests)
- Modify: `litellm-config.yaml` (`litellm_settings.callbacks`), `docker-compose.yml` (mount)
- Test: `tests/test_failure_hook.py`

**Steps:**

- [ ] **Spike first:** confirm the callback signature for post-call status capture on the deployed litellm tag (`async def async_post_call_failure_hook` / `log_success_event` equivalents — check `litellm.integrations.custom_logger`). One hour box; if the mechanism does not expose status codes cleanly, stop and report — the phase is optional by design.
- [ ] **Step 1: Failing tests:** 429/5xx appends `{"ts", "model", "status"}` to `routing.jsonl` (or a sibling `reliability.jsonl` on the same volume — pick one, name it in the implementation); success does not; hook never raises (same C1 discipline; litellm wraps callbacks, but assert it anyway); learning-off → hook inert.
- [ ] **Step 2:** Run; fail. **Step 3:** Implement. **Step 4:** Green.
- [ ] **Step 4:** Expose in `/health`: last failure age and count per model, read from the log by the control-service. No routing change. Document in README that reliability remains neutral to quality verdicts (spec's verdict table stands).

**Gate:** `python -m pytest tests/ -v` green; manual: kill an upstream (point one base URL at `example.invalid` in a local run) and see the failure recorded and surfaced, with routing unaffected.

---

## Task → spec-section coverage map

| Spec section | Task(s) |
|---|---|
| Part 2: control-service, read/write endpoints | A2, A3 |
| Part 2: mtime-cached control read, no HTTP in policy | A1, A4 |
| Part 2: lease, not pin (+ review B5 fixes) | A1 (validation/clamps), A2, A4 (precedence, orphan cleanup) |
| Part 2: weights endpoint, live retune | A2, A4 |
| Part 3: rule | A7 |
| Part 3: MCP server | A6 |
| Verification table: `test_control.py` rows | A1, A4 (torn write, expiry, invalid-file = last-good, per review M5) |
| Part 1 phases table, phase 1 row | 1.1–1.5 |
| Guardrails: evidence gate, bounds, fail-open, POLICY_LEARN=0, delete-to-reset | 1.2, 1.3 (+ A5 golden in all modes) |
| Constraint table: verdicts incl. reask fix, direct-switch attribution, 429 neutral | 1.1, 1.4, 3.1 |
| Retention: RAM-only raw text, sanitized distillations, newest-message-only | 2.1, 2.2 |
| Part 1 phase 2 row: extraction, phrase_kind, key points | 2.1–2.3 |
| Open item 1: 90-day half-life (evidence decay) | 1.2 (C3) |
| Open item 2: lease in `GET /health` | A2 (lease block in `/health`) — adopted: yes |
| Part 1 phase 3 row: 429/5xx | 3.1 (reduced scope per review q4) |
| To-verify item 1: MCP connectivity through Caddy | A3 (route), A6 (local stdio kept, review q3) |

## Self-review notes

- Spec coverage: every spec section maps to at least one task (table above). The two spec items deliberately *not* built as specified: phase 3's "optional learned classifier" (cut, review q1) and re-ask Jaccard ≥ 0.45 (replaced with strict containment, review B2) — both are review findings with rationale, not omissions.
- Placeholder scan: no TBDs; every code step carries real code or an exact interface contract; the one explicit spike (3.1) is boxed with a stop condition.
- Type consistency: `ControlCache.get() -> dict`, `EvidenceTable`/`LearnedState` names, `pinned-lease` reason string, `LEARNED_PATH`/`CONTROL_PATH` env names are used identically across tasks; `decide()` gains a `weights` parameter in A4 and every later reference assumes it.
- Risks acknowledged in-plan: litellm callback API variance (spike in 3.1), MCP SDK availability (graceful skip mirrors the existing litellm-optional test pattern), phase A worker overlap (per-task reconcile instruction).
