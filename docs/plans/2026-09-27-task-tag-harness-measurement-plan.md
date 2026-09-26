# Task-Tag Harness Measurement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a sticky `[[task:<slug>]]` directive that tags decisions and usage receipts, with per-(task × harness) rollups in `/usage`, the MCP tool, and the dashboard — measure-only, routing untouched.

**Architecture:** The tag parses where existing directives parse (`routing_policy.build_signature`), persists per session in `STATE.sessions`, rides the proven `context.signals["policy"]` bridge into `usage_hook`, and lands in receipts via `usage_state.record_receipt(task=...)`. Rollups come from the receipt lines in `control_service.usage_summary` (not the TTL'd aggregate) so calls and HKD share one window.

**Tech Stack:** Python 3 stdlib (existing discipline), pytest, vanilla JS dashboard, no new dependencies.

**Spec:** `docs/specs/2026-09-27-task-tag-harness-measurement-design.md`

## Global Constraints

- Regex (authoritative): `\[\[task:([a-z0-9][a-z0-9-]{0,38})\]\]` — lowercase, ≤ 39 chars, starts alphanumerically; `[[task:]]` (empty) clears; malformed ignored.
- Measure-only: golden replay for untagged traffic stays byte-identical; the tag must never alter model selection, alarms, gates, or stickiness.
- Fail-open: absent `usage_state` module, `POLICY_USAGE_RECORD=0`, and `task=None` callers all preserve today's behavior exactly.
- Directives are read from the newest human message only; a tag in a *system/assistant* message must be ignored (same scan discipline as `DIRECTIVE_RE`).
- Test isolation: every new test file sets `POLICY_STATE_DIR` / `POLICY_USAGE_*` to its own temp dir BEFORE importing `routing_policy` / `usage_state` (see `tests/test_usage_accounting.py` lines 22–32).
- Suite must pass with `python -m pytest tests/ -v` (currently 180 passed; this plan adds more).
- Never pin a subagent model; use `model: inherit` if delegating.

---

### Task 1: Directive parsing + session stickiness in routing_policy

**Files:**
- Modify: `routing_policy.py` (directive regex near line 564, `Signature` near line 630, `build_signature` near line 932, session record near line 1840, decision log near line 1880)
- Test: `tests/test_routing_policy.py` (append)

**Interfaces:**
- Consumes: existing `build_signature(messages) -> Signature | None`, `STATE.sessions[session_key]` dict, `context.signals["policy"]` dict construction.
- Produces: `Signature.task_tag: str | None`; `TASK_RE = re.compile(r"\[\[task:([a-z0-9][a-z0-9-]{0,38})\]\]")`; `TASK_CLEAR_RE = re.compile(r"\[\[task:\s*\]\]")`; module-level `def _detect_task_tag(ask: str, previous: Mapping | None) -> str | None`; `"task"` key in both `context.signals["policy"]` and the `_log()` record.

- [ ] **Step 1: Write failing tests** (append to `tests/test_routing_policy.py`; reuse the file's existing `Context`, `ask`, `run`, and `reset` helpers — read their definitions at the top of the file before writing):

```python
def test_task_tag_parsed_and_logged():
    reset()
    ctx = run(Context(ask("add the kit generator [[task:module-kit]]")))
    decision = ctx.signals["policy"]
    assert decision["task"] == "module-kit", decision


def test_task_tag_sticky_until_new_tag():
    reset()
    run(Context(ask("add the kit generator [[task:module-kit]]")))
    ctx2 = run(Context(ask("continue the kit work")))          # no tag: sticky
    assert ctx2.signals["policy"]["task"] == "module-kit"
    ctx3 = run(Context(ask("now the crud work [[task:crud-endpoint]]")))
    assert ctx3.signals["policy"]["task"] == "crud-endpoint"   # newest wins


def test_task_tag_clear_directive():
    reset()
    run(Context(ask("do the kit [[task:module-kit]]")))
    ctx2 = run(Context(ask("unrelated chat now [[task:]]")))
    assert ctx2.signals["policy"]["task"] is None


def test_task_tag_malformed_ignored():
    reset()
    ctx = run(Context(ask("do the kit [[task:Bad_Slug]] [[task:" + "x" * 45 + "]]")))
    assert ctx.signals["policy"]["task"] is None


def test_task_tag_in_system_message_ignored():
    reset()
    msgs = [
        {"role": "system", "content": "plan: [[task:module-kit]]"},
        {"role": "user", "content": "seed"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "continue"},
    ]
    ctx = run(Context(msgs))
    assert ctx.signals["policy"]["task"] is None
```

- [ ] **Step 2: Run tests, verify FAIL** — `python -m pytest tests/test_routing_policy.py -k task_tag -v`. Expected: FAIL (`KeyError: 'task'` or assertion).

- [ ] **Step 3: Implement.** In `routing_policy.py`:

  a. Next to `DIRECTIVE_RE` (line ~564) add:
  ```python
  # Task-tag directive ([[task:<slug>]], 2026-09-27 spec): a per-session
  # measurement label that joins Cursor plans to Aider executions. It never
  # affects selection - receipts and rollups only. The regex is authoritative
  # over the spec prose (a trailing hyphen is tolerated and valid).
  TASK_RE = re.compile(r"\[\[task:([a-z0-9][a-z0-9-]{0,38})\]\]")
  TASK_CLEAR_RE = re.compile(r"\[\[task:\s*\]\]")
  ```

  b. Add `"task_tag"` to `Signature.__slots__`, default `None` in `__init__`.

  c. In `build_signature`, after the directive loop (line ~945):
  ```python
  # NOTE: task tag is resolved in the request path (needs `previous`),
  # here we only expose the newest-message parse.
  ```
  and add `task_tag` handling: in the main `run()` after `previous = STATE.sessions.get(session_key) ...` (line ~1405) compute:
  ```python
  def _resolve_task_tag(text: str, previous: Mapping | None) -> str | None:
      """Newest tag in the newest human ask wins; else the sticky tag;
      an explicit [[task:]] clears. System/assistant text is never scanned
      (build_signature only reads the newest human message)."""
      clear = TASK_CLEAR_RE.search(text)
      match = TASK_RE.search(text)
      if match:
          return match.group(1)
      if clear:
          return None
      if isinstance(previous, Mapping):
          prev_tag = previous.get("task_tag")
          return prev_tag if isinstance(prev_tag, str) and prev_tag else None
      return None
  ```
  then `task_tag = _resolve_task_tag(signature.text, previous)` and attach via `signature.task_tag = task_tag` (set attribute after build; also pass through the exec-demotion `Signature(...)` re-creation at line ~1420 with `task_tag=signature.task_tag`).

  d. Session record (line ~1840): add `"task_tag": task_tag,` to the `STATE.sessions[session_key] = {...}` dict.

  e. Signals + log dicts (lines ~1870, ~1910): add `"task": task_tag,` to both.

- [ ] **Step 4: Run task-tag tests + full file** — `python -m pytest tests/test_routing_policy.py -v`. Expected: all PASS including the 5 new.

- [ ] **Step 5: Golden check (measure-only invariant)** — `python -m pytest tests/test_golden.py -v`. Expected: PASS unchanged (the replay compares a fixed field list without `task`; no baseline regen).

- [ ] **Step 6: Commit** — `git add routing_policy.py tests/test_routing_policy.py && git commit -m "Add [[task:]] directive: per-session task tag in decisions"`

### Task 2: Receipt plumbing — task in usage_state + usage_hook

**Files:**
- Modify: `usage_state.py` (`record_receipt` ~line 178, aggregate entry ~line 205, `UsageView` ~line 230, `_parse_entry` ~line 250)
- Modify: `usage_hook.py` (`extract_receipt_parts` return dict ~line 120, `_record` call ~line 188)
- Test: `tests/test_usage_accounting.py` (append)

**Interfaces:**
- Consumes: Task 1's `"task"` key in `signals["policy"]`.
- Produces: `usage_state.record_receipt(..., task: str | None = None) -> float` (keyword-only, default `None` — existing callers unchanged); receipt JSON gains `"task"`; aggregate entry gains `"tasks": {"<slug>\x00<client>": hkd}`; `UsageView.task_hkd: dict[tuple[str, str], float]`; `usage_hook.extract_receipt_parts(...) -> dict` gains `"task": str | None`.

- [ ] **Step 1: Write failing tests** (append to `tests/test_usage_accounting.py`; reuse its existing `FakeUsage`, `make_kwargs`, `reset`, temp-dir isolation):

```python
def test_receipt_roundtrip_with_task():
    reset()
    usage_state.record_receipt(session="s1", model="openai/glm-5.3-flash",
                               kind="code_gen", client="aider", task="module-kit",
                               prompt_tokens=1000, completion_tokens=100, cached_tokens=0)
    records = [json.loads(l) for l in open(usage_state.USAGE_LOG_PATH, encoding="utf-8")]
    assert records[-1]["task"] == "module-kit"
    view = usage_state.load_view("s1")
    assert view.task_hkd[("module-kit", "aider")] > 0.0


def test_receipt_task_none_keeps_old_shape():
    reset()
    usage_state.record_receipt(session="s2", model="openai/glm-5.3-flash",
                               kind="factual", client="cursor",
                               prompt_tokens=10, completion_tokens=5)
    view = usage_state.load_view("s2")
    assert view.task_hkd == {}


def test_hook_extract_task_from_policy_signals():
    kwargs = _kwargs(signals_policy={
        "session": "s3", "kind": "code_gen", "client": "aider",
        "chosen": "openai/glm-5.3-flash", "est_cost_hkd": 0.01, "task": "crud-endpoint"})
    parts = usage_hook.extract_receipt_parts(kwargs, FakeResponse(FakeUsage(100, 20)))
    assert parts["task"] == "crud-endpoint"


def test_hook_extract_task_absent_is_none():
    kwargs = _kwargs(signals_policy={
        "session": "s4", "kind": "factual", "client": "cursor",
        "chosen": "openai/glm-5.3-flash", "est_cost_hkd": 0.001})
    parts = usage_hook.extract_receipt_parts(kwargs, FakeResponse(FakeUsage(50, 10)))
    assert parts["task"] is None
```

(The file's real helper is `_kwargs(signals_policy=...)` with `FakeResponse(FakeUsage(...))` — see the extraction tests around line 176. `reset()` for these two hook tests is harmless but only needed for the receipt ones.)

- [ ] **Step 2: Run, verify FAIL** — `python -m pytest tests/test_usage_accounting.py -k task -v`. Expected: FAIL (`TypeError: unexpected keyword 'task'` / `KeyError: 'task'`).

- [ ] **Step 3: Implement.**

  a. `usage_state.record_receipt` — add keyword param `task: str | None = None`; add `"task": task if isinstance(task, str) and task else None,` to the receipt dict; in the aggregate update block add:
  ```python
  if task and client:
      tasks = entry.setdefault("tasks", {})
      tk = f"{task}\x00{client}"
      tasks[tk] = round(float(tasks.get(tk, 0.0)) + real, 6)
  ```

  b. `UsageView.__slots__` gains `"task_hkd"`; `__init__` sets `self.task_hkd: dict[tuple[str, str], float] = {}`; `_parse_entry` adds:
  ```python
  for tk, val in (entry.get("tasks") or {}).items():
      if isinstance(tk, str) and "\x00" in tk:
          slug, cl = tk.split("\x00", 1)
          view.task_hkd[(slug, cl)] = float(val or 0.0)
  ```

  c. `usage_hook.extract_receipt_parts` return dict gains:
  ```python
  "task": policy.get("task") if isinstance(policy.get("task"), str) else None,
  ```
  and `_record` passes `task=parts["task"],` to `record_receipt`.

- [ ] **Step 4: Run file + suite** — `python -m pytest tests/test_usage_accounting.py tests/test_routing_policy.py -v` then `python -m pytest tests/ -v`. Expected: all PASS.

- [ ] **Step 5: Commit** — `git add usage_state.py usage_hook.py tests/test_usage_accounting.py && git commit -m "Carry task tag into usage receipts and aggregate"`

### Task 3: by_task rollup + redo detector in control_service

**Files:**
- Modify: `control_service.py` (`usage_summary` ~lines 391–439; `/usage` handler unchanged — payload shape grows)
- Test: `tests/test_control.py` (append)

**Interfaces:**
- Consumes: Task 2's receipt records with `"task"` field; `tail_usage(limit)` + the module-level `USAGE_LOG_PATH`.
- Produces: `usage_summary()` returns dict with two new keys: `by_task: {slug: {client: {"calls": int, "real_hkd": float, "est_hkd": float}}}` and `task_redo: [slug, ...]` (sorted; slugs seen under ≥ 2 distinct clients in the window).

- [ ] **Step 1: Write failing test** — call `usage_summary` DIRECTLY (no HTTP server): `control_service` binds `USAGE_LOG_PATH` at import time, so reuse `test_control.py`'s `_service(state)` module-fresh-import helper with `TempState`, then write `routing_usage.jsonl` into `state.dir` BEFORE importing, mirroring `test_service_decisions_tail_skips_torn_lines`:

```python
def test_usage_by_task_and_redo():
    state = TempState()
    records = [
        {"ts": 1, "session": "s", "model": "openai/glm-5.3-flash", "kind": "code_gen",
         "client": "cursor", "task": "module-kit", "prompt_tokens": 100,
         "completion_tokens": 10, "cached_tokens": 0,
         "real_cost_hkd": 0.01, "est_cost_hkd": 0.02},
        {"ts": 2, "session": "s2", "model": "openai/glm-5.3-flash", "kind": "code_gen",
         "client": "aider", "task": "module-kit", "prompt_tokens": 100,
         "completion_tokens": 10, "cached_tokens": 0,
         "real_cost_hkd": 0.005, "est_cost_hkd": 0.02},
        {"ts": 3, "session": "s3", "model": "openai/glm-5.3-flash", "kind": "factual",
         "client": "cursor", "task": None, "prompt_tokens": 10,
         "completion_tokens": 2, "cached_tokens": 0,
         "real_cost_hkd": 0.001, "est_cost_hkd": 0.002},
    ]
    with open(os.path.join(state.dir, "routing_usage.jsonl"), "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    os.environ["CONTROL_TOKEN"] = "s3cr3t-token"
    service = _service(state)   # fresh import binds USAGE_LOG_PATH to state.dir

    def checks():
        data = _http_with_token("s3cr3t-token")("GET", "/usage")["summary"]
        assert data["by_task"]["module-kit"]["cursor"]["calls"] == 1
        assert data["by_task"]["module-kit"]["cursor"]["real_hkd"] == 0.01
        assert data["by_task"]["module-kit"]["aider"]["real_hkd"] == 0.005
        assert data["task_redo"] == ["module-kit"]

    _run_service(service, checks)
```

(The `_service` helper pops `POLICY_USAGE_LOG` from its saved-env list — if `control_service` reads `POLICY_USAGE_LOG` over the default in this environment, extend the `saved`/restore dict in `_service` with `POLICY_USAGE_LOG` as part of this task and set it to `os.path.join(state.dir, "routing_usage.jsonl")` alongside `POLICY_LOG`.)

- [ ] **Step 2: Run, verify FAIL** — `python -m pytest tests/test_control.py -k by_task -v`. Expected: FAIL (`KeyError: 'by_task'`).

- [ ] **Step 3: Implement** in `usage_summary` (after the existing `_bump` loop): change `totals` init to include `"by_task": {}` and `"task_redo": []`; inside the record loop capture `task = record.get("task") if isinstance(record.get("task"), str) else None`; after the loop:
  ```python
  # Per-(task, harness) rollup + redo detector (2026-09-27 spec): a slug
  # under >= 2 distinct clients in this window means the same task class
  # was executed by more than one harness - typically "Aider failed,
  # Cursor redid it". Flag only; no routing or alarm action.
  for record in records:
      task = record.get("task") if isinstance(record.get("task"), str) else None
      if not task:
          continue
      client = record.get("client") if isinstance(record.get("client"), str) else "unknown"
      real = float(record.get("real_cost_hkd") or 0.0)
      raw_est = record.get("est_cost_hkd")
      est = float(raw_est) if isinstance(raw_est, (int, float)) else 0.0
      row = totals["by_task"].setdefault(task, {}).setdefault(client, {"calls": 0, "real_hkd": 0.0, "est_hkd": 0.0})
      row["calls"] += 1
      row["real_hkd"] = round(row["real_hkd"] + real, 6)
      row["est_hkd"] = round(row["est_hkd"] + est, 6)
  totals["task_redo"] = sorted(
      slug for slug, clients in totals["by_task"].items() if len(clients) >= 2
  )
  ```
  (Single pass over `records` is fine; the two loops shown are for clarity against the existing code — merge if cleaner.)

- [ ] **Step 4: Run file + suite** — `python -m pytest tests/test_control.py tests/ -v`. Expected: all PASS.

- [ ] **Step 5: Commit** — `git add control_service.py tests/test_control.py && git commit -m "Add by_task rollup + task_redo detector to /usage"`

### Task 4: Dashboard + MCP surfacing

**Files:**
- Modify: `deploy/dashboard.html` (summary renderer ~lines 186–204, table head ~lines 58–74)
- Modify: `mcp_server.py` (`router_usage` tool description ~line 216)
- Test: `tests/test_mcp_e2e.py` (extend only if it asserts on tool descriptions; otherwise no test change)

**Interfaces:**
- Consumes: Task 3's `by_task` / `task_redo` summary keys; Task 1's `"task"` decision field.
- Produces: dashboard task-class section; decisions table gains a `Task` column; MCP description mentions per-task comparison.

- [ ] **Step 1: Dashboard.** In `renderSummary(summary)` after the by-model block (~line 202) add:

```javascript
    const tasks = Object.entries(summary.by_task || {});
    if (tasks.length) {
      const rows = tasks.sort((a, b) => a[0].localeCompare(b[0])).map(([slug, clients]) => {
        const cells = Object.entries(clients)
          .sort((a, b) => b[1].real_hkd - a[1].real_hkd)
          .map(([c, v]) => `${c}: ${v.real_hkd.toFixed(3)} HKD / ${v.calls} calls`).join(' · ');
        const redo = (summary.task_redo || []).includes(slug) ? ' ⚠ redo' : '';
        return `${slug}${redo} — ${cells}`;
      });
      parts.push('<b>By task:</b><br>' + rows.join('<br>'));
    }
```

  In the decisions table `<thead>` after `<th>Kind</th>` (line ~64) add `<th>Task</th>`; in the row renderer add a cell for `d.task || ''` matching the existing column order (read the renderer body and insert consistently — Client/Kind/Task ordering must match the header).

- [ ] **Step 2: MCP description.** In `mcp_server.py` replace the `router_usage` description string (~line 217) with:

```python
        "description": "Receipt-based spend: what the provider actually billed per model, "
        "client (cursor/aider/other), task kind, and TASK TAG ([[task:<slug>]] from plan "
        "docs - the cursor-vs-aider comparison), side by side with the router's pre-call "
        "estimates, plus live per-session real spend and a task_redo flag for slugs "
        "executed by more than one harness. Use to answer 'what did this really cost' "
        "(estimates assume a cold cache every turn; receipts price cached reads at the "
        "discount rate) and to check whether a kind-alarm threshold is approached.",
```

- [ ] **Step 3: Verify** — `python -m pytest tests/test_mcp_e2e.py -v` (PASS unchanged unless it snapshots descriptions; update the snapshot intentionally if so). Manually open `deploy/dashboard.html` with a token against the local control service (`python control_service.py` + a couple of fake receipts) and confirm: task rows render, redo badge shows for a two-client slug, Task column aligns with the header.

- [ ] **Step 4: Commit** — `git add deploy/dashboard.html mcp_server.py && git commit -m "Surface task-tag rollup in dashboard + MCP"`

### Task 5: Full suite + config validation + spec cross-check

**Files:**
- No source changes expected. Touches only failures found.

- [ ] **Step 1:** `python -m pytest tests/ -v` — all pass (180 + new).
- [ ] **Step 2:** `python tests/validate_config.py` — config validation still green.
- [ ] **Step 3:** Golden double-check — `python -m pytest tests/test_golden.py -v` PASS with the ORIGINAL baseline (no `--update`).
- [ ] **Step 4:** Spec cross-check — walk `docs/specs/2026-09-27-task-tag-harness-measurement-design.md` section by section (Directive / Signal flow / Rollups / Non-goals / Fail-open / Testing) and confirm each has a task above; fix gaps in place.
- [ ] **Step 5:** Commit any fixes: `git add -A && git commit -m "Task-tag measurement: suite green, spec cross-check"`
