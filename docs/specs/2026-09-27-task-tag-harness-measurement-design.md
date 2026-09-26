# Task-tag harness measurement — design

Date: 2026-09-27
Status: approved design, not yet implemented
Scope: `llm-router` only (policy plugin, usage accounting, control service, MCP, dashboard)

## Problem

Harness comparison ("is Aider cheaper-and-good-enough than Cursor for task
class X?") needs a stable per-task label that survives across harnesses. The
2026-09-26 measurement plumbing (commit `2d1bcd8`) records `client` (which
harness) and `kind` (what type of ask) on every receipt, but `kind` is too
coarse: "generate module kit from a boundary doc" and "add one CRUD endpoint
from an existing pattern" are both `code_gen`, with very different cost and
success profiles.

A tag must also cross the harness boundary itself: the plan is written in
Cursor, but execution happens in Aider, and the two sessions share nothing
but the plan text.

## Decision

A **`[[task:<slug>]]` directive**, parsed by the routing plugin next to the
existing `[[cheap]]` / `[[use:...]]` / `[[escalate]]` directives, sticky per
session, carried into usage receipts, and rolled up per (task × harness) in
the existing surfaces.

Why a directive: Cursor writes the tag into the plan doc once; every Aider
session launched from that plan inherits it inside the prompt text it is
fed — zero manual tagging on the execution side, and no control-plane
round-trip. This was chosen over control-plane tagging (tag does not ride in
the plan; per-session manual re-tag) and a task-lifecycle ledger (human
friction; front-loads machinery the measure-only phase does not need).

## Contract

### Directive

- Regex family shared with existing directives; new pattern
  `\[\[task:([a-z0-9][a-z0-9-]{0,38})\]\]` — lowercase, ≤ 39 chars, must
  start alphanumerically. The regex is authoritative; prose calls it "kebab"
  only for brevity (a trailing hyphen is tolerated by the regex and valid).
- `[[task:]]` (empty slug) **clears** the session's tag.
- Precedence: a tag in the newest human message overrides the sticky tag;
  otherwise the sticky tag persists for the session (same semantics as model
  stickiness). No tag anywhere → `task` is null.
- Malformed tags (`[[task:Bad_Slug]]`, over-length) are ignored, not errors.
- Parsing happens where existing directives are parsed, before routing; an
  unparseable tag leaves routing byte-identical.

### Signal flow (join into usage receipts)

- The tag joins `context.signals["policy"]` (the same bridge receipts
  already use for `session` / `kind` / `client` / `chosen` /
  `est_cost_hkd`), so `usage_hook.extract_receipt_parts` picks it up with no
  new transport.
- `usage_state.record_receipt` gains an optional `task: str | None`
  parameter. Receipt lines gain `"task"`. The per-session aggregate gains
  `entry["tasks"]["<slug>\x00<client>"] += real_cost_hkd` alongside the
  existing `kinds` / `clients` maps (same `\x00` pair-key convention).

### Rollups and surfaces

- `UsageView` gains `task_hkd: dict[(slug, client), float]`, parsed from the
  aggregate, mtime-cached like the existing fields.
- Control `GET /usage` gains `by_task`:
  `{slug: {client: {calls, real_hkd, est_hkd}}}` — computed by `usage_summary`
  from the same receipt lines as `by_model` / `by_client` (the aggregate is
  TTL'd and session-capped, so receipt lines are the consistent source for
  both the call counts and the HKD over the requested window).
- **Redo detector:** a slug observed under ≥ 2 distinct clients within the
  current receipt file (`routing_usage.jsonl`; the rotated `.1` file is not
  consulted) is surfaced in `/usage`'s summary as `task_redo: [slug…]`.
  This is the "Aider failed, Cursor redid it" (or "plan forgot Aider can't
  see X") flag. It is a flag only — no routing action, no alarm.
- MCP `router_usage` and the dashboard gain a task-class table
  (slug × client, HKD + calls + redo flag), reusing the receipts-summary
  rendering added in `2d1bcd8`.

### Non-goals (v1)

- No routing changes: the tag never alters model selection, alarms, or
  stickiness of models. Golden replay for untagged traffic stays
  byte-identical.
- No auto-harness-selection and no JEV/harness scorer wiring — that is the
  follow-up once weeks of receipts exist (the `by_task` rollup is the data
  contract JEV will read).
- No quality verdicts, phase bookkeeping, or task open/close lifecycle.
- No `X-Task-Class` header (YAGNI; revisit if a third harness appears).

### Fail-open discipline

- Absent `usage_state` module: today's behavior, byte-identical.
- `POLICY_USAGE_RECORD=0`: receipts off entirely; tag still logged in the
  decision log (it is routing-side data), absent from receipts.
- Malformed/missing tags: ignored; `task` is `None` end to end.
- `record_receipt(task=None)` from any caller: aggregate `tasks` map simply
  not updated; existing callers unchanged (keyword-only, default `None`).

### Workflow this enables

1. Cursor plans a task; the plan doc carries `router: [[task:module-kit]]`.
2. Aider is fed the plan; its requests carry the tag in the prompt.
3. Router decision log tags both sessions' turns; receipts bill both
   harnesses under the same slug.
4. `/usage` shows `module-kit: cursor {calls, HKD}` vs
   `aider {calls, HKD}`; a redo shows as the slug under both clients.
5. After weeks of data, the JEV follow-up reads `by_task` to learn which
   classes belong in which harness.

## Testing

- Directive: new-wins-over-sticky, sticky persistence, `[[task:]]` clear,
  malformed ignored, slug length/charset bounds (table-driven).
- Golden replay: untagged traffic byte-identical (baseline untouched).
- Receipts: roundtrip with `task`; aggregate `tasks` map accumulates per
  (slug, client); `task=None` callers unchanged.
- Rollups: `/usage by_task` grouping (calls + HKD), `task_redo` flag fires
  at ≥ 2 clients, absent when 1; MCP + dashboard include the table.
- Kill switches: `POLICY_USAGE_RECORD=0` drops `task` from receipts but not
  from the decision log.
