# Learning + Control Surface — design

> **Note (2026-09-21, post-deploy):** the VPS front door is **nginx**, not
> Caddy — the VPS already ran nginx on 80/443. Read "Caddy" in this doc as
> "the nginx server block in `deploy/nginx-router.ifmphk.com.conf`"; the
> contracts (path token, `/control` route, auth layering) are unchanged.
>
> **Note (2026-09-21, fleet):** the 4-model fleet referenced in the examples
> is now 8 live-probed models with real prices; text-only models are
> `deepseek-v4.1-flash` AND `qwen3.8-flash`. Authoritative source:
> `PROFILES` in `routing_policy.py`.

Date: 2026-09-21
Status: phase A implemented (control plane + MCP + rule); phase B (learner
daemon + LLM extraction) not yet built - the learned.json schema and the
policy's read path for it are in place and test-covered, so phase B is purely
additive.
Scope: `llm-router` (LiteLLM proxy + policy plugin) and the Cursor-side artifacts

### Review resolution (2026-09-21)

Findings B1-B5, M5 and M7 from the adversarial review
(`docs/specs/2026-09-21-learning-and-control-review.md`) are resolved by the
implementation plan (`docs/plans/2026-09-21-learning-and-control-plan.md`) and
landed in phase A. One line each; details live in the review doc, not here:

- **B1** (fail-open/mode matrix): locked by `tests/test_golden.py` — baseline
  replay vs. no-control-file / learning-disabled / `POLICY_LEARN=0` must be
  byte-identical; env kill-switch wins over control.json.
- **B2** (re-ask semantics) and **B3** (direct-switch attribution): learner-side;
  the plan's strict-containment and skip-path-observer contracts bind phase B.
- **B4** (two writers): crossed read-only compose mounts — the proxy can never
  write `control.json`, the control service can never write `learned.json`.
- **B5** (leases): seconds-primary stateless deadline, no counters; precedence
  `[[use:...]]` > lease > stickiness; lease-expiry orphan sweep; bounds clamped
  at the control service.
- **M5** (control.json reads): atomic tmp+fsync+rename writes, `(mtime_ns, size)`
  inequality cache, failed parse never caches; corrupt falls to the last-good
  snapshot (persisted sidecar), never to defaults; absent file reads as defaults.
- **M7** (stall detector): a stall now requires a consecutive trailing run
  (k>=3 identical tool calls, nothing different between them), so healthy
  test→edit→test TDD loops produce no verdict; golden regenerated post-fix.
- With the deploy items: **m2/m3** (switch precedence documented; the control
  service authenticates every request itself via `CONTROL_TOKEN`, Caddy route
  in DEPLOY.md Step 8b).

Phase B (learning) resolutions, landed 2026-09-21:

- **B2 in code** (re-ask): strict containment implemented in
  `learning.is_reask` (>= 0.9 of the previous ask's tokens, at most
  max(1, len(prev)//20) new tokens; tokenization lowercased
  `[a-z0-9_./-]{3,}`); per-session re-ask contribution capped at 2;
  `kind_bar_bias` consumes only escalate/direct-switch verdicts (q2 adopted).
- **B3 in code** (skip-path observer): `routing_policy._observe_skip_path`
  attributes -0.20 only when a recent cursor-auto session exists, the
  switched-to model differs, and token overlap >= 0.3; routing behaviour on
  the skip path is unchanged.
- **B4 in code** (single writer): the learner (`learning.py`, inside the
  proxy) is the sole writer of learned.json; resets are intents in
  control.json consumed apply-once via `reset_id` + `last_applied_reset_id`
  with no write-back; crossed read-only mounts unchanged.
- **M2 in code**: `extractor.py` calls `ROUTER_KEY_DEFAULT_BASE`
  chat-completions directly; no proxy traversal.
- **M4 in code**: mechanical `sanitize()` (credential regexes + 400/60-char
  caps) on every persisted string; RAM-only ask queue (40 items, 10 min
  age); observations.jsonl rotated; payloads never interpolated into
  exceptions; canary-tested including forced-crash paths.
- **M8 in code**: the combined runtime+learned clamp sits in
  `PolicyState.effective_trust_of` (phase A); learned values are gated
  (8 observations, 90-day half-life decay) before the clamp sees them, and
  the ControlStore re-applies gate + clamp at read time.
- **q4 adopted**: phase 3 is log-and-expose only
  (`failure_hook.py` -> `reliability.jsonl` -> `/health`), no routing
  effect; the optional learned classifier stays cut (q1).

## Goal

Two capabilities the current setup lacks:

1. **Learning** — the router adapts across sessions to which model actually works
   for a given kind of work, and to how this user phrases requests. Silent: it
   changes decisions, it does not produce a report.
2. **Control + visibility** — an agent or the operator can see why a model was
   chosen and can steer the router (pause learning, retune, lease a model) without
   editing files or restarting containers.

## Non-goals

- No prompt-quality feedback to the user. Explicitly declined: learning is silent.
- No terminal UI. The litellm admin UI and the JSONL log remain the human view.
- No persistent model pins. See the lease decision below.

## Constraint that shapes everything

The router never learns whether an answer was good. It sees decisions and the
*next* request, nothing in between. Every learnable signal is an indirect leak:

| Leak on the following turn | Verdict | Weight |
|---|---|---|
| user writes `ESCALATE` | bad, strong | -0.20 |
| user re-asks / rephrases (token Jaccard >= 0.45) | bad, strong | -0.20 |
| user switches to a direct model name | bad, strong | -0.20 |
| stalled loop (same tool call 3+) | bad, medium | -0.15 |
| upstream 429 / 5xx | neutral (reliability, not quality) | 0 |
| next ask unrelated, no complaint | good, weak | +0.03 |

This ceiling is the design's main limitation and is stated rather than worked
around. It is still enough to learn "this user escalates design work" and "this
phrase means debug, not factual".

## Part 1 — Learning

### What is learned

Three tables in `learned.json`, all bounded:

| Table | Source | Effect |
|---|---|---|
| `phrase_kind` | LLM-extracted phrases from asks, tagged with true task kind | biases classification: the user's idioms map to kinds |
| `kind_bar_bias` | escalate/re-ask rate per task kind | raises the capability bar for kinds this user finds under-served |
| `model_trust` | outcome leaks per (kind, model) | the existing in-memory table, now persisted |

### Retention

Raw prompt text is **never written to disk**. The learner:

1. takes the newest human message from the in-memory queue,
2. sends it to `glm-5.3-flash` for extraction: key points, notable phrases,
   true task kind, instructed to strip credentials, tokens and secrets,
3. writes only the distillation to `observations.jsonl`,
4. drops the raw text when the batch completes.

Only the newest human message is ever extracted. Never tool output, never file
contents, never the system prompt — that is where secrets live.

### Guardrails

An unsupervised loop that rewrites its own routing weights is how this kind of
system rots. Bounds, all enforced at read time:

- **Minimum evidence: 8 observations** before any learned rule applies at all.
- **Bounded influence.** A learned phrase moves the bar at most +/-0.10; a
  `kind_bar_bias` at most +/-0.15; `model_trust` stays within +/-0.40. Learning
  nudges `PROFILES`; it can never replace the hand-tuned values.
- **Fail-open.** Unreadable or malformed `learned.json` is ignored and the
  hand-tuned defaults stay live. A learner exception is counted and swallowed;
  the request path never sees it.
- **One switch, one reset.** `POLICY_LEARN=0` disables; deleting `learned.json`
  resets to pristine.

### Phases

Phase 1 needs no LLM call and no prompt text — it is pure bookkeeping over the
decision log and outcome leaks. Phase 2 adds the extraction call.

| Phase | Delivers | New dependency |
|---|---|---|
| 1 | persistence, outcome verdicts, `kind_bar_bias`, persisted trust | none |
| 2 | LLM extraction: `phrase_kind` (the user's idioms) + key points | ~1 flash call per 20 turns |
| 3 | model reliability from 429/5xx; optional learned classifier | none |

## Part 2 — Control surface

### Why a separate service, not endpoints in litellm

The MCP server runs locally (stdio, on the Windows machine). The policy runs
inside the container on the VPS. They cannot share memory, so every read or
write is a network call. Rather than extend litellm's server, a small service in
the same compose owns a control file that the policy reads — so the policy gains
no HTTP dependency and still fails open.

```
Cursor agent --stdio--> MCP server (local PC)
                            | HTTPS via Caddy, path token + master key
                            v
                    control-service (in compose)
                            | reads and writes a shared volume
                            v
        routing.jsonl  learned.json  control.json
                            ^ mtime-cached read, no HTTP in the policy
                    routing_policy.py (in the proxy)
```

### Read endpoints

| Endpoint | Returns |
|---|---|
| `GET /health` | proxy up, learner queue depth, learning enabled or not, last decision age |
| `GET /decisions?n=50` | recent decisions: chosen, kind, reason, gate, candidates, scores |
| `GET /learned` | `phrase_kind`, `kind_bar_bias`, `model_trust`, recent key points |

### Write endpoints

| Endpoint | Effect |
|---|---|
| `POST /lease` | bounded pin: `{model, requests?}` or `{model, seconds?}` |
| `POST /weights` | live retune of `POLICY_W_*`, no restart |
| `POST /learning` | `{enabled: bool}` or `{reset: "phrases"|"trust"|"bars"|"all"}` |

### Lease, not pin

`POST /lease` exists because a persistent pin has no defensible session identity:
an MCP call carries no session id, so "pin this for the task" would have to mean
"the most recent caller" — racy against concurrent chats and subagents. A lease
is bounded by request count or seconds, so stale state is impossible and the
blast radius is a known number of requests.

Per-request routing hints stay **in-band** (`[[use:...]]`, `[[escalate]]`,
`[[cheap]]`), because those already work, are visible in the transcript, and are
scoped to the task by construction. MCP writes are the **control plane** only.

## Part 3 — Cursor-side artifacts

| Layer | Artifact | Status |
|---|---|---|
| Wiring | Cursor Settings -> Models: base URL + master key + `cursor-auto` | manual, required |
| Skill | `~/.cursor/skills/router-hygiene/SKILL.md` | exists |
| Rule | always-on invariants, 10 lines | to add |
| MCP | local stdio server exposing the endpoints above | to build |

### The rule

The one invariant with no other mechanism: the policy reads directives from the
newest human message, so a user's `ESCALATE` does not cross the delegation
boundary. Subagent requests route on the prompt the parent wrote.

```markdown
---
description: Router invariants for the cursor-auto LiteLLM proxy
alwaysApply: true
---

# Model routing invariants

- Subagents: use `model: inherit`. Never pin `composer-2.5-fast` — it bypasses the router.
- When the user writes ESCALATE, copy `[[escalate]]` into every subagent prompt this turn.
  Directives are read from the newest human message, so they do not cross delegation on their own.
- Never send image work to `deepseek-v4.1-flash` or `qwen3.8-flash` (text-only).
```

Deliberately excluded: the bulk-work tagging convention, because the capability
bar and cost term already route bulk work to the cheapest capable model
(currently `qwen3-vl-flash`). An always-on
line that duplicates the policy costs tokens for no behaviour change.

The image line is the one apparent duplicate that stays, because it guards a path
the policy does not cover: the direct model names (`deepseek-v4.1-flash` and
friends) are escape-hatch groups that skip the policy entirely, so an agent
pinning one by name can still hand an image to a text-only model and get a 400.
The rule covers the bypass; the policy covers everything else.

### Rule cost

Rules sit in the system prefix and are excluded from classification (the policy
reads only `role == "user"` messages), so they are paid by the model that already
won, after routing. One exception: `_estimate_tokens` counts all messages, so
rule text feeds the `scale` input — a 100-token rule moving a 12,000-token
threshold is nil.

100-token rule, ~50 requests per session, prompt caching on:

| Winning model | Cost per session |
|---|---|
| `qwen3-vl-flash` $0.016/1M | $0.000016 |
| `glm-5.3-flash` $0.10/1M | $0.0001 |
| `kimi-k2.7-code` $0.77/1M | $0.00077 |
| `glm-5.3` $1.13/1M | $0.00113 |

Under $0.01 per session worst case. One avoided bad delegation (50k in + 10k out
on `glm-5.3`) is about $0.05 — roughly 100x the monthly rule cost.

## Verification

| Check | Proves |
|---|---|
| `tests/test_routing_policy.py` | unchanged behaviour with learning disabled |
| `tests/test_learning.py` | verdict mapping, evidence gate, influence bounds, malformed `learned.json` fails open |
| `tests/test_control.py` | control-file parse, mtime cache, lease expiry, invalid control file is ignored |
| `tests/validate_config.py` | config, profiles, plugin still agree |
| `tests/smoke_proxy.ps1` | real proxy boots, routes, logs a decision |

The rule that matters: with `learned.json` corrupted, deleted, or absent, routing
must be byte-identical to today's behaviour. That is the test for fail-open.

## To verify during implementation

1. Whether Cursor's MCP client can reach the control-service through Caddy, or
   whether the MCP server needs to run on the VPS instead. Determines the MCP
   config shape; the endpoint surface is unchanged either way.
2. Whether provider prompt caching engages on the system prefix as assumed. The
   ROI conclusion holds even uncached, so this is a tuning detail.
3. `glm-5.3-flash` extraction quality at the batch size chosen (start at 20
   asks per call, measure, adjust).

## Open items

- Extraction batch size and cadence: start 20 asks, one call, then tune.
- Whether `phrase_kind` needs a decay so last month's idioms do not pin this
  month's routing. Leaning toward a 90-day half-life.
- Whether the lease should be visible in `GET /health` (leaning yes, so an agent
  can see a lease it did not set).
