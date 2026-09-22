# Review — Learning + Control Surface design

Date: 2026-09-21
Spec reviewed: `docs/specs/2026-09-21-learning-and-control-design.md`
Reviewed against: `routing_policy.py`, `litellm-config.yaml`, `docker-compose.yml`, `Caddyfile`, `DEPLOY.md`, `README.md`, `.env.example`, `tests/test_routing_policy.py`, `tests/validate_config.py`, `tests/smoke_proxy.ps1`.

Context: phase A (MCP server, control-service, rule, policy control-file handling) is being implemented concurrently in this directory. Findings tagged **[phase A]** must be folded into that in-flight work rather than implemented later; everything else belongs to the learning phases. This review covers the spec and the pre-existing code only.

---

## Summary

The architecture is sound. Control-file-over-HTTP instead of endpoints inside litellm, fail-open as a hard requirement, bounded influence for every learned table, and lease-over-pin are all the right calls, and the spec is honest about its central limitation (no quality signal, only leaks). But the spec is not implementable as written: the fail-open guarantee is undefined at exactly the points where it is hard (mode matrix, evidence gate vs. byte-identity), the re-ask verdict as specified misreads normal iterative work as strong negative evidence and would persist that misreading, two of the six leaks have no capture point in the current architecture, and the control surface has a two-writer conflict on `learned.json` and lease semantics that contradict their own rationale. Five blockers, all fixable with spec edits and small design corrections; none require abandoning the approach.

---

## 1. Verification of code-level claims

Every code-behaviour claim the spec relies on, checked against `routing_policy.py`:

| # | Claim | Result | Evidence |
|---|---|---|---|
| 1 | The policy classifies only the newest `role=="user"` message | **Verified, with nuance** | `_newest_human_ask` iterates `reversed(messages)`, skips non-user roles, and returns the first user message that is not reminder-only. So classification uses the newest *substantive* human message; a user turn containing only `<system-reminder>` blocks falls through to an older one. Other signature components (scale, stall, has_image) read all messages — the spec accounts for this only for `_estimate_tokens`. |
| 2 | `_estimate_tokens` counts all messages | **Verified** | Sums `len(content)` (or extracted text) plus `str(tool_calls)` over every message, `// 4`. System prompt and rules included; the spec's "100-token rule moving a 12,000-token threshold is nil" holds. |
| 3 | The plugin skips groups with fewer than two known models | **Verified** | `known` dedupes candidates by model string, then `len(known) < 2` returns `skipped-not-fleet-group` untouched. Dedupe is by model string, so glm-5.3's two deployments count once. Direct escape-hatch groups therefore bypass the policy entirely — the basis of the rule's image line, and of finding B3. |
| 4 | Direct model names are escape hatches that skip the gates | **Verified** | Consequence of #3; a `[[use:]]`-less direct pick of `deepseek-4.1-flash` can receive an image and 400. The spec's rule line covers a real gap. |
| 5 | Verdict weights match current code | **Verified** | ESCALATE −0.20, stall −0.15, clean-turn +0.03 at `turns >= 3` (`_run` feedback block). Note two discrepancies with the spec table, in findings B2 and m10. |
| 6 | In-memory trust is clamped | **Verified** | `STATE.adjust` clamps to `TRUST_FLOOR`/`TRUST_CEILING` (±0.40, env-overridable). See M8 for why the clamp must be re-stated for the combined two-layer trust. |
| 7 | Today's decision log carries no raw text | **Verified** | `_log` writes chosen/reason/kind/scale/est_tokens/has_image/stall/directives/candidates/session only. But it also lacks `gate`, `bar` and `scores`, which `GET /decisions` promises — see m11. |
| 8 | `POLICY_W_*` are env-only today | **Verified** | Read once at import by `_env_float`. `POST /weights` needs a live-override path through `decide()`, which the spec does not mention. |
| 9 | Stickiness holds the model per task | **Verified** | Pin requires same kind, no directives, and the gates still pass. Lease interaction undefined — B5. |
| 10 | Rule cost: prices quoted match `PROFILES` | **Verified** | $0.05/$0.10/$0.55/$0.60 per 1M input all match `Profile.cost_in`. The table's arithmetic does not — see m1. Conclusion (under $0.01/session) survives the correction. |
| 11 | "Directives do not cross the delegation boundary" | **Verified** | Directives are read from the request's own newest human message; a subagent's messages contain the parent-written prompt, not the user's chat turn. The rule's `[[escalate]]`-copy line is the correct mitigation. |
| 12 | 429/5xx are invisible to the policy plugin | **Verified** | The plugin is pre-call only (it narrows `candidate_models`); no post-call hook exists anywhere in the repo. Basis of M1. |
| 13 | (Bonus) The stall detector fires on progress loops | **Verified — bug** | `_detect_stall` counts occurrences of the newest fingerprint anywhere in the last 6 calls. `[test, edit, test, edit, test]` → newest appears 3× → stall fires on a healthy TDD loop. See M7. |
| 14 | (Bonus) `"writing"` is unreachable as a kind | **Verified** | `KIND_KEYWORDS` has no `writing` key and `order` omits both `writing` and `agentic`; only the no-signal fallback can produce `agentic`. Harmless today (BASE_BAR has both keys), but `learning.py`'s kind validation should use `BASE_BAR` keys, not the keyword map. |

---

## 2. Findings

Severity legend: **blocker** = the design is wrong or its central guarantee undefined as written; **major** = will produce incorrect behaviour or cannot be built as specified without a design decision; **minor** = fix cheaply, does not block; **question** = a decision the spec should make explicitly.

### Blockers

**B1 — The fail-open guarantee is undefined where it matters: mode matrix and evidence gate contradict byte-identity.**
Section: *Guardrails*, *Verification*.
The spec requires: (a) corrupt/deleted/absent `learned.json` leaves routing "byte-identical to today", (b) "8 observations before any learned rule applies at all", (c) `POLICY_LEARN=0` disables. These conflict. Today's behaviour *includes* ungated in-RAM feedback: an ESCALATE docks −0.20 from the first occurrence, with no evidence gate. If the 8-observation gate covers the in-RAM path, behaviour differs from today and (a) is false. If it covers only persisted tables, (a) holds but the spec must say so. Worse, the new verdicts (re-ask, direct-switch) change today's semantics — a verbatim re-ask currently earns *+0.03* (the `turns >= 3` reward fires because no re-ask detection exists); the spec flips it to −0.20. Is re-ask detection part of "learning" (off when `POLICY_LEARN=0`) or part of the base policy? Each answer gives a different definition of "today's behaviour".
**Fix:** define one mode matrix and make every mode's behaviour derivable from it:

| Mode | In-RAM session feedback (escalate −0.20, stall −0.15, clean +0.03) | New verdicts (re-ask, direct-switch) | Persisted tables read | Persisted tables written |
|---|---|---|---|---|
| today (pre-change) | yes | — | — | — |
| learning on | yes | yes (learner RAM) | yes, gated ≥8, clamped | yes |
| `POLICY_LEARN=0` | yes | no | no | no |
| control `learning.enabled=false` | yes | no | no | no |
| `learned.json` corrupt/absent | yes | no | no (zero influence) | yes — next flush rewrites the file (self-heal) |

Byte-identity then means: decisions under any of the last three rows equal the first row for any scripted deterministic request sequence, from the moment of corruption until the learner heals the file. Also state explicitly: `POLICY_LEARN=0` is the hard kill-switch and wins over `control.json` `learning.enabled:true`. The golden-replay test in the plan operationalizes this.

**B2 — The re-ask verdict misreads normal iterative work as strong negative evidence.**
Section: *Constraint that shapes everything* (verdict table).
Token Jaccard ≥ 0.45 between consecutive asks fires on the most common healthy follow-up pattern. Example with the spec's own threshold: "fix the failing test in report_parser.py" → "fix the failing test in report_parser.py and the date parser too" — token sets give Jaccard ≈ 0.56, and ≈ 0.67 with common stopwords dropped (higher, not lower). That is an *extension*, not a complaint. Agentic sessions are built out of lexically-similar follow-ups, so false −0.20s accumulate exactly where turn counts are highest. Three amplifiers make it worse: (1) stickiness concentrates the penalties — the pinned model accrues re-ask verdicts a fresh choice never sees; (2) `kind_bar_bias` is fed by "escalate/re-ask rate", so a kind the user merely iterates on (debug, by nature) gets its bar raised up to +0.15 and routes everything expensive; (3) provider instability induces re-sends — after a 429/5xx the user re-submits, which is a re-ask of a request that never got an answer, and phase 1 cannot see the error (that visibility is phase 3), so instability periods generate storms of false −0.20s against innocent models. The tokenization itself is also unspecified (word level? lowercased? stopwords?).
**Fix:** replace Jaccard with a high-precision containment test: a re-ask must repeat the previous ask *without adding scope* — ≥ 0.9 of the previous ask's tokens still present and at most `max(1, len(prev)//20)` new tokens. Define tokenization exactly (lowercased `[a-z0-9_./-]{3,}`). Cap contributed re-ask verdicts per session (e.g., 2). Document precision-over-recall as the deliberate trade: a missed re-ask costs nothing (no signal fires), a false one costs persisted trust. Optionally add a softer keyword-based signal ("still", "again", "I said", "not what I asked") at −0.10 later, after the strict detector proves out. State in the spec that error-induced re-sends are a known noise window until phase 3.

**B3 — "User switches to a direct model name" has no capture point and no attribution rule.**
Section: *Constraint that shapes everything* (verdict table), *Part 2*.
The policy returns `skipped-not-fleet-group` before it ever looks at messages, sessions, or state — the skip path records nothing today. The leak is real (the request does traverse the proxy with a direct model group), but detecting it requires new behaviour the spec never describes: hooking the skip path, correlating the direct-model request with the previous `cursor-auto` session, and deciding what counts as a *complaint*. Attribution has two failure modes the spec must resolve: a switch in a brand-new chat shares no session key with the previous cursor-auto session (first-message hash differs) and must not count; a switch to the *same* model the policy was already serving is a router bypass, not a dissatisfaction, and must not dock.
**Fix:** specify the skip-path observer: on a skipped group, look up `STATE.sessions` by session key; record −0.20 only when (a) a recent cursor-auto session exists, (b) the switched-to model differs from the model that session was served by, and (c) the new ask is topically continuous with the session's previous ask (token overlap ≥ ~0.3). Otherwise ignore. Accept and document that new-chat switches are invisible.

**B4 — `learned.json` has two writers.**
Section: *Part 2* (diagram), *Write endpoints*.
The diagram shows the control-service reading *and writing* a shared volume that holds `learned.json`, and `POST /learning {reset}` implies the control-service rewrites it. But the learner inside the proxy also writes it (persisted trust). Two writers on one JSON file with no locking protocol is a corruption bug: a reset during a learner flush is a lost update either way. The learner cannot even detect an external reset because its authoritative state is in RAM.
**Fix:** single-writer by construction. The proxy (learner) is the only writer of `learned.json`; the control-service is the only writer of `control.json`. `POST /learning {reset}` writes an *intent* into `control.json` (`{reset: "trust", reset_id: <uuid>}`); the policy applies it on its next mtime poll, wipes the named table, and records `consumed_reset_id` in `learned.json`'s meta, which the control-service reads back to report applied state. Enforce with crossed read-only volume mounts (see the plan's compose diff): `router-logs` rw in litellm / ro in control; `control-data` rw in control / ro in litellm. Then no convention is needed — the mounts make any other write impossible.

**B5 — Lease semantics contradict their own rationale and leave precedence undefined.**
Section: *Lease, not pin*.
The stated reason to prefer leases over pins is "stale state is impossible". A request-count lease breaks that: someone must decrement per request. If the policy counts in its own memory, a restart re-arms the lease (stale state, exactly what was ruled out) and a second proxy process would double-count. If the countdown lives in `control.json`, the policy must write a file the control-service owns — violating B4's single-writer rule. Second hole: precedence is undefined. An active lease plus a `[[use:X]]` directive, plus an existing `pinned-cache` session pin — three pinning mechanisms, no order specified. Third: expiry orphans sessions. A 60-second lease pins sessions to model X; when it expires, those sessions keep X via `pinned-cache` for up to `SESSION_TTL` (1 hour), so the blast radius is *not* "a known number of requests" as claimed.
**Fix:** make the lease stateless-first: the authoritative expiry is `until` (epoch seconds) in `control.json`, evaluated without any in-proxy state, so restarts cannot re-arm or lose it. Keep `max_requests` as a best-effort in-memory counter with the seconds bound as the backstop, and document the small overshoot. Define precedence: in-band `[[use:...]]` wins for that request (most specific, visible in transcript); otherwise an active lease wins over session stickiness, with decision reason `pinned-lease`; on expiry, session entries recorded during the lease are dropped so those tasks re-decide fresh. Clamp lease bounds at the control-service (e.g., ≤ 900 s, ≤ 200 requests, model must be in the fleet, one active lease at a time).

### Major

**M1 — 429/5xx has no capture mechanism, and phase 3 contradicts the verdict table.**
Sections: *Constraint that shapes everything*, *Phases*.
The verdict table rates 429/5xx neutral, yet phase 3 delivers "model reliability from 429/5xx". Neutral-for-quality is the right call, but phase 3 needs (a) a capture point — the routing plugin is pre-call only (verified, claim 12); status codes require a litellm post-call hook (`litellm_settings.callbacks` module), a different plugin mechanism the spec never mentions; and (b) a home — the three learned tables have no reliability slot, so phase 3 either adds a fourth table or overloads `model_trust`, which would contradict "neutral". Also note litellm already does per-deployment cooldowns (`allowed_fails: 2`, `cooldown_time: 60`); a learned reliability layer partly duplicates it.
**Fix:** decide now: either cut phase 3's reliability to "log failures, expose in `/health`, no routing effect" (recommended — see q4), or spec the callback hook, a separate `model_reliability` table with its own bound, and its read-time interaction with the utility function.

**M2 — The extraction call must not ride the proxy, or it poisons the router's own signals.**
Section: *Retention*, *Phases*.
Unspecified: how the learner reaches `glm-5.3-flash`. If it calls the `cursor-auto` group through the proxy, the extraction request is itself routed (classified, logged as a decision, session-pinned), and its payload — a batch of user asks — becomes the newest human message for classification. If it calls the `glm-5.3-flash` escape-hatch group through the proxy, it lands in the skip path, which after B3's fix would record it as a "direct model switch" leak. Either way the router observes its own monitoring.
**Fix:** the learner calls the upstream provider directly (`ROUTER_KEY_DEFAULT_BASE` + key, model `glm-5.3-flash`) — the env vars are already in the container. No proxy traversal, no recursion, no marker needed. Trade-off to document: extraction spend is invisible to litellm's spend log (negligible: ~1 flash call per 20 turns). If proxy-traversal is ever required, mandate an internal marker checked by both the skip path and the learner queue.

**M3 — The phrase_kind influence bound is not a bound if phrases flip classification.**
Section: *Guardrails* vs *What is learned*.
The guardrail says a learned phrase moves the bar at most ±0.10, but the table says `phrase_kind` "biases classification". These are different mechanisms with different worst cases: flipping the *kind* from factual to debug changes the bar by 0.30 and swaps the whole capability column used for scoring — far beyond ±0.10. As written, the bound is unenforceable.
**Fix:** pick one mechanism and bound it mechanically at read time. Recommended: a phrase hit adds a bounded bonus (≤ 0.10) to that kind's score in `_detect_kind`, and the bonus may only decide near-ties — if the base scores' margin exceeds the bonus, the base winner stands. With keyword scores typically whole numbers, a 0.10 bonus flips only genuine ties and decides the no-signal fallback — which is the feature working ("this phrase means debug"), still within the bound.

**M4 — The raw-text-retention promise has leak paths the spec does not close.**
Section: *Retention*.
Four gaps. (1) "Instructed to strip credentials" is an instruction, not enforcement — the extraction model can quote verbatim, and "notable phrases" are quoted by design. The distillation is the disk-write path; a secret in a distillation violates the promise just as much as raw text. (2) "Never tool output, never file contents" is overstated: the newest human message regularly *contains* pasted file content and attachments (that is how users bring files into chats); the mechanism cannot exclude what the user pasted. (3) No max queue age: a batch of 20 that never completes holds raw text in RAM indefinitely. (4) Crash paths: kernel swap writes the queue to disk on a swapping VPS; an exception message that interpolates the payload (`f"bad batch: {text}"`) lands raw text in container logs; `observations.jsonl` has no rotation (the decision log does) and no retention bound.
**Fix:** mechanical post-filter on every persisted line (regex for key/token shapes — `sk-…`, `BEGIN PRIVATE KEY`, `ghp_…`, `AKIA…`, `xox…` — plus hard length caps: key point ≤ 400 chars, phrase ≤ 60); queue max age (e.g., 10 minutes, then drop — losing a batch is safe); never interpolate payloads into exception messages (count-and-swallow, per the spec's own rule); rotate `observations.jsonl` with the existing `LOG_MAX_BYTES` pattern; document swap as an accepted residual risk or set `memswap_limit` on the litellm service; reword the promise to "raw text is never written to disk by the router; distillations are mechanically sanitized before persistence."

**M5 — control.json read semantics: torn writes, mtime comparison, and what "invalid file is ignored" means.**
Section: *Part 2*, *Verification* (`test_control.py` row).
Three issues. (1) Torn writes: if the control-service writes in place, the policy can stat a new mtime and read a half-written file; the parse fails; if that failure caches the mtime as "seen", the control state sticks stale until the *next* write. The spec mandates neither atomic writes nor failed-parse retry. (2) mtime comparison must be by inequality of a `(mtime_ns, size)` pair, not `mtime > last` — an in-place rewrite with a backwards or equal coarse mtime is otherwise missed. Clock skew itself is a non-issue here: both containers share one host kernel clock; the hazards are granularity and torn reads, not skew. Note container-vs-volume: keep `control.json` on a named Docker volume (ext4/overlayfs, ns mtime) — never a bind mount, which has mtime quirks under Docker Desktop on Windows and would also leak the file onto the host. (3) "Invalid control file is ignored" is the *wrong* fail-open for the control plane, and the spec's own test row encodes it. For `learned.json`, ignore → pristine defaults is safe (pristine = today). For `control.json`, pristine defaults *re-enable* a feature the operator deliberately disabled: corrupt the file while `learning.enabled=false` and learning comes back on. Control must fail to last-known-good, not to defaults.
**Fix:** the control-service writes atomically (tempfile + `os.replace` in the same directory); the policy caches `(st_mtime_ns, st_size)` and compares by inequality; a parse failure keeps the last-good snapshot in memory and forces a re-read on the next request (failed parses never count as a successful read of that mtime); "invalid control file" means "keep last good", and only an absent file at first boot means defaults. `test_control.py` must test the torn-write-then-repair sequence, not just "invalid ignored".

**M6 — `/health` promises data the architecture cannot produce.**
Section: *Read endpoints*.
"learner queue depth" lives in the proxy process; the control-service has no channel to it (the policy reads control.json; nothing flows back). "Proxy up" needs an internal probe (fine, `http://litellm:4000/health/liveliness` from the control-service — not in the request path). "Last decision age" is derivable from `routing.jsonl` (shared volume) — fine.
**Fix:** piggyback learner telemetry on the existing outbound channel: include `queue_depth` and `learner_errors` in each decision-log record (and thus in `GET /decisions`), and have `/health` read the last record. Or drop queue depth from `/health`. Do not add a second writable file for it.

**M7 — The stall verdict inherits a detector that fires on healthy loops.**
Section: *Constraint that shapes everything* (verdict table).
Verified (claim 13): `_detect_stall` counts the newest tool fingerprint anywhere in the last six calls, so `test → edit → test → edit → test` — the canonical TDD loop, making progress between calls — reads as a stall. Today that costs a transient in-RAM −0.15 and a bar bump. Persisted, it systematically docks whichever model serves test-driven sessions and feeds `kind_bar_bias` for `debug`, the most iterative kind of all. Persisting a noisy signal is strictly worse than the current ephemeral version.
**Fix:** require a *consecutive* trailing run: the last k≥3 tool-call fingerprints must all be identical, with no different call between them. Land this as an explicit, separately-tested policy change *before* phase 1 persistence, and regenerate the behaviour golden after it — phase 1's byte-identity baseline is the post-fix behaviour.

**M8 — Trust layering can stack past the clamp, and the observation unit is undefined.**
Section: *Guardrails*, *What is learned*.
If persisted trust loads into `STATE.trust` at startup and in-RAM deltas accumulate on top, the same ±0.40 clamp applies to the sum — fine. But if the two layers clamp separately and add, effective trust reaches ±0.80, and `W_TRUST × 0.80 = 0.24` of utility swing — enough to overpower real capability gaps, i.e., learning *replacing* hand-tuned values, which the guardrails forbid. Separately, "8 observations" has no unit: per phrase? per kind? per (kind, model) cell? And if persisted trust is the gated aggregate while in-RAM trust stays ungated (today's behaviour, required by B1), the spec must say that a restart re-applies only the gated layer.
**Fix:** one combined clamp at the point of use: `trust(kind, model) = clamp(STATE.trust_of(...) + learned.trust(...), ±0.40)`, where `learned.trust` is zero until its decayed evidence count ≥ 8. Define the observation unit per table: per (kind, model) for trust, per kind for bar bias, per phrase for phrases. State that in-RAM session trust resets on restart and only the gated persisted layer survives.

### Minor

**m1 — Rule-cost table arithmetic.** Section: *Rule cost*. The per-model numbers match 10 requests × 100 tokens, not the stated 50 requests (e.g., deepseek at $0.05/1M: 1,000 tokens = $0.00005, but 5,000 tokens = $0.00025). All rows are consistently 5× low. The conclusion survives correction (worst case ≈ $0.003/session, still under $0.01) and the "roughly 100x" comparison is fuzzy either way ($0.05 avoided vs ≈ $0.003 spent ≈ 17×, or vs ≈ $0.04/month ≈ break-even). Fix the premise or the numbers; the ROI claim should rest on "negligible", which is true.

**m2 — Two off-switches, no precedence, near-identical names.** Sections: *Guardrails*, *Write endpoints*. `POLICY_DISABLE` (whole policy) vs `POLICY_LEARN` (learning) vs `control.json learning.enabled`. Define precedence once: env `POLICY_LEARN=0` is the hard kill-switch and wins over `control.json`; `POLICY_DISABLE=1` still wins over everything. Document in `.env.example` and the README tuning table.

**m3 — Control-plane exposure and the missing Caddy route.** Sections: *Part 2*, *To verify during implementation*. The spec never says how control endpoints traverse Caddy: a new, longer-prefix `handle_path /<token>/control/*` block must precede the `/v1` catch-all. The control-service must authenticate every request itself (`Authorization: Bearer <master key>`) — the path token is not authentication and appears in Caddy's access log (pre-existing, worth a note). Consider an IP allowlist on the control path in Caddy, and optionally a separate control key so MCP access cannot be replayed against `/v1` for completions. Also: nothing in the spec says the control-service checks auth at all — as drawn, a new unauthenticated HTTP service would sit behind only a secret path.

**m4 — `/decisions` promises fields the log does not contain.** Section: *Read endpoints*. `gate`, `bar`, `scores` exist only in `context.signals`, not in the `_log` record. Either enrich the decision record (recommended — the log is the durable artifact) or shrink the endpoint's promise.

**m5 — Extraction output needs schema validation.** Section: *Retention*. "True task kind" from an LLM can be anything; validate `kind ∈ BASE_BAR` keys (not `KIND_KEYWORDS` — see claim 14), cap phrase/key-point counts and lengths, drop non-conforming rows. Enforced at read time as well as write time.

**m6 — Single-worker assumption is load-bearing and undocumented.** Section: whole design. Module-level `STATE`, the learner queue, and single-writer `learned.json` are all correct only with one proxy process. The compose file runs one container with default workers, but nothing pins or asserts it. Document it; optionally assert at startup (fail loudly if `WEB_CONCURRENCY`/workers > 1).

**m7 — mtime-based tests need an injectable clock/stat.** Section: *Verification*. Tests for the mtime cache must not depend on real filesystem mtime granularity (notably on the Windows dev box or Docker Desktop bind mounts). Design the reader so tests can force cache invalidation (env knob or injectable stat), and keep `control.json` on a named volume in production.

**m8 — ESCALATE substring match persists noise.** Section: *Constraint that shapes everything*. `"ESCALATE" in ask` fires on "ESCALATED" and on any chat *about* the router (this spec itself would trigger it repeatedly). In-RAM and ephemeral, that was tolerable; persisted, discussion sessions dock trust. Fix the learning path to a word-boundary match (`\bESCALATE\b`), keep the legacy substring for compatibility, or at minimum mention the hazard.

**m9 — Session key weakness matters more for learning attribution.** Section: *Part 2* (lease rationale). The first-message hash (first 400 chars) collides across chats with identical openers — fine for stickiness, weak for the cross-session attribution B3 depends on. Document the limitation; consider a stronger correlation key for the learner only (first-message hash + first assistant turn hash), leaving today's session key untouched.

**m10 — "Next ask unrelated" is not what the code does.** Section: verdict table. The +0.03 reward fires on any 4th+ turn without a stall; there is no unrelatedness check. Either implement the qualifier (the B2 containment test gives it for free: not-a-re-ask ≈ unrelated-enough) or drop the word from the table.

**m11 — `POLICY_LEARN=0` vs the "reset" story.** Section: *Guardrails*. "Deleting `learned.json` resets to pristine" — with the B1 matrix, deleting yields zero learned influence immediately and the learner starts accumulating fresh evidence, healing the file on the next flush. That is pristine-plus-immediate-relearning; if the operator's intent was "off", they need `POLICY_LEARN=0` or the control toggle, not deletion. Say so explicitly so deletion is not mistaken for an off-switch.

### Questions

**q1 — Phase 3's "optional learned classifier": cut it?** Recommendation: yes. It is the only component that can exceed the influence bounds by construction (it *replaces* classification — see M3), it is marked optional, and `phrase_kind` already covers the stated need ("this phrase means debug, not factual"). Revisit only after 90 days of real `phrase_kind` data shows a gap. Cutting it also removes the phase-3-with-no-table-home inconsistency (M1).

**q2 — Should `kind_bar_bias` consume re-ask verdicts?** Recommendation: initially no — escalate and confirmed direct-switches only (both strong, both hard to produce accidentally). Re-ask evidence is the noisiest input (B2); let it feed `model_trust` (where it is clamped ±0.40 and per-cell sparse) but not the bar until the strict detector has field time. This keeps the "user finds design work under-served" signal clean.

**q3 — MCP connectivity (spec's "to verify" item 1): local stdio or VPS?** Recommendation: keep the local stdio server. It is just an HTTPS client; the only failure modes are a corporate proxy/firewall on the Windows box, which the spec's verification step will surface immediately. Moving it to the VPS forfeits stdio entirely (it would need SSE/streamable-HTTP transport) and gains nothing — the endpoint surface is unchanged either way, as the spec says. Add the Caddy route (m3) and control-service auth before testing.

**q4 — Does phase 3 reliability earn its place given litellm's cooldowns?** Recommendation: reduce to log-and-expose (write failures to a table or the decision log, surface in `/health`, no routing effect) or cut entirely. `allowed_fails: 2` + `cooldown_time: 60` already move traffic off failing deployments within a minute; a learned reliability term adds value only for *slow-burn* degradation, which one user's traffic will rarely measure meaningfully. Decide before phase 1 so the schema does not reserve a slot for it.

---

## 3. The two open items

### 90-day phrase-decay half-life — yes, but decay the evidence, not the bias

Adopt the half-life, implemented as **evidence decay**: every table row keeps `count` and `last_ts`; the effective count is `count × 0.5^(age_days/90)`; a row applies only while its effective count ≥ 8 (the existing gate). Consequences: a phrase last seen 90 days ago drops to half evidence and typically falls below the gate and vanishes on its own; phrases that keep appearing stay alive indefinitely; no separate decay math on the values, no new knob, and the same mechanism serves `kind_bar_bias` and `model_trust` for free. Add a hard row cap (e.g., 200 phrases, evict lowest effective count) so `learned.json` stays bounded. Approximation note: decaying by `last_ts` slightly under-decays counts accumulated over long periods; acceptable for this system, and worth one sentence in the spec.

### Lease visibility in `GET /health` — yes

Include `{active, model, until_epoch, set_at}` in `/health`, and add decision reason `pinned-lease` to the log so `GET /decisions` shows lease-driven turns historically. Rationale: the spec's own lease rationale is that any caller can set one — so any *other* caller needs to see it. Without visibility, a lease is indistinguishable from a router bug; the existing router-hygiene skill already warns agents not to misread a stable model as "the router is stuck", and an invisible lease is exactly that trap. Cost is nil: the control-service owns the lease state already. (B5's fixes — seconds-primary, clamped bounds, one lease at a time — should land at the same time.)

---

## 4. What holds up

Credit where due, because the review should not read as if the design were weak:

- Control-file-over-HTTP keeps the request path free of network dependencies and preserves fail-open — the right architecture, and the mtime read is cheap.
- Bounded influence enforced *at read time* (not write time) is the correct choice; write-time enforcement rots the moment a bug or hand-edit produces a bad file.
- Lease-over-pin is right; the reasoning about session identity is correct, and in-band directives staying in-band is the correct split.
- The verdict table's honesty about 429/5xx being reliability-not-quality is correct.
- The rule content is accurate against the code (claims 4, 11) and correctly excludes the bulk-tagging duplicate.
- Phasing bookkeeping (1) before extraction (2) is the right order — phase 1 is testable without any LLM dependency.
- The three-table decomposition with three different bounds matches how the signals actually differ in trustworthiness.

---

## Verdict

**needs-changes** — the architecture and phasing are sound, but B1–B5 must be resolved in the spec (and B4/B5/M5/M6/m2/m3 folded into the in-flight phase A work) before the learning phases are built on top of them. None of the blockers require abandoning the design; all five are resolved by the fixes above, which the accompanying plan (`docs/plans/2026-09-21-learning-and-control-plan.md`) treats as requirements.
