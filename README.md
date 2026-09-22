# LLM Router - task-aware model selection for Cursor

One local (or VPS) LiteLLM proxy in front of both of your API keys. Cursor
points at it once and asks for a single model, `cursor-auto`. A policy plugin
reads each request and decides which model should serve it.

```
Default key: glm-5.3-flash, glm-5.3, kimi-k2.7-code, claude-haiku-4-5, kimi-k3 (direct)
Basic key:   qwen3-vl-flash, qwen3.8-flash, glm-5.3, deepseek-v4.1-flash, qwen3.8-max (direct)
```

## Why a policy plugin, not tiers

Fixed tiers answer "how hard is this?" but not "which model, and at what
price?". `routing_policy.router_policy` is a litellm `RoutingPlugin`: it
sees the whole request before routing and narrows the candidate deployments to
the one model that should serve it. litellm still owns retries, per-deployment
cooldowns and cross-key failover.

The pipeline:

1. **Hard gates** - drop models that cannot serve the request at all. An image
   turn never reaches a text-only model; a prompt larger than a model's window
   never reaches that model.
2. **Task signature** - what kind of work the newest human ask is: debug,
   refactor, review, design, code_gen, code_edit, explain, bulk, factual,
   writing, agentic. Plus scale (context size, turn count) and stall detection
   over the assistant's own recent tool calls.
3. **Capability bar** - the minimum capability that work needs. A cheap model
   is only eligible when it clears the bar, so cost pressure can never route a
   hard debug to a flash model.
4. **Utility** - `capability - cost + learned trust + headroom - latency` over
   the eligible models, argmax, ties to the cheaper model.
5. **Stickiness** - the winner is pinned to the task so the provider's
   prompt cache stays warm - but as a bounded preference, not an override.
   Turns 1-4 hold unconditionally (`pinned-grace`); after that the held
   model re-competes each turn against the fleet with a decaying loyalty
   bonus (`pinned-loyalty`), and a challenger that beats it by more than
   the bonus takes over (`pin-swap`, which restarts grace so there is no
   ping-pong). A High-importance/spec turn always re-decides: if the
   current bar outruns the held model's capability, the pin breaks on the
   spot. One expensive early win can no longer lock in a whole session.
6. **Feedback** - an escalation or a stalled loop lowers that model's trust for
   that kind of work in-process. The JSONL decision log is the raw material for
   offline tuning.

Full path: `routing_policy.py`.

## Caller directives

Put these in a message, in the newest turn:

| Directive | Effect |
|---|---|
| `[[escalate]]` or plain `ESCALATE` | raise the capability bar two steps and mark the turn High-importance |
| `[[cheap]]` | drop to the cheapest model that still clears the gates |
| `[[low]]` | mark this turn Low-importance: bar down one step, flash models serve it |
| `[[high]]` | mark this turn High-importance: bar up one step, premium models stay |
| `[[use:kimi-k2.7-code]]` | pin an exact model for this task |

Importance is also auto-detected from the marginal ask (one-liners,
"summarize/what does this" -> Low; spec/plan/architect phrasing -> High),
and a premium model pinned by stickiness will not serve a Low follow-up:
the pin breaks and a cheap flash model takes the turn. `ESCALATE` also
penalises the model that was serving the task, so the signal feeds the
trust table rather than only this one request.

### Per-session budget cap

The control plane accepts a `budget_hkd` value. Once a session's estimated
spend crosses the cap, every subsequent `cursor-auto` turn is forced to the
cheapest capable model unless you explicitly override with `[[high]]` or
`[[use:model]]`. The cap is read from `control.json`; set it with the MCP
tool `router_budget`, the HTTP endpoint `POST /budget {hkd: 30}`, or by
hand-editing `control.json`. The dashboard shows the live cap and turns the
row red once a session is in budget-alarm state. This is the only control
that would have stopped the 162 HKD hour: importance routing already chose
the right model per turn, but the volume of ~130k-token premium turns still
added up.

### Cascade-lite failure escalation

`failure_hook.py` records upstream 429/5xx failures per model. The policy
reads those timestamps before each `cursor-auto` decision. If the previous
model (or any fleet model) failed within the last 120 seconds, it is
excluded for this turn and its trust is docked, so the retry naturally
goes to the next best model. `[[use:model]]` and live leases override the
exclusion. When this changes the pick the decision reason is
`cascade-escalation`. The window is tunable via
`POLICY_CASCADE_WINDOW_S`; cascade is disabled when learning is off.

### Laya shadow scorer

`laya_scorer.py` optionally loads `convaiinnovations/laya` and scores each
routing signature against the task-kind taxonomy in a background thread.
By default the control-plane `laya` weight is `0`, so scores are only
written to `laya_shadow.jsonl` and have no routing effect. Set the weight
to a value between `0` and `1` with the MCP tool `router_weights` or the
HTTP endpoint `POST /weights {laya: 0.3}` to blend a small additive nudge
into the utility function. Because laya is near-chance zero-shot on custom
taxonomies, this is shadow-only until you have labeled data to fine-tune.

## Files

| File | Role |
|---|---|
| `litellm-config.yaml` | model fleet, the `cursor-auto` group, `router_settings.plugins`, fallbacks |
| `routing_policy.py` | the policy plugin: profiles, signature detection, selection, feedback, control-plane reads |
| `failure_hook.py` | litellm CustomLogger that records upstream 429/5xx failures for cascade-lite |
| `laya_scorer.py` | optional local shadow scorer using convaiinnovations/laya (telemetry-only by default) |
| `control_store.py` | fail-open reader for `control.json` + `learned.json` (mtime-cached) |
| `control_service.py` | the control plane: single writer of `control.json`, HTTP API for leases/weights/learning |
| `mcp_server.py` | local stdio MCP server exposing the control plane to Cursor agents |
| `tests/test_routing_policy.py` | 34 property tests over the algorithm, including one that loads the module the way litellm does |
| `tests/test_control.py` | 35 tests: store clamps, lease gates, fail-open contract, HTTP surface, MCP fleet pin |
| `tests/test_mcp_e2e.py` | end-to-end: stdio MCP client -> control service, incl. degraded mode |
| `tests/validate_config.py` | pre-flight: config, profiles and plugin agree |
| `tests/smoke_proxy.ps1` | boots the real proxy on port 4099 and asserts it routes and logs a decision |
| `deploy/nginx-router.ifmphk.com.conf` | nginx server block for the VPS front door, incl. the `/control` route |
| `DEPLOY.md` | VPS deployment, operations, troubleshooting |
| `VPS-DEPLOY-cloudflare-tunnel.md` | alternative exposure path if you want the origin IP hidden |
| `docker-compose.yml` | litellm + control service, healthchecks, shared volume |
| `.env.example` | the five values to fill |

## Quick local test

```powershell
cd llm-router
python tests\test_routing_policy.py     # algorithm (21 tests)
python tests\test_control.py            # control plane (32 tests)
python tests\validate_config.py         # config <-> profiles <-> plugin
.\tests\smoke_proxy.ps1                 # boots the real proxy, asserts it routes
python tests\test_mcp_e2e.py            # MCP server <-> control service, end to end
```

No API keys needed for any of them. The smoke test points the fleet at
unreachable placeholder upstreams on purpose: routing happens before the
upstream call, so it asserts the decision log without spending a token. It is
the only one of the three that exercises litellm's plugin loader and the real
routing pipeline, and it is the only one that caught the two startup bugs below.

```powershell
Copy-Item .env.example .env   # then fill it
.\start-router.ps1
```

For local testing with the MCP server, also start the control service in a
second terminal — it defaults to `127.0.0.1:4010`, which is where
`~/.cursor/mcp.json` points the `llm-router` MCP server by default:

```powershell
python control_service.py
```

Note that Cursor cannot reach a localhost base URL: it fetches custom base
URLs server-side and refuses private addresses. Local runs are for testing with
`curl` and for the test suite; Cursor needs the public endpoint from
`DEPLOY.md`.

## Tuning without a code edit

Set these in the container environment:

| Variable | Default | Meaning |
|---|---|---|
| `POLICY_W_COST` | 0.35 | price pressure |
| `POLICY_W_TRUST` | 0.30 | weight on learned success |
| `POLICY_W_HEADROOM` | 0.10 | bonus for room left in the window |
| `POLICY_W_LATENCY` | 0.10 | speed pressure |
| `POLICY_SESSION_TTL` | 3600 | seconds a task keeps its pinned model |
| `POLICY_DISABLE` | unset | `1` turns the policy off (litellm's own selection applies) |
| `POLICY_LOG` | `/app/logs/routing.jsonl` | decision log path |

## Before you trust the numbers

`PROFILES` in `routing_policy.py` carries a cost per model and a capability
score per task kind. The costs are real, from the operator's HKD list at
7.8 HKD/USD (2026-09-21); the capability scores are judgements that real usage
should confirm. Wrong
numbers do not break routing, they only skew it. `tests/validate_config.py`
fails if the config's `model_info` costs and the profile costs drift apart, so
correcting one without the other cannot slip through.

`model_info` also carries `cache_read_input_token_cost` and
`cache_creation_input_token_cost`. Those are placeholders, and litellm
prices cache reads at zero when they are absent, which quietly understates the
value of the stickiness layer in the spend log. Fill them from each provider's
cache pricing so the numbers you tune against are real.

## Two failure modes this setup already guards against

Both were found by running the real proxy, not by the unit tests, and both are
now covered by `tests/validate_config.py`:

- **The plugin module must not assume `sys.modules` registration.** litellm
  loads it with `importlib` without registering it, so `dataclasses` with
  stringized annotations crashes at startup on Python 3.14. The module uses
  `NamedTuple` and plain classes, and one test loads it through litellm's own
  loader.
- **The config must name the plugin instance, not the class.** A class passes
  litellm's `RoutingPlugin` Protocol check and then fails on every request.
  `router_settings.plugins` therefore names `routing_policy.router_policy`, and
  validation asserts the resolved object is not a type.
