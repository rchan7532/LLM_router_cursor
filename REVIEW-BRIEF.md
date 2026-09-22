# Review brief — routing_policy.py

One-off brief for `cavecrew-reviewer`. Safe to delete after the review pass.

## What to run

```
cavecrew-reviewer on the llm-router routing policy: read REVIEW-BRIEF.md, then the files it lists
```

## What this code is

`routing_policy.py` is a LiteLLM `RoutingPlugin` that runs on every request to
the `cursor-auto` model group. It narrows `context.candidate_models` (a list of
`litellm_params.model` strings such as `openai/glm-5.3`) down to the models
allowed to serve the request. Leaving the list untouched means "no opinion";
narrowing it to an empty list makes LiteLLM raise and fails the user's request.

## Files

- `routing_policy.py` — the main artifact
- `litellm-config.yaml` — model fleet and plugin wiring
- `docker-compose.yml`, `deploy/nginx-router.ifmphk.com.conf` — deployment
- `tests/test_routing_policy.py`, `tests/validate_config.py`, `tests/smoke_proxy.ps1`

## Things the reviewer must know to judge correctly

1. **It must never raise.** An exception in the plugin breaks a user request. A
   `try/except` wraps `_run` and records `context.signals["policy_error"]`.
   Check that no expected failure path escapes it, including failures raised
   while building signals.
2. **already fixed, do not re-report** — two startup bugs were found by running
   the real proxy and are fixed:
   - `dataclasses` with `from __future__ import annotations` crashes under
     LitellM's `importlib` loader, which does not register the module in
     `sys.modules`. Fixed by using `NamedTuple` and plain classes. Verify the
     fix is complete: nothing in the module may assume registration.
   - The config must name the plugin **instance**, not the class, because a
     class satisfies the `RoutingPlugin` Protocol check and then fails per
     request. Fixed by `router_policy = CursorAutoPolicy()`.
3. **Concurrency.** Module-level `STATE` holds session pins and a trust table,
   shared across all requests in a process. `run` is async but its body is
   synchronous. Look hard at thread-safety, unbounded growth, and whether
   eviction is correct and can ever drop a live session.
4. **The trust table** is a learning signal: escalation and stall lower a
   model's trust for a task kind, clean multi-turn work raises it. Check the
   arithmetic and the clamping at `TRUST_FLOOR` / `TRUST_CEILING`.
5. **`decide()`** is the pure selection core: hard gates (vision, context
   window), then a capability bar, then
   `capability - cost + trust + headroom - latency`. Check the gate-relaxation
   fallbacks and the `_normalize` helper for division by zero or empty input.
6. **`tests/validate_config.py`** asserts the config's `model_info` costs match
   the `Profile` costs and that every model in the `cursor-auto` group has a
   profile. Check the assertion holds in both directions and cannot pass
   vacuously.

Focus on real defects: crashes, wrong routing decisions, unbounded growth,
concurrency, silently swallowed errors. Ignore style, naming, and design taste.

## Known non-issues

- Prices in `PROFILES` and `model_info` are deliberate placeholders to be filled
  from provider pricing pages; do not report them as wrong values.
- The proxy returns 500 in the smoke test because its upstreams point at
  `example.invalid` on purpose.
