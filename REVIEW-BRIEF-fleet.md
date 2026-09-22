# Review brief: fleet swap + default-path hardening

## Context
The llm-router is a LiteLLM proxy whose `routing_policy.py` plugin picks one of
8 models per request for the `cursor-auto` group. This review covers the
follow-up work done after the phase-B learning subsystem landed (all 9 suites
currently green). No git in this repo; review the files as they are.

## Scope (files changed in this pass)
- `litellm-config.yaml` — full fleet swap: 8 cursor-auto deployments
  (qwen3-vl-flash, glm-5.3-flash, qwen3.8-flash, deepseek-v4.1-flash,
  kimi-k2.7-code, glm-5.3 on both keys w=9/1, claude-haiku-4-5) + direct
  escape hatches (glm-5.3, kimi-k2.7-code, glm-5.3-flash, deepseek-v4.1-flash,
  kimi-k3, qwen3.8-max). Real HKD->USD prices (7.8 rate).
- `routing_policy.py` — `PROFILES` rewritten for the 8-model fleet
  (kimi design cap 0.78->0.70, weak += design; qwen3.8-flash added;
  claude-haiku-4-5 added). Default state dir is now platform-aware
  (`/app/logs` if it exists, else `tempfile.gettempdir()`) at 3 sites.
- `control_store.py` — `_default_state_dir()` helper; `default_store()` uses it.
- `learning.py`, `failure_hook.py`, `control_service.py` — same default-dir
  pattern.
- `mcp_server.py` — `FLEET` tuple updated to the 7-profile fleet (glm-5.3 once).
- `tests/validate_config.py` — now sets POLICY_STATE_DIR/LEARNED_PATH/
  CONTROL_PATH to a temp dir before importing the policy (was writing learned
  state to C:\app\logs on Windows); bulk expectation now checks cost_in <= 0.05
  instead of a hard-coded model id.
- `tests/test_golden.py` — same temp-state isolation; bulk counterfactual now
  expects qwen3.8-flash as the runner-up after negative trust on qwen3-vl-flash.
- `tests/test_routing_policy.py`, `tests/test_control.py`,
  `tests/golden_routing.json`, `tests/smoke_proxy.ps1` — id renames
  (kimi-2.7-code -> kimi-k2.7-code, deepseek-4.1-flash -> deepseek-v4.1-flash)
  and the two bulk expectation updates.
- Docs: DEPLOY.md, README.md, .env.example, .env comments,
  VPS-DEPLOY-cloudflare-tunnel.md, deploy/nginx-router.ifmphk.com.conf,
  .cursor/rules/router-invariants.mdc (text-only list now deepseek + qwen3.8),
  ~/.cursor/skills/router-hygiene/SKILL.md.

## Focus points
1. Any place still assuming the old 4-model fleet or old ids (excluding
   docs/plans + docs/specs dated 2026-09-21 review/plan historical blocks,
   logs/, and the golden JSON which is regenerated).
2. The default-dir change must not break the Docker path: in-container
   `/app/logs` exists, so `os.path.isdir("/app/logs")` must be True there.
   Check control_store._default_state_dir logic for edge cases (e.g. Windows
   containers, relative-path weirdness).
3. mcp_server.FLEET vs routing_policy.PROFILES sync: is any test pinning them
   together? The comment claims tests/test_control.py keeps them in sync —
   verify that is still true.
4. Price consistency: `input_cost_per_token` in the YAML must equal
   `cost_in / 1e6` in PROFILES for every model; validate_config checks this,
   but sanity-check the qwen3-vl-flash (0.016/0.18), qwen3.8-flash
   (0.12/0.38), deepseek-v4.1-flash (0.30/1.21), kimi-k2.7-code (0.77/3.22),
   glm-5.3 (1.13/3.54), claude-haiku-4-5 (1.01/5.03) entries specifically.
5. The bulk counterfactual in test_golden: is `qwen3.8-flash` genuinely the
   next-best bulk pick after negative trust on qwen3-vl-flash, or did the test
   just get bent to pass?
6. validate_config's `policy.STATE = policy.PolicyState()` reset: does the
   policy read STATE lazily elsewhere such that this could mask a real issue?

## Verdict format
One-line findings with severity (blocker/major/minor), sorted by file and line,
plus a totals line. Empty output sections are fine — do not invent findings.
