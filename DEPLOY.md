# Deploy: VPS + nginx + Let's Encrypt

Cursor calls a custom base URL from its own servers, so the URL has to be
public HTTPS. Loopback and private addresses are refused no matter what is set
locally. This puts the router on the VPS and gives it a real certificate.

Result: `https://router.ifmphk.com/<token>/v1`, router bound to loopback,
nginx owning TLS, no third party able to read the traffic.

VPS: `31.97.44.5` (Hostinger, Frankfurt, DE), Debian, root. Note this is not
the hostname the earlier draft named; its PTR is `srv1914487.hstgr.cloud`.

---

## Step 0 - pick the region deliberately

Frankfurt is fine only if the provider endpoints are also far from you. Cursor
(HK) -> Frankfurt -> provider adds a leg that a Hong Kong or Singapore box
would not. Measure before committing:

```bash
# on the VPS, after .env is filled
source <(grep -v '^#' .env | sed 's/^/export /')
probe() {  # $1 = key, $2 = that key's base url
  echo "== $2"
  curl -s -o /dev/null -w 'connect=%{time_connect}s ttfb=%{time_starttransfer}s total=%{time_total}s\n' \
    "$2/models" -H "Authorization: Bearer $1"
}
probe "$ROUTER_KEY_DEFAULT" "$ROUTER_KEY_DEFAULT_BASE"
probe "$ROUTER_KEY_BASIC" "$ROUTER_KEY_BASIC_BASE"
```

Add ~450 ms if you are in Hong Kong and the box is in Frankfurt. If a
`ttfb` above ~0.3 s shows up on either leg and the providers are Asia-based,
move the router to a Hong Kong or Singapore VPS; nothing else in this setup
changes.

## Step 1 - DNS (Cloudflare dashboard, 2 min)

DNS -> Add record:

- Type `A`, Name `router`, IPv4 `31.97.44.5`
- Proxy status **DNS only (grey cloud)** - Cloudflare's proxy terminates TLS
  itself and breaks nginx's HTTP-01 certificate challenge
- TTL Auto

Confirm before continuing, from any machine:

```powershell
Resolve-DnsName router.ifmphk.com -Type A
```

## Step 2 - files onto the VPS

From the `llm-router` folder in Windows:

```powershell
ssh root@31.97.44.5 "mkdir -p /opt/llm-router"
scp litellm-config.yaml routing_policy.py control_store.py control_service.py learning.py extractor.py failure_hook.py laya_scorer.py docker-compose.yml deploy/nginx-router.ifmphk.com.conf .env.example DEPLOY.md root@31.97.44.5:/opt/llm-router/
```

## Step 3 - Docker and the env file (on the VPS)

```bash
ssh root@31.97.44.5
curl -fsSL https://get.docker.com | sh
cd /opt/llm-router
cp .env.example .env
nano .env
```

Fill six lines, nothing else:

```
ROUTER_KEY_DEFAULT=<default key>
ROUTER_KEY_BASIC=<basic key>
ROUTER_KEY_DEFAULT_BASE=<default key's provider url>/v1
ROUTER_KEY_BASIC_BASE=<basic key's provider url>/v1
LITELLM_MASTER_KEY=<openssl rand -hex 24>
CONTROL_TOKEN=<openssl rand -hex 24, distinct from the master key; see Step 8b>
```

## Step 4 - verify the upstream model ids

The config assumes these ids (all live-probed with 200s on 2026-09-21):
`glm-5.3-flash`, `glm-5.3`, `kimi-k2.7-code`, `claude-haiku-4-5`,
`kimi-k3` (default key); `qwen3-vl-flash`, `qwen3.8-flash`, `deepseek-v4.1-flash`,
`qwen3.8-max` (basic key). Upstream ids can still drift - re-probe if an id 404s.

```bash
source <(grep -v '^#' .env | sed 's/^/export /')
curl -s "$ROUTER_KEY_DEFAULT_BASE/models" -H "Authorization: Bearer $ROUTER_KEY_DEFAULT" | grep -o '"id":"[^"]*"'
curl -s "$ROUTER_KEY_BASIC_BASE/models" -H "Authorization: Bearer $ROUTER_KEY_BASIC" | grep -o '"id":"[^"]*"'
```

If an id differs, fix the `model:` line in `litellm-config.yaml` **and** the
matching key in `PROFILES` in `routing_policy.py`, then re-run
`python tests/validate_config.py` locally. That check exists so a rename cannot
silently leave a model unroutable.

While you are here, check the prices: `model_info` costs in the config are
USD per single token (per-1M price / 1e6) and must match `Profile.cost_in` /
`cost_out` in the policy, which are USD per 1M tokens. The shipped numbers are
real (operator HKD list at 7.8, 2026-09-21) and `tests/validate_config.py`
fails if the two sides drift apart.

## Step 5 - start the router

```bash
cd /opt/llm-router
docker compose up -d
docker compose logs litellm | tail -40
```

Expect `Application startup complete` and no `routing_policy` error. The proxy
validates the plugin at config load, so a bad path or a sync `run()` fails here
rather than on the first request.

```bash
curl -s http://127.0.0.1:4000/v1/models -H "Authorization: Bearer $LITELLM_MASTER_KEY"
```

Expect the model ids including `cursor-auto`.

## Step 6 - nginx + Let's Encrypt

> **Already done on this VPS (2026-09-21).** DNS, the nginx server block, and
> the certificate are live; the path token in the config is
> `24b8fa195657da6bc97495e5e28051b304bded55446933a5`. Keep this section for
> rebuilds; skip to Step 7 for verification.

The VPS runs nginx (it already serves ifmphk.com / direct.ifmphk.com), so
nginx — not Caddy — fronts the router. The server block in
`deploy/nginx-router.ifmphk.com.conf` maps the secret path prefix onto the two
upstreams and strips the prefix (`handle_path` equivalent, via a regex
capture):

    https://router.ifmphk.com/<token>/v1/...       -> 127.0.0.1:4000 (LiteLLM)
    https://router.ifmphk.com/<token>/control/...  -> 127.0.0.1:4010 (control)

Deploy it fresh (idempotent):

```bash
apt install -y nginx certbot python3-certbot-nginx
install -m 644 /opt/llm-router/deploy/nginx-router.ifmphk.com.conf \
  /etc/nginx/sites-available/router.ifmphk.com
ln -sf /etc/nginx/sites-available/router.ifmphk.com /etc/nginx/sites-enabled/
```

**Set the token before use.** Replace both occurrences of
`24b8fa195657da6bc97495e5e28051b304bded55446933a5` in the config with a long
random string (`openssl rand -hex 24`). It becomes part of Cursor's base URL.

Then request the certificate (HTTP block must be live first):

```bash
nginx -t && systemctl reload nginx
certbot --nginx -d router.ifmphk.com --non-interactive --agree-tos \
  --register-unsafely-without-email
```

Certbot rewrites the block with TLS lines and a renewal timer
(`certbot renew` runs twice daily). Nothing to maintain.

Ports 80/443 are already open in UFW on this box; if rebuilding from scratch:

```bash
ufw allow 80/tcp
ufw allow 443/tcp
```

## Step 7 - verify from Windows

```powershell
$k = 'the LITELLM_MASTER_KEY'
curl.exe -s https://router.ifmphk.com/<token>/v1/models -H "Authorization: Bearer $k"
```

Expect the model list. A wrong path gives `404 Not found`; a wrong key gives
`401`; a certificate problem gives a TLS error.

## Step 8 - point Cursor at it

Settings -> Models -> OpenAI API Key:

- API Key: `LITELLM_MASTER_KEY`
- Override OpenAI Base URL: `https://router.ifmphk.com/<token>/v1`
- Add model `cursor-auto`, select it, Verify

The direct model names (`kimi-k2.7-code`, `glm-5.3`, `deepseek-v4.1-flash`,
`kimi-k3`, `qwen3.8-max`) are also served, as an escape hatch when you want
one model on purpose.

Then register the MCP server in Cursor (`~/.cursor/mcp.json`) so agents can
inspect and steer the router. Set its env to the PUBLIC control URL so it
works from anywhere:

```json
"llm-router": {
  "command": "python",
  "args": ["-X", "utf8", "C:\\Users\\rchan\\random\\llm-router\\mcp_server.py"],
  "env": {
    "ROUTER_CONTROL_URL": "https://router.ifmphk.com/<token>/control",
    "ROUTER_MASTER_KEY": "<LITELLM_MASTER_KEY>"
  }
}
```

(The file already carries a localhost default for testing before the VPS
exists; switch the two values above once the nginx front door is live.)

## Step 8b - deploy the control service and its own auth

The control service (`control_service.py`) is the single writer of
`control.json` and serves the MCP tools. Two things the nginx path token
does not give it:

1. **Its own auth.** The path token is routing, not authentication - it
   appears in nginx's access log and in Cursor's config, so treat it as
   public. The service authenticates every request itself with
   `CONTROL_TOKEN`, a value distinct from `LITELLM_MASTER_KEY` (a leaked
   control token must not be replayable against `/v1` for completions).
   Add it to `/opt/llm-router/.env` on the VPS:

```
CONTROL_TOKEN=<openssl rand -hex 24>
```

   Without it the service runs read-only: reads are served, every write is
   refused, so a missing token fails closed instead of standing up an
   unauthenticated control plane.

2. **An nginx route.** The control plane lives under the same path token;
   the shipped server block already contains the regex location mapping
   `/<token>/control/...` to `127.0.0.1:4010` (see
   `deploy/nginx-router.ifmphk.com.conf`). If you changed the token in the
   `/v1` block, change it here too (`nginx -t && systemctl reload nginx`
   after editing).

Then verify the whole chain from Windows - unauthenticated must be refused,
authenticated must answer:

```powershell
curl.exe -s https://router.ifmphk.com/<token>/control/health
#   -> {"error": "unauthorized"}  (write-capable deploy) or a health body (read-only mode)
curl.exe -s https://router.ifmphk.com/<token>/control/health -H "Authorization: Bearer $CONTROL_TOKEN"
#   -> {"status": "ok", ...}
```

`docker compose up -d` starts the `control` container alongside `litellm`;
its mounts are crossed and read-only on purpose (the proxy can read
`control.json` but never write it; the control service reads the decision
log and `learned.json` but never writes them), so the single-writer rule is
enforced by the mounts rather than by convention.

## Step 9 - watch the routing decisions

```bash
docker compose exec litellm tail -f /app/logs/routing.jsonl
```

One JSON object per request: `chosen`, `kind`, `reason`, `est_tokens`,
`has_image`, `stall`, `directives`, and the candidate list. `reason` is
`fresh`, `pinned-cache` (same model held for the task, cache warm),
`pinned-directive` (a `[[use:...]]`), or the fallback paths. Each record
also carries a `learner` block (`queue_depth`, `errors`, `extractions`) -
the control service's `/health` surfaces the newest one.

## Step 10 - the learning subsystem (phase B)

The learner runs inside the proxy process and persists what it learns to
`/app/logs/learned.json` on the `router-logs` volume. It is the ONLY writer
of that file; the control service reads it read-only. Artifacts on the
`router-logs` volume:

| File | Written by | Contents |
|---|---|---|
| `routing.jsonl` | proxy | decision log, incl. the `learner` telemetry block |
| `learned.json` | proxy (learner) | `model_trust`, `kind_bar_bias`, `phrase_kind`, `key_points`, reset bookkeeping |
| `observations.jsonl` | proxy (learner) | sanitized distillations from phase-2 extraction (rotated at `POLICY_LOG_MAX_BYTES`) |
| `reliability.jsonl` | proxy (failure hook) | one `{ts, model, status}` line per upstream 429/5xx (phase 3: log-and-expose only) |

Raw prompt text NEVER reaches any of them: the ask queue is RAM-only
(bounded 40 items, 10 min max age), extraction results are mechanically
sanitized (credential-shape scrub + length caps) before persistence, and
`reliability.jsonl` carries no payload fields at all.

**Extraction call auth.** Phase 2 sends ONE `glm-5.3-flash` request per ~20
asks straight to the upstream provider - deliberately NOT through this
proxy (the router must not observe its own monitoring, review M2). It
reuses the existing `ROUTER_KEY_DEFAULT` / `ROUTER_KEY_DEFAULT_BASE` env
vars, so there is nothing new to provision; it just means those
credentials must remain valid for direct chat-completions calls (they
already are). With `POLICY_LEARN=0` the learner makes zero network calls.

**Verification after a day of traffic.**

```bash
curl -s http://127.0.0.1:4010/learned  -H "Authorization: Bearer $CONTROL_TOKEN" | python3 -m json.tool
curl -s http://127.0.0.1:4010/health   -H "Authorization: Bearer $CONTROL_TOKEN" | python3 -m json.tool
```

Expect: `learned.json` tables within bounds (trust +/-0.40, bars +/-0.15,
phrases +/-0.10), every `kind` value a real task kind, phrase rows <= 200,
`/health` showing `learner.queue_depth` and per-model `reliability`
counters. Learning knobs (all optional):

- `POLICY_LEARN=0` - the hard kill-switch; beats the control toggle. No
  learning hooks, no extraction calls, no writes.
- `POST /learning {"enabled": false}` - the daily-use toggle.
- `POST /learning {"reset": "phrases|trust|bars|all"}` - wipes one table
  (or all) via an apply-once intent; the proxy consumes it without the
  control service ever touching `learned.json`.
- `POLICY_EXTRACT_POLL_S` / `POLICY_FLUSH_POLL_S` - learner loop cadence
  (defaults 30 s / 60 s).
- `POLICY_OBSERVATIONS_PATH` - where observations.jsonl lives (default:
  next to the decision log).
- `POLICY_CASCADE_WINDOW_S` - cascade-lite failure window (default 120 s).
  A model that failed upstream inside this window is excluded from the
  next turn's decision (unless explicitly leased or `[[use:]]`-pinned),
  and the session's held model takes a trust penalty. Off with learning.
- `POLICY_CASCADE_TRUST_PENALTY` - that penalty (default -0.25).

**Laya shadow scorer.** Disabled by default (`laya` weight 0). To activate:

```bash
# inside the container, once
docker compose exec litellm pip install laya
docker compose restart litellm
```

Then set the weight via MCP `router_weights {"laya": 0.2}` (or the control
service `POST /weights`). Until then the scorer only logs to
`laya_shadow.jsonl`; scores never touch routing decisions.

---

## Operating it

**Everyday**

```bash
docker compose logs -f litellm          # proxy
docker compose restart litellm          # after a config edit
docker compose logs -f control          # control plane
docker compose exec litellm cat /app/logs/routing.jsonl | tail -50
```

**Retune without a code edit.** The policy reads these from the environment:
`POLICY_W_COST`, `POLICY_W_TRUST`, `POLICY_W_HEADROOM`, `POLICY_W_LATENCY`,
`POLICY_SESSION_TTL`, `POLICY_DISABLE=1`. Set them under `environment:` in
`docker-compose.yml` and restart.

**Change routing logic.** Edit `routing_policy.py` (and/or `control_store.py`,
`control_service.py`, `learning.py`, `extractor.py`, `failure_hook.py`,
`laya_scorer.py`), then
run all checks locally before shipping it:

```powershell
python tests\test_routing_policy.py    # algorithm
python tests\test_control.py           # control plane
python tests\test_learning.py          # learner: verdicts, gates, persistence
python tests\test_extractor.py         # phase-2 extraction client (mocked HTTP)
python tests\test_failure_hook.py      # phase-3 reliability hook
python tests\test_laya_scorer.py       # laya shadow scorer (skips if laya absent)
python -X utf8 tests\test_golden.py    # fail-open golden replay (B1)
python tests\validate_config.py        # config <-> profiles <-> plugin
.\tests\smoke_proxy.ps1                # litellm's loader + the real pipeline
```

`smoke_proxy.ps1` boots the proxy on port 4099 against placeholder upstreams
and asserts a debug ask lands on `kimi-k2.7-code`. It is the only check that
exercises litellm's plugin loader, and it is what caught the two startup bugs
this setup shipped with (a dataclass load failure and a plugin path naming a
class). Both passed the unit tests.

Then `scp` the file and `docker compose restart litellm`. Never edit it only on
the VPS: the repo is the source of truth and the tests are the reason the
algorithm stays honest.

**Rollback.** Keep the previous `litellm-config.yaml` and `routing_policy.py`
alongside as `.bak`. `docker compose restart litellm` picks the file up; no
image rebuild is involved.

**When the router is down, Cursor is down.** The escape hatch is Cursor's own
model picker: switch to a native Cursor model and keep working. The tunnel and
container both survive reboots (`restart: unless-stopped` + systemd), but a
Hostinger-side outage is not something this setup can route around.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `403 Access to private networks are forbidden` | base URL is not the public https one |
| `404 Not found` from nginx | path token in the base URL does not match the server block |
| `401` | Cursor's key is not `LITELLM_MASTER_KEY` |
| TLS error on first request | grey-cloud DNS record or ports 80/443 closed |
| `502 Bad Gateway` on tokened paths | router container not up yet, or not listening on 127.0.0.1:4000/4010 |
| `Upstream 404` | upstream model id mismatch, redo Step 4 |
| All traffic on one model | check the decision log's `reason`; `pinned-cache` means a session is holding it, `skipped-not-fleet-group` means the group guard fired |
| Plugin never runs | `router_settings.plugins` path must be loadable next to the config |
