# Alternative: VPS + Cloudflare Tunnel

> **Superseded path.** The primary deployment is `DEPLOY.md` (nginx + Let's
> Encrypt direct TLS, DNS-only A record). Use this only if you want the VPS
> origin IP hidden, and accept that Cloudflare terminates TLS and can read
> request bodies.
>
> One correction to the notes below: the VPS PTR is `srv1914487.hstgr.cloud`
> (Hostinger, Frankfurt DE, 31.97.44.5). The hostname this draft used was
> wrong, and it does not change the steps.

Target: srvi84487.httpgeart.cloud (31.97.44.5), Debian 13, root.
Result: router at `https://router.ifmphk.com/v1`, no open ports, no certs
to manage, zero changes to SSL settings or other sites on the zone.

Architecture: Cursor -> Cloudflare edge (TLS) -> tunnel -> cloudflared on
VPS -> LiteLLM (localhost only) -> vendor APIs.

---

## Step 1 - DNS record (Cloudflare dashboard, 2 min)

DNS -> Add record:

- Type: `A`, Name: `router`, IPv4: `31.97.44.5`
- Proxy status: **DNS only** (grey). Tunnel does TLS; orange proxy would
  double-wrap and complicate cert issuance.
- TTL: Auto

## Step 2 - files onto VPS (from Windows, in llm-router folder)

```powershell
ssh root@31.97.44.5 "mkdir -p /opt/llm-router"
scp litellm-config.yaml docker-compose.yml .env.example root@31.97.44.5:/opt/llm-router/
```

## Step 3 - Docker + fill .env (on VPS)

```bash
ssh root@31.97.44.5
curl -fsSL https://get.docker.com | sh
cd /opt/llm-router
cp .env.example .env
nano .env
```

Fill in ONLY these five lines:

```
ROUTER_KEY_DEFAULT=<paste default key>
ROUTER_KEY_BASIC=<paste basic key>
ROUTER_KEY_DEFAULT_BASE=<default key's provider url>/v1
ROUTER_KEY_BASIC_BASE=<basic key's provider url>/v1
LITELLM_MASTER_KEY=<output of: openssl rand -hex 24>
```

(Domain vars unused in tunnel path; leave as-is.)

Mapping per key already fixed in litellm-config.yaml - nothing to edit:
Default key: glm-5.3-flash, kimi-k2.7-code, glm-5.3 (weight 1), claude-haiku-4-5, kimi-k3 (direct).
Basic key: qwen3-vl-flash, qwen3.8-flash, glm-5.3 (weight 9), deepseek-v4.1-flash, qwen3.8-max (direct).

## Step 4 - verify upstream model ids (on VPS)

```bash
source <(grep -v '^#' .env | sed 's/^/export /')
curl -s "$ROUTER_KEY_DEFAULT_BASE/models" -H "Authorization: Bearer $ROUTER_KEY_DEFAULT" | grep -o '"id":"[^"]*"'
curl -s "$ROUTER_KEY_BASIC_BASE/models" -H "Authorization: Bearer $ROUTER_KEY_BASIC" | grep -o '"id":"[^"]*"'
```

If any id differs from config (`glm-5.3-flash`, `glm-5.3`,
`kimi-k2.7-code`, `deepseek-v4.1-flash`, `claude-haiku-4-5`, `qwen3.8-flash`,
`qwen3-vl-flash`, `kimi-k3`, `qwen3.8-max`), fix the `model:` lines.

## Step 5 - start router

```bash
docker compose up -d
docker compose logs litellm | tail -20
# expect: "Application startup complete" + 7 model names incl cursor-auto
curl -s http://127.0.0.1:4000/v1/models -H "Authorization: Bearer $LITELLM_MASTER_KEY"
```

## Step 6 - cloudflared (host, not container; matches your live-* tunnels)

```bash
# install
curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared
chmod +x /usr/local/bin/cloudflared
cloudflared service install <EXISTING-TUNNEL-TOKEN>   # token from your CURRENT tunnel, see below
```

Choice: your panel already runs a tunnel (`tun-hkit-office`,
`tun-texwood-b01` are its CNAMEs). Two options:

- **New tunnel** (recommended, isolation): Cloudflare dashboard -> Zero
  Trust -> Networks -> Tunnels -> Create tunnel -> name `llm-router` ->
  copy the token into the command above. Then in the tunnel's Public
  Hostname tab: subdomain `router`, domain `ifmphk.com`, service
  `http://localhost:4000`. Save.
- **Existing tunnel**: add one more Public Hostname row to it, same
  service target. Token reuse, service install already done.

## Step 7 - test from Windows

```powershell
curl.exe -s https://router.ifmphk.com/v1/models -H "Authorization: Bearer <MASTER_KEY>"
```

Expect the 6 model ids. Then Cursor:

- Settings -> Models -> OpenAI API Key
- API Key: `<MASTER_KEY>`
- Override OpenAI Base URL: `https://router.ifmphk.com/v1`
- Add model `cursor-auto`, select it, Verify.

## Troubleshooting

- **403 private networks**: Base URL must be the public https one.
- **530/1033 from CF**: cloudflared down or Public Hostname not saved.
- **401**: Cursor key != LITELLM_MASTER_KEY.
- **Upstream 404**: model id mismatch, redo Step 4.
- **Router down after reboot**: `docker compose up -d` is
  restart-unless-stopped; cloudflared service install = systemd. Both
  survive reboots.
