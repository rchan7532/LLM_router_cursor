#!/bin/bash
# Diagnose: old vs new tokens, proxy health, and what Cursor/mimo would hit
set -e
cd /opt/llm-router
echo "=== container health ==="
docker inspect llm-router --format 'proxy: {{.State.Status}} {{.State.Health.Status}}' 2>/dev/null
docker inspect llm-router-control --format 'control: {{.State.Status}} {{.State.Health.Status}}' 2>/dev/null
echo "=== current path tokens (root-only file) ==="
cat /root/.router-path-tokens 2>/dev/null || echo "(no token file)"
echo "=== nginx locations currently live ==="
grep -E "location.*(24b8fa|51fffd|47d9ed|e339b4)" /etc/nginx/sites-available/router.ifmphk.com | head -10
echo "=== old main token (what Cursor likely still uses) ==="
curl -s -o /dev/null -w "HTTP %{http_code}\n" "https://router.ifmphk.com/24b8fa195657da6bc97495e5e28051b304bded55446933a5/v1/models"
echo "=== new main token (from .env) ==="
set -a; source <(sed 's/\r$//' .env); set +a
curl -s -o /dev/null -w "HTTP %{http_code}\n" "https://router.ifmphk.com/${MAIN_PATH_TOKEN}/v1/models"
echo "=== loopback proxy (bypass nginx) ==="
set -a; source <(sed 's/\r$//' .env); set +a
curl -s -o /dev/null -w "HTTP %{http_code}\n" "http://127.0.0.1:4000/v1/models" -H "Authorization: Bearer $LITELLM_MASTER_KEY"
echo "=== recent nginx 404s ==="
tail -n 30 /var/log/nginx/access.log 2>/dev/null | grep -E "404|400|401" | tail -8
