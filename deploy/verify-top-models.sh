#!/bin/bash
# Probe kimi-k3 + qwen3.8-max-0902 through the public router; verify the
# receipt prices them (real_cost_hkd > 0 from the new PRICING entries).
set -e
cd /opt/llm-router
set -a; source <(sed 's/\r$//' .env); set +a
BASE="https://router.ifmphk.com/${MAIN_PATH_TOKEN}/v1/chat/completions"

echo "=== kimi-k3 probe ==="
curl -s -m 120 "$BASE" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H "Content-Type: application/json" \
  -H "X-Client-Type: cursor" \
  -d '{"model":"kimi-k3","messages":[{"role":"user","content":"hi [[task:k3-trial]]"}],"max_tokens":3}' \
  -o /tmp/k3probe.json -w "http=%{http_code}\n"
head -c 200 /tmp/k3probe.json; echo

echo "=== qwen3.8-max-0902 probe ==="
curl -s -m 120 "$BASE" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H "Content-Type: application/json" \
  -H "X-Client-Type: cursor" \
  -d '{"model":"qwen3.8-max-0902","messages":[{"role":"user","content":"hi [[task:qmax-trial]]"}],"max_tokens":3}' \
  -o /tmp/qmaxprobe.json -w "http=%{http_code}\n"
head -c 200 /tmp/qmaxprobe.json; echo

sleep 6
echo "=== last 2 receipts (expect real_cost_hkd > 0, task set) ==="
docker exec llm-router sh -c 'tail -2 /app/logs/routing_usage.jsonl'
