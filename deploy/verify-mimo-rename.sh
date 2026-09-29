#!/bin/bash
# Rename mimo -> mimo-v2.6-pro: verify live group name + probe through proxy.
cd /opt/llm-router
echo "=== config grep ==="
grep -A2 "model_name: mimo" litellm-config.yaml | head -6
echo "=== container up ==="
docker ps --format '{{.Names}} {{.Status}}' | grep llm-router
echo "=== probe mimo-v2.6-pro through proxy (public endpoint) ==="
set -a; source <(sed 's/\r$//' .env); set +a
curl -s -m 60 https://router.ifmphk.com/24b8fa195657da6bc97495e5e28051b304bded55446933a5/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H "Content-Type: application/json" \
  -d '{"model":"mimo-v2.6-pro","messages":[{"role":"user","content":"hi"}],"max_tokens":3}' | head -c 300
echo
echo "=== probe old name mimo (should 400/404) ==="
curl -s -m 30 https://router.ifmphk.com/24b8fa195657da6bc97495e5e28051b304bded55446933a5/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H "Content-Type: application/json" \
  -d '{"model":"mimo","messages":[{"role":"user","content":"hi"}],"max_tokens":3}' | head -c 200
echo
echo "=== latest mimo receipt ==="
sleep 8
docker exec llm-router sh -c 'grep mimo /app/logs/routing_usage.jsonl | tail -1 | cut -c1-260'
