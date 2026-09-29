#!/bin/bash
# Deploy escape hatches: sync config + pricing, recreate proxy, probe both.
set -e
cd /opt/llm-router
docker compose up -d --force-recreate litellm
for i in $(seq 1 25); do
  S=$(docker inspect llm-router --format '{{.State.Health.Status}}' 2>/dev/null || echo unknown)
  echo "attempt $i: $S"
  [ "$S" = healthy ] && break
  sleep 10
done
set -a; source <(sed 's/\r$//' .env); set +a
echo "=== probe glm-5.3-flashx through proxy ==="
curl -s -m 60 http://127.0.0.1:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H "Content-Type: application/json" \
  -d '{"model":"glm-5.3-flashx","messages":[{"role":"user","content":"hi"}],"max_tokens":3}' | head -c 150
echo
echo "=== probe mimo through proxy ==="
curl -s -m 60 http://127.0.0.1:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H "Content-Type: application/json" \
  -d '{"model":"mimo","messages":[{"role":"user","content":"hi"}],"max_tokens":3}' | head -c 150
echo
sleep 6
echo "=== receipts for both probes ==="
docker exec llm-router sh -c 'grep -E "flashx|mimo" /app/logs/routing_usage.jsonl | tail -2 | cut -c1-230'
echo "=== cursor-auto unaffected: fleet decision still works ==="
curl -s http://127.0.0.1:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H "Content-Type: application/json" \
  -d '{"model":"cursor-auto","messages":[{"role":"user","content":"hi"}],"max_tokens":3}' -o /dev/null -w "cursor-auto http=%{http_code}\n"
