#!/bin/bash
# Complete the broken attribution deploy: usage_state.py was left stale on
# the VPS, so usage_hook's task= call raised TypeError and receipts died.
set -e
cd /opt/llm-router
echo "=== copy current usage_state.py over ==="
docker compose restart litellm 2>/dev/null || docker restart llm-router
for i in $(seq 1 20); do
  S=$(docker inspect llm-router --format '{{.State.Health.Status}}' 2>/dev/null || echo unknown)
  echo "attempt $i: $S"
  [ "$S" = healthy ] && break
  sleep 10
done
echo "=== smoke: stamped call then receipt check ==="
set -a; source <(sed 's/\r$//' .env); set +a
curl -s https://router.ifmphk.com/51fffd0555fe5909cffa13899c68f33720de183dfa241de8/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -H "User-Agent: OpenAI/Python 2.20.0" \
  -d '{"model":"cursor-auto","messages":[{"role":"user","content":"say ok"}],"max_tokens":4}' -o /dev/null
echo "call sent, waiting for callback"
sleep 15
docker exec llm-router sh -c 'tail -n 1 /app/logs/routing_usage.jsonl | cut -c1-220'
