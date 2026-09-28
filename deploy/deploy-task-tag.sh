#!/bin/bash
# Full deploy of task-tag measurement: sync files, recreate containers, verify.
set -e
cd /opt/llm-router
echo "=== recreate both containers ==="
docker compose up -d --force-recreate
for i in $(seq 1 25); do
  S1=$(docker inspect llm-router --format '{{.State.Health.Status}}' 2>/dev/null || echo unknown)
  S2=$(docker inspect llm-router-control --format '{{.State.Health.Status}}' 2>/dev/null || echo unknown)
  echo "attempt $i: proxy=$S1 control=$S2"
  [ "$S1" = healthy ] && [ "$S2" = healthy ] && break
  sleep 10
done
echo "=== /usage must now include by_task + task_redo keys ==="
set -a; source <(sed 's/\r$//' .env); set +a
curl -s http://127.0.0.1:4010/usage -H "Authorization: Bearer $CONTROL_TOKEN" | python3 -c "
import json, sys
s = json.load(sys.stdin)['summary']
print('has by_task:', 'by_task' in s, '| has task_redo:', 'task_redo' in s)
print('by_task content:', json.dumps(s.get('by_task'))[:200])
print('task_redo:', s.get('task_redo'))
"
echo "=== stamped-path smoke: tag rides into decision ==="
curl -s https://router.ifmphk.com/51fffd0555fe5909cffa13899c68f33720de183dfa241de8/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -H "User-Agent: OpenAI/Python 2.20.0" \
  -d '{"model":"cursor-auto","messages":[{"role":"user","content":"say ok [[task:deploy-verify]]"}],"max_tokens":4}' -o /dev/null
sleep 6
docker exec llm-router sh -c 'tail -n 1 /app/logs/routing.jsonl' | grep -o '"task": "[a-z-]*"'
docker exec llm-router sh -c 'tail -n 1 /app/logs/routing_usage.jsonl | cut -c1-200'
