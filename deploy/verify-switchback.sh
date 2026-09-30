#!/bin/bash
# Verify switch-back detector live: same session (same first user msg),
# one escape-hatch turn then one fleet turn -> fleet receipt must carry
# switch_back_from=mimo-v2.6-pro exactly once.
cd /opt/llm-router
docker restart llm-router llm-router-control
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
  S=$(docker inspect llm-router --format '{{.State.Health.Status}}' 2>/dev/null)
  T=$(docker inspect llm-router-control --format '{{.State.Health.Status}}' 2>/dev/null)
  echo "attempt $i: router=$S control=$T"
  [ "$S" = healthy ] && [ "$T" = healthy ] && break
  sleep 8
done
set -a; source <(sed 's/\r$//' .env); set +a
SEED="switchback-probe-20260929"
BASE=https://router.ifmphk.com/${MAIN_PATH_TOKEN}/v1/chat/completions
echo "=== turn 1: escape hatch (mimo) in session ==="
curl -s -m 60 "$BASE" -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H "Content-Type: application/json" \
  -H "X-Client-Type: cursor" \
  -d "{\"model\":\"mimo-v2.6-pro\",\"messages\":[{\"role\":\"user\",\"content\":\"$SEED\"},{\"role\":\"assistant\",\"content\":\"ok\"},{\"role\":\"user\",\"content\":\"work [[task:mimo-trial]]\"}],\"max_tokens\":3}" -o /dev/null -w "http=%{http_code}\n"
sleep 3
echo "=== turn 2: fleet (cursor-auto) same session -> expect switch_back_from ==="
curl -s -m 60 "$BASE" -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H "Content-Type: application/json" \
  -H "X-Client-Type: cursor" \
  -d "{\"model\":\"cursor-auto\",\"messages\":[{\"role\":\"user\",\"content\":\"$SEED\"},{\"role\":\"assistant\",\"content\":\"ok\"},{\"role\":\"user\",\"content\":\"continue the work\"}],\"max_tokens\":3}" -o /dev/null -w "http=%{http_code}\n"
sleep 3
echo "=== turn 3: fleet again -> flag must NOT repeat ==="
curl -s -m 60 "$BASE" -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H "Content-Type: application/json" \
  -H "X-Client-Type: cursor" \
  -d "{\"model\":\"cursor-auto\",\"messages\":[{\"role\":\"user\",\"content\":\"$SEED\"},{\"role\":\"assistant\",\"content\":\"ok\"},{\"role\":\"user\",\"content\":\"keep going\"}],\"max_tokens\":3}" -o /dev/null -w "http=%{http_code}\n"
sleep 8
echo "=== last 3 receipts ==="
docker exec llm-router sh -c 'tail -3 /app/logs/routing_usage.jsonl'
echo "=== /usage switch_backs ==="
TOKEN=$(docker exec llm-router-control sh -c 'echo -n "$CONTROL_TOKEN"')
curl -s -m 30 "http://127.0.0.1:4010/usage?n=200" -H "Authorization: Bearer $TOKEN" | python3 -c "
import json,sys
d=json.load(sys.stdin)['summary']
print('switch_backs:', d.get('switch_backs'))
print('by_task_model mimo-trial:', json.dumps(d.get('by_task_model', {}).get('mimo-trial', {}), indent=1))
"
