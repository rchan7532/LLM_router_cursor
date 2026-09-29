#!/bin/bash
# Deploy measurement upgrade: task-tag on escape hatches, latency receipts,
# per-model task rollup. Syncs are done before this runs (scp).
set -e
cd /opt/llm-router
echo "=== restart proxy + control ==="
docker restart llm-router llm-router-control
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
  S=$(docker inspect llm-router --format '{{.State.Health.Status}}' 2>/dev/null)
  T=$(docker inspect llm-router-control --format '{{.State.Health.Status}}' 2>/dev/null)
  echo "attempt $i: router=$S control=$T"
  [ "$S" = healthy ] && [ "$T" = healthy ] && break
  sleep 8
done
echo "=== tagged escape-hatch call (expect task=mimo-trial in receipt) ==="
set -a; source <(sed 's/\r$//' .env); set +a
curl -s -m 60 https://router.ifmphk.com/24b8fa195657da6bc97495e5e28051b304bded55446933a5/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H "Content-Type: application/json" \
  -H "X-Client-Type: cursor" \
  -d '{"model":"mimo-v2.6-pro","messages":[{"role":"user","content":"say hi [[task:mimo-trial]]"}],"max_tokens":3}' | head -c 150
echo
sleep 8
echo "=== latest receipts (task + duration fields) ==="
docker exec llm-router sh -c 'tail -2 /app/logs/routing_usage.jsonl'
echo "=== control /usage summary (by_model + by_task_model) ==="
TOKEN=$(docker exec llm-router-control sh -c 'echo -n "$CONTROL_TOKEN"')
curl -s -m 30 "http://127.0.0.1:4010/usage?n=200" -H "Authorization: Bearer $TOKEN" | python3 -c "
import json,sys
d=json.load(sys.stdin)['summary']
for m,v in sorted(d['by_model'].items()):
    avg = (v.get('duration_s', 0) / v['duration_calls']) if v.get('duration_calls') else None
    print(m, 'calls', v['calls'], 'real', v['real_hkd'],
          'prompt', v.get('prompt_tokens', 0), 'cached', v.get('cached_tokens', 0),
          'avg_s', round(avg, 2) if avg else None)
print('--- by_task_model ---')
print(json.dumps(d.get('by_task_model', {}), indent=1))
"
echo "=== fleet group unaffected (cursor-auto) ==="
curl -s https://router.ifmphk.com/24b8fa195657da6bc97495e5e28051b304bded55446933a5/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H "Content-Type: application/json" \
  -d '{"model":"cursor-auto","messages":[{"role":"user","content":"say ok"}],"max_tokens":4}' -o /dev/null -w "cursor-auto http=%{http_code}\n"
