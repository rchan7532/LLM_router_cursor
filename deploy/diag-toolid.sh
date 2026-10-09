#!/bin/bash
# Find the failing request and its routing decision + reliability entries.
RID="c39ae142-7152-4151-9752-6f5ac05eff17"
echo "=== routing.jsonl recent (20 lines) ==="
docker exec llm-router sh -c "tail -20 /app/logs/routing.jsonl" > /tmp/rj.json
python3 - <<'PY'
import json
for line in open('/tmp/rj.json'):
    try:
        d = json.loads(line)
    except Exception:
        continue
    print(f"{d.get('ts',0):.0f} {str(d.get('chosen','')).split('/')[-1]:18} kind={str(d.get('kind')):9} "
          f"reason={d.get('reason'):14} task={d.get('task')} session={str(d.get('session',''))[:10]}")
PY
echo "=== reliability.jsonl tail ==="
docker exec llm-router sh -c "tail -8 /app/logs/reliability.jsonl"
echo "=== search request id in logs ==="
docker exec llm-router sh -c "grep -l c39ae142 /app/logs/*.jsonl 2>/dev/null; grep c39ae142 /app/logs/routing.jsonl | tail -2"
echo "=== litellm proxy logs tail (tool errors) ==="
docker logs llm-router --since 30m 2>&1 | grep -iE "tool_call_id|BadRequest|c39ae142" | tail -8
