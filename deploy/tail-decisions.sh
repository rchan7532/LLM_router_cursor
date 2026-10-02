#!/bin/bash
# Inspect the newest cursor-auto decisions: which kind did the classifier
# assign, and where did each turn go?
docker exec llm-router sh -c 'tail -25 /app/logs/routing.jsonl' > /tmp/decisions_tail.json
python3 - <<'PY'
import json
for line in open('/tmp/decisions_tail.json'):
    try:
        d = json.loads(line)
    except Exception:
        continue
    if d.get('reason') in ('disabled',):
        continue
    print(f"{d.get('ts',0):.0f} {str(d.get('chosen','')).split('/')[-1]:18} kind={str(d.get('kind')):9} "
          f"imp={d.get('importance')} ask_tok={d.get('ask_tokens')} reason={d.get('reason'):14} "
          f"task={d.get('task')} client={d.get('client')}")
PY
