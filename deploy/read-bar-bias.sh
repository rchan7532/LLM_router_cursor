#!/bin/bash
# Dump kind_bar_bias + phrase tables (the bias that lowered the live bar).
docker exec llm-router cat /app/logs/learned.json > /tmp/learned_full.json
python3 - <<'PY'
import json
j = json.load(open('/tmp/learned_full.json'))
kbb = j.get('kind_bar_bias', {})
print('=== kind_bar_bias ===')
for k, v in sorted(kbb.items()):
    print(f"{k}: {v}")
pk = j.get('phrase_kinds') or j.get('phrase_kind') or {}
print('\n=== phrase examples (top 15) ===')
items = list(pk.items())[:15] if isinstance(pk, dict) else []
for k, v in items:
    print(f"{k!r}: {v}")
PY
