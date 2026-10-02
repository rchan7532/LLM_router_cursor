#!/bin/bash
# Full learned.json dump: model_trust, bar_biases, phrases with counts.
docker exec llm-router cat /app/logs/learned.json > /tmp/learned_full.json
python3 - <<'PY'
import json
j = json.load(open('/tmp/learned_full.json'))
print('keys:', sorted(j.keys()))
mt = j.get('model_trust', {})
print('\n=== model_trust (design/debug/review rows) ===')
for k, v in sorted(mt.items()):
    if any(kind in k for kind in ('design', 'debug', 'review')):
        print(f"{k}: n={v[0]:.0f} sum={v[1]:.2f} last={v[2]:.0f}")
bb = j.get('bar_bias') or j.get('bar_biases') or {}
print('\n=== bar biases ===')
for k, v in sorted(bb.items()):
    print(f"{k}: {v}")
PY
