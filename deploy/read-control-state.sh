#!/bin/bash
# Confirm the learned bar bias source: dump control.json's learning tables.
docker exec llm-router sh -c "ls /control/ 2>/dev/null; cat /control/control.json 2>/dev/null" > /tmp/control_dump.txt
python3 - <<'PY'
import json, re
raw = open('/tmp/control_dump.txt').read()
# strip the ls output lines before the first '{'
start = raw.find('{')
if start < 0:
    print('no control.json found'); raise SystemExit
j = json.loads(raw[start:])
print('top-level keys:', sorted(j.keys()))
learn = j.get('learning') or {}
print('learning keys:', sorted(learn.keys()) if isinstance(learn, dict) else type(learn))
bars = learn.get('bars') if isinstance(learn, dict) else None
print('bars:', json.dumps(bars, indent=1)[:600] if bars else None)
trust = learn.get('trust') if isinstance(learn, dict) else None
if trust:
    print('trust sample:', json.dumps(dict(list(trust.items())[:8]), indent=1))
PY
