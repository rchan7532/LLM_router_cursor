#!/bin/bash
cd /opt/llm-router
TOKEN=$(docker exec llm-router-control sh -c 'echo -n "$CONTROL_TOKEN"')
curl -s -m 30 "http://127.0.0.1:4010/receipts?n=20000" -H "Authorization: Bearer $TOKEN" > /tmp/receipts.json
python3 << 'EOF'
import json

with open("/tmp/receipts.json") as handle:
    rows = [r for r in json.load(handle)["receipts"] if r.get("task")]

by = {}
for r in rows:
    by.setdefault(r.get("model"), set()).add(r.get("session"))
for model, sessions in sorted(by.items()):
    print(model, "sessions:", len(sessions), sorted(s for s in sessions if s)[:5])
EOF
