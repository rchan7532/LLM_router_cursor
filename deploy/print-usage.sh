#!/bin/bash
# Print the receipt-based usage summary plus a trial-window cut.
# Usage: print-usage.sh [n] [window_start_iso]
set -e
TOKEN=$(docker exec llm-router-control sh -c 'echo -n "$CONTROL_TOKEN"')
curl -s -m 60 "http://127.0.0.1:4010/usage?n=${1:-20000}" -H "Authorization: Bearer $TOKEN" > /tmp/usage.json
# Full receipt history straight from the mounted log (control /receipts caps at 500).
docker exec llm-router cat /app/logs/routing_usage.jsonl > /tmp/receipts.json
START=$(python3 -c "import sys,datetime;print(datetime.datetime.fromisoformat(sys.argv[1]).timestamp())" "$2")
python3 /opt/llm-router/deploy/print-usage.py /tmp/usage.json /tmp/receipts.json "$START"
