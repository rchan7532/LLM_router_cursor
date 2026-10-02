#!/bin/bash
# Live-verify the 2026-10-02 fleet promotion through the PUBLIC endpoint:
#   design ask -> qwen3.8-max-0902, debug ask -> kimi-k2.7-code,
#   plain code ask -> glm-5.3-flash. Checks model + reason in the decision log.
set -e
cd /opt/llm-router
set -a; source <(sed 's/\r$//' .env); set +a
BASE="https://router.ifmphk.com/${MAIN_PATH_TOKEN}/v1/chat/completions"

probe() {
  local label="$1" text="$2"
  echo "=== $label ==="
  curl -s -m 120 "$BASE" \
    -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H "Content-Type: application/json" \
    -H "X-Client-Type: cursor" \
    -d "{\"model\":\"cursor-auto\",\"messages\":[{\"role\":\"user\",\"content\":$(python3 -c "import json,sys;print(json.dumps(sys.argv[1]))" "$text")}],\"max_tokens\":3}" \
    -o /tmp/route_probe.json -w "http=%{http_code}\n"
  sleep 2
  # Newest decision for this fresh probe session: confirm kind + chosen model.
  docker exec llm-router sh -c 'tail -1 /app/logs/routing.jsonl'
  echo
}

probe "design ask (expect qwen3.8-max-0902)" "design the offline sync layer architecture, evaluate CRDT vs event sourcing trade-offs, and plan the migration step by step"
probe "debug ask (expect kimi-k2.7-code)" "why does the parser raise ValueError on ISO strings with a timezone suffix - find the root cause and fix the bug"
probe "plain code edit (expect glm-5.3-flash)" "update src/report_parser.py to add a new date format branch"
