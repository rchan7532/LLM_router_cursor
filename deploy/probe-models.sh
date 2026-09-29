#!/bin/bash
cd /opt/llm-router
set -a; source <(sed 's/\r$//' .env); set +a
echo "=== MiMo model ids ==="
curl -s -m 20 "$MIMO_BASE/models" -H "Authorization: Bearer $MIMO_KEY" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print([m.get('id') for m in data.get('data', [])])
"
echo "=== chat probe: exact id test ==="
for MODEL in "MiMo-V2.6-Pro" "mimo-v2.6-pro"; do
  echo "--- $MODEL"
  curl -s -m 30 "$MIMO_BASE/chat/completions" -H "Authorization: Bearer $MIMO_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":3}" | head -c 250
  echo
done
echo "=== FlashX probe via ToAI (1 token) ==="
curl -s -m 30 "$ROUTER_KEY_DEFAULT_BASE/chat/completions" -H "Authorization: Bearer $ROUTER_KEY_DEFAULT" \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5.3-flashx","messages":[{"role":"user","content":"hi"}],"max_tokens":3}' | head -c 250
echo
