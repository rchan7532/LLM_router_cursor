#!/bin/bash
# Probe kimi-k3 and qwen3.8-max availability and list available models per key.
set -e
cd /opt/llm-router
set -a; source <(sed 's/\r$//' .env); set +a

echo "=== available models (default key) ==="
curl -s -m 30 "${ROUTER_KEY_DEFAULT_BASE}/v1/models" -H "Authorization: Bearer ${ROUTER_KEY_DEFAULT}" -o /tmp/models_default.json -w "http=%{http_code} bytes=%{size_download}\n"
python3 - <<'PY'
import json
try:
    j = json.load(open('/tmp/models_default.json'))
except Exception as e:
    print('parse failed:', e)
    print(open('/tmp/models_default.json').read()[:300])
else:
    for m in j.get('data', []):
        mid = m['id'].lower()
        if any(k in mid for k in ('kimi', 'qwen', 'glm', 'claude', 'mimo')):
            print(m['id'])
PY

echo ""
echo "=== available models (basic key) ==="
curl -s -m 30 "${ROUTER_KEY_BASIC_BASE}/v1/models" -H "Authorization: Bearer ${ROUTER_KEY_BASIC}" -o /tmp/models_basic.json -w "http=%{http_code} bytes=%{size_download}\n"
python3 - <<'PY'
import json
try:
    j = json.load(open('/tmp/models_basic.json'))
except Exception as e:
    print('parse failed:', e)
    print(open('/tmp/models_basic.json').read()[:300])
else:
    for m in j.get('data', []):
        mid = m['id'].lower()
        if any(k in mid for k in ('kimi', 'qwen', 'glm', 'claude', 'mimo')):
            print(m['id'])
PY

echo ""
echo "=== probe kimi-k3 (default key) ==="
curl -s -m 60 "${ROUTER_KEY_DEFAULT_BASE}/v1/chat/completions" \
  -H "Authorization: Bearer ${ROUTER_KEY_DEFAULT}" -H "Content-Type: application/json" \
  -d '{"model":"kimi-k3","messages":[{"role":"user","content":"hi"}],"max_tokens":1}' -o /tmp/probe_k3.json -w "http=%{http_code}\n"
head -c 300 /tmp/probe_k3.json; echo

echo "=== probe qwen3.8-max (basic key) ==="
curl -s -m 60 "${ROUTER_KEY_BASIC_BASE}/v1/chat/completions" \
  -H "Authorization: Bearer ${ROUTER_KEY_BASIC}" -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-max","messages":[{"role":"user","content":"hi"}],"max_tokens":1}' -o /tmp/probe_qmax.json -w "http=%{http_code}\n"
head -c 300 /tmp/probe_qmax.json; echo
