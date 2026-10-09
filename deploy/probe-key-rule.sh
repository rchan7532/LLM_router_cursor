#!/bin/bash
# Verify the provider's new key rule: GLM + deepseek must be served by BASIC.
# Probe 1-token completions per (model, key). 200 = usable, 4xx = not served.
set -e
cd /opt/llm-router
set -a; source <(sed 's/\r$//' .env); set +a

probe() {
  local model="$1" key="$2" base="$3" label="$4"
  local code
  code=$(curl -s -m 30 -o /tmp/probe_out.json -w '%{http_code}' \
    "${base}/chat/completions" \
    -H "Authorization: Bearer ${key}" -H "Content-Type: application/json" \
    -d "{\"model\":\"${model}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}")
  local detail
  detail=$(head -c 120 /tmp/probe_out.json | tr -d '\n')
  printf "%-22s [%s] http=%s  %s\n" "${model}" "${label}" "${code}" "${detail}"
}

B=${ROUTER_KEY_BASIC}
BB=${ROUTER_KEY_BASIC_BASE}
D=${ROUTER_KEY_DEFAULT}
DB=${ROUTER_KEY_DEFAULT_BASE}

echo "=== BASIC key (must serve all GLM + deepseek per new rule) ==="
probe glm-5.3-flash   "$B" "$BB" BASIC
probe glm-5.3         "$B" "$BB" BASIC
probe glm-5.3-flashx  "$B" "$BB" BASIC
probe deepseek-v4.1-flash "$B" "$BB" BASIC

echo "=== DEFAULT key (control: what changes after we move) ==="
probe glm-5.3-flash   "$D" "$DB" DEFAULT
probe glm-5.3-flashx  "$D" "$DB" DEFAULT
