#!/bin/bash
# Dump full /models metadata for kimi-k3 and qwen3.8-max(-0902).
set -e
cd /opt/llm-router
set -a; source <(sed 's/\r$//' .env); set +a
python3 - <<'PY'
import json, urllib.request, os

def fetch(which):
    base = os.environ[f"ROUTER_KEY_{which}_BASE"]
    key = os.environ[f"ROUTER_KEY_{which}"]
    req = urllib.request.Request(base + "/models", headers={
        "Authorization": "Bearer " + key, "User-Agent": "litellm/1.101.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

for which in ("DEFAULT", "BASIC"):
    try:
        j = fetch(which)
    except Exception as e:
        print(which, "fetch failed:", e)
        continue
    for m in j.get("data", []):
        if m.get("id") in ("kimi-k3", "qwen3.8-max", "qwen3.8-max-0902", "kimi/kimi-k3"):
            print(f"--- {which} {m['id']} ---")
            print(json.dumps(m, indent=1, ensure_ascii=False)[:900])
PY
