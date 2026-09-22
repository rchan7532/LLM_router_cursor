#!/usr/bin/env python3
"""One-shot VPS probe: control health + nginx path routing + model list.

scp to the VPS and run:  python3 deploy_probe.py
Reads CONTROL_TOKEN and LITELLM_MASTER_KEY from /opt/llm-router/.env.
"""
import json
import urllib.request
import urllib.error

def env_of(name):
    with open("/opt/llm-router/.env") as fh:
        for line in fh:
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip()
    return ""

TOKEN = "24b8fa195657da6bc97495e5e28051b304bded55446933a5"
CT = env_of("CONTROL_TOKEN")
MK = env_of("LITELLM_MASTER_KEY")

def get(url, key=None):
    req = urllib.request.Request(url)
    if key:
        req.add_header("Authorization", "Bearer " + key)
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            return r.status, r.read().decode()[:220]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:220]
    except Exception as e:
        return None, repr(e)[:220]

print("== direct (localhost) ==")
print("control /health      :", get("http://127.0.0.1:4010/health"))
print("control /health auth :", get("http://127.0.0.1:4010/health", CT))

print("== through nginx (public URL, localhost) ==")
print("tokenless            :", get("https://router.ifmphk.com/v1/models"))
print("tokened /v1/models   :", get("https://router.ifmphk.com/" + TOKEN + "/v1/models", MK)[0])
print("tokened control      :", get("https://router.ifmphk.com/" + TOKEN + "/control/health", CT))
print("tokened control noauth:", get("https://router.ifmphk.com/" + TOKEN + "/control/health"))
