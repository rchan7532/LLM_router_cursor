"""Send three probe asks through the live router: High (spec), Normal, Low.

Reads LITELLM_MASTER_KEY from .env; prints chosen model + importance + cost
per ask, then fetches the decision log tail for confirmation.
"""
import json
import os
import urllib.request

with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
          encoding="utf8") as fh:
    env = {}
    for line in fh:
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip()

TOKEN = "24b8fa195657da6bc97495e5e28051b304bded55446933a5"
MK = env["LITELLM_MASTER_KEY"]
BASE = f"https://router.ifmphk.com/{TOKEN}/v1"

ASKS = [
    ("HIGH  (spec/plan)",
     "Architect the new offline sync layer for our mobile app: write the "
     "specification, evaluate the trade-offs between CRDT and event sourcing, "
     "and plan the migration step by step."),
    ("NORMAL (code fix)",
     "Fix the failing date-parsing test in report_parser.py; the parser "
     "raises ValueError on ISO strings with a timezone suffix."),
    ("LOW   (follow-up)",
     "thanks, looks good"),
]

for label, text in ASKS:
    body = json.dumps({
        "model": "cursor-auto",
        "max_tokens": 40,
        "messages": [{"role": "user", "content": text}],
    }).encode()
    req = urllib.request.Request(
        BASE + "/chat/completions", data=body, method="POST",
        headers={"Authorization": "Bearer " + MK, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            print(f"{label}: HTTP {r.status}")
    except Exception as e:
        print(f"{label}: {e}")
