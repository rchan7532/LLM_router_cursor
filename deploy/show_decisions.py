#!/usr/bin/env python3
"""Print chosen/importance/kind/cost for every decision line (run in container)."""
import json

for line in open("/app/logs/routing.jsonl", encoding="utf8"):
    try:
        r = json.loads(line)
    except ValueError:
        continue
    print(f"{r['chosen']:28s} imp={r.get('importance')} kind={r['kind']:8s} "
          f"reason={r['reason']:16s} hkd~{r.get('est_cost_hkd')}")
