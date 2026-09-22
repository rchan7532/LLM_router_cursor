#!/usr/bin/env python3
"""Read LiteLLM container log from stdin, count POST /v1/chat/completions per HKT hour."""
import sys, re
from datetime import datetime, timezone, timedelta

h = {}
for line in sys.stdin:
    if "POST /v1/chat/completions" not in line:
        continue
    m = re.search(r"([0-9]{2}:[0-9]{2}:[0-9]{2}) -", line)
    if not m:
        continue
    dt = datetime.strptime(m.group(1), "%H:%M:%S").replace(tzinfo=timezone.utc)
    hkt = dt + timedelta(hours=8)
    h[hkt.hour] = h.get(hkt.hour, 0) + 1

for hour in sorted(h):
    print(f"{hour:02d}:00-{(hour+1)%24:02d}:00 HKT  {h[hour]:3d} requests")
