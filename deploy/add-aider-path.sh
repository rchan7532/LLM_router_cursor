#!/bin/bash
# Add an aider entry path to nginx: same proxying as the main token path,
# but stamps X-Client-Type: aider so receipts attribute the harness by
# route, not by User-Agent guesswork.
set -e
CONF=/etc/nginx/sites-available/router.ifmphk.com

# Generate a fresh path token (openssl, hex like the existing one)
AIDER_TOKEN=$(openssl rand -hex 24)
echo "new aider path token: $AIDER_TOKEN"

# Idempotency: bail if already present
if grep -q "aider entry path" "$CONF"; then
  echo "already configured; nothing to do"
  exit 0
fi

# Insert BEFORE the existing "location ~ ^/<old-token>(" block so the more
# specific aider route is matched in file order (nginx regex locations run
# in order; first match wins).
python3 - "$CONF" "$AIDER_TOKEN" <<'PY'
import sys
conf, token = sys.argv[1], sys.argv[2]
with open(conf, encoding="utf-8") as f:
    lines = f.readlines()
marker = None
for i, line in enumerate(lines):
    if "location ~ ^/" in line and "upstream_path" in line:
        marker = i
        break
if marker is None:
    raise SystemExit("could not find the main upstream location block")
block = f"""    # aider entry path (2026-09-28): stamps the harness so router receipts
    # attribute spend by route, not User-Agent. Keep secret like the other.
    location ~ ^/{token}(?<upstream_path>/.*)$ {{
        proxy_set_header X-Client-Type aider;
        proxy_pass http://127.0.0.1:4000$upstream_path$is_args$args;
    }}

"""
lines.insert(marker, block)
with open(conf, "w", encoding="utf-8") as f:
    f.writelines(lines)
print("inserted aider location block before main token route")
PY

nginx -t
systemctl reload nginx
echo "OK - aider endpoint: https://router.ifmphk.com/$AIDER_TOKEN/v1"
