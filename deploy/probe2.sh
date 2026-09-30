#!/usr/bin/env bash
# Path-token reachability probe: /v1/models and /control/health through nginx.
cd /opt/llm-router
set -a; source <(sed 's/\r$//' .env); set +a
TOKEN=$MAIN_PATH_TOKEN
echo "tokened v1:"
curl -s -m 8 "https://router.ifmphk.com/$TOKEN/v1/models" -o /dev/null -w "%{http_code}\n"
echo "tokened control:"
curl -s -m 8 "https://router.ifmphk.com/$TOKEN/control/health" -o /dev/null -w "%{http_code}\n"
