#!/usr/bin/env bash
TOKEN=24b8fa195657da6bc97495e5e28051b304bded55446933a5
echo "tokened v1:"
curl -s -m 8 "https://router.ifmphk.com/$TOKEN/v1/models" -o /dev/null -w "%{http_code}\n"
echo "tokened control:"
curl -s -m 8 "https://router.ifmphk.com/$TOKEN/control/health" -o /dev/null -w "%{http_code}\n"
