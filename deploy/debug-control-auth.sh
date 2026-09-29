#!/bin/bash
cd /opt/llm-router
set -a; source <(sed 's/\r$//' .env); set +a
echo "CONTROL_TOKEN set? [${CONTROL_TOKEN:+yes}] len=${#CONTROL_TOKEN}"
echo "calling control/usage inside container:"
docker exec llm-router sh -c 'wget -qO- --header="Authorization: Bearer ${CONTROL_TOKEN}" http://127.0.0.1:4010/usage?n=2000' | python3 -m json.tool | head -80
