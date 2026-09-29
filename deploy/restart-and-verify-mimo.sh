#!/bin/bash
# Restart llm-router so it picks up renamed config, wait healthy, verify.
docker restart llm-router
sleep 10
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
  S=$(docker inspect llm-router --format '{{.State.Health.Status}}' 2>/dev/null)
  echo "attempt $i: $S"
  [ "$S" = healthy ] && break
  sleep 8
done
bash /tmp/v.sh
