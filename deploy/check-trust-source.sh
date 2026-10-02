#!/bin/bash
# Where does persisted trust live, and why did the live design pick show 3
# candidates when the bar math admits only qmax? Compare the live decision
# log entry's importance/est_tokens against the probe's.
echo "=== last design decisions: importance + est_tokens + candidates ==="
docker exec llm-router sh -c 'grep "\"kind\": \"design\"" /app/logs/routing.jsonl | tail -3'
echo
echo "=== trust file candidates inside container ==="
docker exec llm-router sh -c "ls -la /app/logs/learned.json; head -c 400 /app/logs/learned.json"
