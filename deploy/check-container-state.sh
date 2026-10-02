#!/bin/bash
# Decisive state check: what code/config does the container actually run?
echo "=== compose mounts ==="
docker inspect llm-router --format '{{range .Mounts}}{{.Source}} -> {{.Destination}}{{"\n"}}{{end}}' | head -10
echo "=== qmax profile inside container (cap design + fleet entry) ==="
docker exec llm-router sh -c "grep -n 'qwen3.8-max-0902' /app/routing_policy.py | head -5"
echo "=== design cap value in container ==="
docker exec llm-router sh -c "grep -A3 'openai/qwen3.8-max-0902\": Profile' /app/routing_policy.py | head -6"
echo "=== cursor-auto qmax deployment in container config ==="
docker exec llm-router sh -c "grep -B2 -A2 'qwen3.8-max-0902' /app/litellm-config.yaml | head -30"
echo "=== gate field in local vs container log schema ==="
docker exec llm-router sh -c "grep -c '\"gate\"' /app/routing_policy.py"
grep -c '"gate"' C:/Users/rchan/random/llm-router/routing_policy.py 2>/dev/null || echo "(local check skipped)"
