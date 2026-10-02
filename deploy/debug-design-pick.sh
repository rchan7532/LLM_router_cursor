#!/bin/bash
# Print the exact signature+bar+caps the container computes for the design ask.
cat > /tmp/design_debug2.py <<'PY'
import asyncio

import routing_policy as policy

ASK = ("design the offline sync layer architecture, evaluate CRDT vs event "
       "sourcing trade-offs, and plan the migration step by step")

FLEET = [
    "openai/qwen3-vl-flash", "openai/glm-5.3-flash", "openai/qwen3.8-flash",
    "openai/deepseek-v4.1-flash", "openai/kimi-k2.7-code",
    "openai/qwen3.8-max-0902", "openai/glm-5.3", "openai/claude-haiku-4-5"]

class Ctx:
    raw_messages = [{"role": "user", "content": ASK}]
    structured_messages = raw_messages
    candidate_models = list(FLEET)
    metadata = {}
    signals = {}

sig = policy.build_signature(Ctx().raw_messages)
print("kind:", sig.kind, "| importance:", sig.importance, "| ask_tokens:", sig.ask_tokens)
bar = policy.BASE_BAR.get(sig.kind, 0.6) + policy.SCALE_BAR_STEP * min(
    policy.MAX_SCALE, (sig.ask_tokens or 1) // 400) + policy.IMPORTANCE_OFFSETS[sig.importance]
print("bar:", bar)
for m in FLEET:
    p = policy.PROFILES[m]
    print(f"{m}: design_cap={p.cap.get('design')} blended={policy._blended_cost(p):.3f} "
          f"capable={p.cap.get('design', 0.5) >= bar}")
print("persisted trust (design, glm-5.3):",
      policy.CONTROL.persisted_trust("design", "openai/glm-5.3") if policy.CONTROL else "no-control")
print("persisted trust (design, qmax):",
      policy.CONTROL.persisted_trust("design", "openai/qwen3.8-max-0902") if policy.CONTROL else "no-control")
PY
chmod 644 /tmp/design_debug2.py
docker cp /tmp/design_debug2.py llm-router:/tmp/design_debug2.py
docker exec -w /app llm-router sh -c "PYTHONPATH=/app python3 /tmp/design_debug2.py"