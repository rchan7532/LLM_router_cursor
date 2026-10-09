#!/bin/bash
# Show process_pre_call_hook_response source (dict-return semantics).
docker exec -i llm-router python - <<'PY'
import inspect
import litellm.proxy.utils as pu

src = inspect.getsource(pu)
idx = src.find("async def process_pre_call_hook_response")
print(src[idx:idx + 1600])
PY