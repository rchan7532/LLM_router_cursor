#!/bin/bash
# Inspect litellm's async_pre_call_hook contract inside the pinned image.
docker exec -i llm-router python - <<'PY'
import inspect
from litellm.integrations.custom_logger import CustomLogger

print("== signature ==")
print(inspect.signature(CustomLogger.async_pre_call_hook))
print("== docstring ==")
print((CustomLogger.async_pre_call_hook.__doc__ or "")[:600])
try:
    print("== source ==")
    print(inspect.getsource(CustomLogger.async_pre_call_hook))
except Exception as exc:
    print("no source:", exc)

import litellm.utils as u
src = inspect.getsource(u)
import re
hits = [m.start() for m in re.finditer(r"async_pre_call_hook", src)]
print("== call sites in litellm.utils:", len(hits), "==")
for i in hits[:8]:
    print(src[max(0, i - 500):i + 250])
    print("-----")
PY