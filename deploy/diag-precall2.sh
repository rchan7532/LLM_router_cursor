#!/bin/bash
# Find every async_pre_call_hook call site in the pinned litellm package.
docker exec -i llm-router python - <<'PY'
import os, re, litellm

root = os.path.dirname(litellm.__file__)
for dirpath, _dirs, files in os.walk(root):
    for name in files:
        if not name.endswith(".py"):
            continue
        path = os.path.join(dirpath, name)
        try:
            src = open(path, encoding="utf-8").read()
        except Exception:
            continue
        for m in re.finditer(r"async_pre_call_hook", src):
            start = max(0, m.start() - 350)
            print(f"### {os.path.relpath(path, root)} @{src[:m.start()].count(chr(10)) + 1}")
            print(src[start:m.start() + 350])
            print("-----")
PY