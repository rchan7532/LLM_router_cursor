#!/bin/bash
# Hash the deployed runtime files so local vs VPS can be diffed.
cd /opt/llm-router
sha256sum routing_policy.py control_store.py usage_state.py usage_hook.py mcp_server.py litellm-config.yaml control_service.py | awk '{print $1, $2}'
