"""Boot the proxy for the smoke test. Used by tests/smoke_proxy.ps1.

Port comes from argv so a smoke run never collides with a router you have
running on 4000.
"""

import sys

port = sys.argv[1] if len(sys.argv) > 1 else "4099"
sys.argv = ["litellm", "--config", "litellm-config.yaml", "--port", port]

from litellm import run_server  # noqa: E402

run_server()
