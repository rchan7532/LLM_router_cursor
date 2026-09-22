"""Drive mcp_server.py through real JSON-RPC over stdio against a live
control_service.py. Not part of the unit suite (it spawns processes); run
explicitly:

    python tests/test_mcp_e2e.py

Asserts the initialize handshake, tools/list, a lease round trip through
router_lease, health reflecting it, and a clean error when the control
service is down (the degraded-mode contract for Cursor chats).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 4399


def main() -> int:
    state_dir = tempfile.mkdtemp(prefix="router-mcp-e2e-")

    service_env = dict(os.environ)
    service_env.update({
        "POLICY_STATE_DIR": state_dir,
        "CONTROL_PORT": str(PORT),
        "POLICY_LOG": os.path.join(state_dir, "routing.jsonl"),
        "CONTROL_TOKEN": "e2e-control-token",  # the service authenticates every call (m3)
        "PYTHONIOENCODING": "utf-8",
    })
    service = subprocess.Popen(
        [sys.executable, "-X", "utf8", os.path.join(ROOT, "control_service.py")],
        env=service_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    time.sleep(1.5)

    mcp_env = dict(os.environ)
    mcp_env["ROUTER_CONTROL_URL"] = f"http://127.0.0.1:{PORT}"
    mcp_env["ROUTER_MASTER_KEY"] = "e2e-control-token"
    mcp_env["CONTROL_TOKEN"] = "e2e-control-token"  # service side must match
    mcp_env["PYTHONIOENCODING"] = "utf-8"
    mcp = subprocess.Popen(
        [sys.executable, "-X", "utf8", os.path.join(ROOT, "mcp_server.py")],
        env=mcp_env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, encoding="utf-8",
    )

    failures: list[str] = []

    def send(message):
        mcp.stdin.write(json.dumps(message) + "\n")
        mcp.stdin.flush()

    def receive():
        line = mcp.stdout.readline()
        if not line:
            raise AssertionError("mcp server closed stdout")
        return json.loads(line)

    def call_tool(name, arguments):
        send({"jsonrpc": "2.0", "id": next(call_tool.counter), "method": "tools/call",
              "params": {"name": name, "arguments": arguments}})
        response = receive()
        result = response.get("result", {})
        text = result.get("content", [{}])[0].get("text", "{}")
        return json.loads(text), result.get("isError", False)

    call_tool.counter = iter(range(100, 999))

    try:
        # 1. initialize handshake
        send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                         "clientInfo": {"name": "test", "version": "0"}}})
        init = receive()
        assert init["result"]["serverInfo"]["name"] == "llm-router", init
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})

        # 2. tools/list
        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        listed = receive()
        names = {tool["name"] for tool in listed["result"]["tools"]}
        expected = {"router_health", "router_decisions", "router_learned",
                    "router_lease", "router_lease_clear", "router_weights",
                    "router_learning", "router_budget"}
        if names != expected:
            failures.append(f"tools/list mismatch: {names ^ expected}")

        # 3. health (no lease yet)
        health, _ = call_tool("router_health", {})
        if health.get("status") != "ok":
            failures.append(f"health not ok: {health}")

        # 4. lease round trip
        lease, is_error = call_tool("router_lease",
                                    {"model": "openai/kimi-k2.7-code", "requests": 5})
        if is_error or lease.get("lease", {}).get("model") != "openai/kimi-k2.7-code":
            failures.append(f"lease failed: {lease}")

        # 5. health shows the lease
        health, _ = call_tool("router_health", {})
        if health.get("lease", {}).get("model") != "openai/kimi-k2.7-code":
            failures.append(f"health missing lease: {health}")

        # 6. lease validation: both bounds
        bad, is_error = call_tool("router_lease",
                                  {"model": "openai/glm-5.3", "requests": 5, "seconds": 60})
        if not is_error:
            failures.append(f"both-bounds lease did not error: {bad}")

        # 7. clear
        cleared, is_error = call_tool("router_lease_clear", {})
        if is_error or cleared.get("lease") is not None:
            failures.append(f"clear failed: {cleared}")

        # 8. weights partial update
        weights, is_error = call_tool("router_weights", {"cost": 0.5})
        if is_error or weights.get("weights", {}).get("cost") != 0.5:
            failures.append(f"weights failed: {weights}")

        # 9. unknown tool
        unknown, is_error = call_tool("router_nope", {})
        if not is_error:
            failures.append(f"unknown tool did not error: {unknown}")
    finally:
        mcp.stdin.close()
        mcp.wait(timeout=10)
        service.terminate()
        service.wait(timeout=10)

    # 10. degraded mode: control service DOWN, MCP must return clean error text
    mcp_env["ROUTER_CONTROL_URL"] = f"http://127.0.0.1:{PORT}"  # nothing listening now
    mcp2 = subprocess.Popen(
        [sys.executable, "-X", "utf8", os.path.join(ROOT, "mcp_server.py")],
        env=mcp_env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, encoding="utf-8",
    )
    try:
        mcp2.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                     "params": {"protocolVersion": "2024-11-05",
                                                "capabilities": {},
                                                "clientInfo": {"name": "t", "version": "0"}}}) + "\n")
        mcp2.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                                     "params": {"name": "router_health", "arguments": {}}}) + "\n")
        mcp2.stdin.flush()
        mcp2.stdin.close()
        lines = [json.loads(line) for line in mcp2.stdout if line.strip()]
        tool_response = next(item for item in lines if item.get("id") == 2)
        payload = json.loads(tool_response["result"]["content"][0]["text"])
        if "error" not in payload or "cannot reach" not in payload["error"]:
            failures.append(f"degraded mode did not produce clean error: {payload}")
    finally:
        mcp2.wait(timeout=10)

    if failures:
        print("FAIL")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("mcp e2e: all green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
