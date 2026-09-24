"""
MCP server exposing the llm-router control plane to Cursor agents.

Runs LOCALLY on the user's machine via stdio (Cursor starts it, talks
JSON-RPC over stdin/stdout) and forwards to the control service on the VPS
over HTTPS through nginx. The nginx path token only cleans the URL; the
control service authenticates every call itself with its own
CONTROL_TOKEN (distinct from LITELLM_MASTER_KEY, so a leaked MCP config
cannot be replayed against /v1 for completions - review m3). The VPS IP or
router being down degrades to clear error text in tool results; it never
hangs a chat, because every call has a timeout.

Config (~/.cursor/mcp.json), after DEPLOY.md is done:

    "llm-router": {
      "command": "python",
      "args": ["C:\\\\Users\\\\rchan\\\\random\\\\llm-router\\\\mcp_server.py"],
      "env": {
        "ROUTER_CONTROL_URL": "https://router.ifmphk.com/<token>/control",
        "ROUTER_MASTER_KEY": "<CONTROL_TOKEN from the VPS .env>"
      }
    }

For local testing (control_service.py running on 127.0.0.1:4010), leave
ROUTER_MASTER_KEY empty and start the service with CONTROL_TOKEN unset in
read-only mode, or set both sides to the same value.

Tools:
  router_health        - status, active lease, learning state, learner queue,
                         reliability counters, revision
    router_decisions     - recent routing decisions with reasons and scores
  router_learned       - learned phrases, bar biases, trust table, key points
  router_lease         - bounded pin: model + seconds (or requests,
                         converted to a deadline server-side)
  router_lease_clear   - drop the active lease
  router_weights       - live-retune cost/trust/headroom/latency weights
  router_learning      - enable/disable learning, or reset one learned table
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Mapping

BASE_URL = os.environ.get("ROUTER_CONTROL_URL", "http://127.0.0.1:4010").rstrip("/")
MASTER_KEY = os.environ.get("ROUTER_MASTER_KEY", "")
TIMEOUT_SECONDS = float(os.environ.get("ROUTER_CONTROL_TIMEOUT", "10"))

# The fleet, for validating lease models before the round trip. A test pins
# this tuple to routing_policy.PROFILES so the two cannot drift.
FLEET = (
    "openai/qwen3-vl-flash",
    "openai/glm-5.3-flash",
    "openai/qwen3.8-flash",
    "openai/deepseek-v4.1-flash",
    "openai/kimi-k2.7-code",
    "openai/glm-5.3",
    "openai/claude-haiku-4-5",
)


def _request(method: str, path: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    url = f"{BASE_URL}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Content-Type", "application/json")
    if MASTER_KEY:
        request.add_header("Authorization", f"Bearer {MASTER_KEY}")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as error:
        try:
            detail = json.loads(error.read().decode("utf-8")).get("error", "")
        except Exception:  # noqa: BLE001 - best-effort detail extraction
            detail = ""
        return {"error": f"HTTP {error.code}: {detail or error.reason}", "ok": False}
    except urllib.error.URLError as error:
        return {
            "error": f"cannot reach the router control service at {BASE_URL} ({error.reason}). "
            "If the router is on a VPS, check ROUTER_CONTROL_URL and the token; "
            "if testing locally, start control_service.py first.",
            "ok": False,
        }


# ---------------------------------------------------------------------------
# MCP stdio protocol (JSON-RPC 2.0). Minimal: initialize, tools/list, tools/call.
# No sessions, no sampling, no resources - a tool server and nothing else.
# ---------------------------------------------------------------------------

PROTOCOL_VERSION = "2024-11-05"

TOOLS: list[dict[str, Any]] = [
    {
        "name": "router_health",

        "description": "Router control-plane health: learning on/off, active model lease, "
        "weight overrides, learner queue depth and error count, per-model 429/5xx "
        "reliability counters, config revision. Read this first when routing looks wrong.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "router_decisions",
        "description": "Recent routing decisions from the proxy: which model was chosen for "
        "each request, the detected task kind, the reason (fresh / pinned-cache / "
        "pinned-directive), the capability bar, and per-candidate scores. Use to answer "
        "'why did model X serve this' and to spot a stuck or surprising pattern.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "How many decisions to return (1-500, default 50)"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "router_learned",
        "description": "What the router has learned about this user: phrase-to-task-kind "
        "table, per-kind capability bar biases, per-(kind, model) trust, and recent "
        "distilled key points (sanitized, no raw prompts). Empty until the learning "
        "subsystem runs and has enough evidence (8 observations per row).",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "router_lease",
        "description": "Pin the router to one model for a BOUNDED stretch: a seconds "
        "deadline (1-900) or a request count (1-200, converted to a deadline "
        "server-side at ~30s per request). Exactly one bound, not both; the "
        "deadline is evaluated statelessly, so proxy restarts cannot re-arm it. "
        "For a single task, prefer the in-band directive [[use:model]] in the "
        "prompt instead - it wins over any active lease.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": "Model to pin, e.g. openai/kimi-k2.7-code",
                    "enum": list(FLEET),
                },
                "requests": {"type": "integer", "description": "Request-count bound (1-200)"},
                "seconds": {"type": "integer", "description": "Time bound (1-900)"},
            },
            "required": ["model"],
            "additionalProperties": False,
        },
    },
    {
        "name": "router_lease_clear",
        "description": "Drop the active model lease, returning to automatic routing.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "router_weights",
        "description": "Live-retune the routing score weights without a restart. Keys: "
        "cost, trust, headroom, latency, laya, each 0..1 (defaults 0.35/0.30/0.10/0.10/0.0). "
        "Partial updates merge with current values. Pass a key with value null to clear it. "
        "The laya key controls the local shadow scorer's additive influence (0 = telemetry only).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "cost": {"type": "number"},
                "trust": {"type": "number"},
                "headroom": {"type": "number"},
                "latency": {"type": "number"},
                "laya": {"type": "number", "description": "Shadow scorer weight 0..1 (default 0)"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "router_learning",
        "description": "Enable or disable the learning subsystem, or reset learned tables. "
        "reset scope: 'phrases', 'trust', 'bars', or 'all' (all deletes learned.json so "
        "learning starts over). Pass enabled and/or reset, not neither.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "reset": {"type": "string", "enum": ["phrases", "trust", "bars", "all"]},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "router_budget",
        "description": "Set or clear a per-session spend cap in HKD. Once a session's "
        "accumulated estimated cost crosses the cap, cursor-auto is forced to the cheapest "
        "capable model unless you explicitly override with [[high]] or [[use:model]]. "
        "Range 1..10000 HKD. Pass hkd (number) to set, or clear=true to remove the cap.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "hkd": {"type": "number", "description": "Cap in HKD (1..10000)"},
                "clear": {"type": "boolean", "description": "Remove the cap"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "router_kind_alarm_clear",
        "description": "Clear the per-(model, task-kind) spend alarm. When one model "
        "has burned more than the configured threshold (default 20 HKD) on one task kind "
        "inside the current session, the router forces that session to the cheapest capable "
        "model. Call this after reviewing the runaway to resume normal routing.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
]


def _call_tool(name: str, args: Mapping[str, Any]) -> dict[str, Any]:
    if name == "router_health":
        return _request("GET", "/health")
    if name == "router_decisions":
        count = args.get("n", 50)
        if not isinstance(count, int) or not (1 <= count <= 500):
            return {"error": "n must be an integer in 1..500", "ok": False}
        return _request("GET", f"/decisions?n={count}")
    if name == "router_learned":
        return _request("GET", "/learned")
    if name == "router_lease":
        model = args.get("model")
        if model not in FLEET:
            return {"error": f"unknown model {model!r}; fleet: {', '.join(FLEET)}", "ok": False}
        requests, seconds = args.get("requests"), args.get("seconds")
        if (requests is None) == (seconds is None):
            return {"error": "give exactly one bound: requests or seconds", "ok": False}
        payload: dict[str, Any] = {"model": model, "set_by": "cursor-agent"}
        if requests is not None:
            payload["requests"] = requests
        else:
            payload["seconds"] = seconds
        return _request("POST", "/lease", payload)
    if name == "router_lease_clear":
        return _request("POST", "/lease/clear", {})
    if name == "router_weights":
        return _request("POST", "/weights", dict(args))
    if name == "router_learning":
        payload = {key: value for key, value in args.items() if key in ("enabled", "reset")}
        if not payload:
            return {"error": "pass enabled and/or reset", "ok": False}
        return _request("POST", "/learning", payload)
    if name == "router_budget":
        if args.get("clear"):
            return _request("POST", "/budget", {"clear": True})
        if "hkd" in args:
            return _request("POST", "/budget", {"hkd": args["hkd"]})
        return {"error": "pass hkd or clear", "ok": False}
    if name == "router_kind_alarm_clear":
        return _request("POST", "/alarm/clear", {})
    return {"error": f"unknown tool {name!r}", "ok": False}


def _send(message: Mapping[str, Any]) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def _handle(request: Mapping[str, Any]) -> None:
    method = request.get("method", "")
    request_id = request.get("id")
    if method == "initialize":
        _send({
            "jsonrpc": "2.0", "id": request_id, "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "llm-router", "version": "1.0.0"},
            },
        })
        return
    if method == "notifications/initialized":
        return  # notification, no response
    if method == "ping":
        _send({"jsonrpc": "2.0", "id": request_id, "result": {}})
        return
    if method == "tools/list":
        _send({"jsonrpc": "2.0", "id": request_id, "result": {"tools": TOOLS}})
        return
    if method == "tools/call":
        params = request.get("params") or {}
        name = params.get("name", "")
        args = params.get("arguments") or {}
        result = _call_tool(name, args if isinstance(args, dict) else {})
        is_error = bool(result.get("ok") is False or result.get("error"))
        _send({
            "jsonrpc": "2.0", "id": request_id,
            "result": {
                "content": [{"type": "text", "text": json.dumps(result, default=str)}],
                "isError": is_error,
            },
        })
        return
    if request_id is not None:
        _send({"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": f"method not found: {method}"}})


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except ValueError:
            continue
        if isinstance(request, dict):
            _handle(request)


if __name__ == "__main__":
    main()
