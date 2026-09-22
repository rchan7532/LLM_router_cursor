"""
Pre-flight check: does the config, the policy module and litellm agree?

Run before every deploy (and in CI if you add one):

    python tests/validate_config.py

Checks:
  1. every distinct model in the `cursor-auto` group has a profile in the policy
  2. every profile's cost matches the deployment's model_info cost
  3. `router_settings.plugins` resolves to a live RoutingPlugin with async run
  4. the policy routes the three canonical asks the way the design intends
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import os
import sys
import tempfile

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, "litellm-config.yaml")

sys.path.insert(0, ROOT)
# Deterministic pre-flight: point the policy's state at an empty throwaway dir
# so this check never reads (or writes!) leftover learned.json from real runs.
# The container default is /app/logs, which on Windows resolves to C:\app\logs.
os.environ["POLICY_LOG"] = ""
_tmp_state = tempfile.mkdtemp(prefix="validate_config_")
os.environ["POLICY_STATE_DIR"] = _tmp_state
os.environ["LEARNED_PATH"] = os.path.join(_tmp_state, "learned.json")
os.environ["CONTROL_PATH"] = os.path.join(_tmp_state, "control.json")
import routing_policy as policy  # noqa: E402

# The module-level store was built before the env override above; rebuild it.
if policy.default_store is not None:
    policy.CONTROL = policy.default_store()
policy.STATE = policy.PolicyState()

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        failures.append(message)
        print(f"  FAIL {message}")


def main() -> int:
    with open(CONFIG, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    group = [entry for entry in config["model_list"] if entry["model_name"] == "cursor-auto"]
    distinct = sorted({entry["litellm_params"]["model"] for entry in group})

    print("1. model profiles")
    check(len(distinct) >= 2, f"cursor-auto has {len(distinct)} distinct models")
    for model in distinct:
        check(model in policy.PROFILES, f"profile exists for {model}")

    print("2. declared costs agree")
    for entry in group:
        model = entry["litellm_params"]["model"]
        info = entry.get("model_info", {})
        profile = policy.PROFILES.get(model)
        if profile is None or "input_cost_per_token" not in info:
            continue
        check(
            abs(info["input_cost_per_token"] - profile.cost_in / 1_000_000) < 1e-12,
            f"input cost matches for {model}",
        )
        check(
            abs(info["output_cost_per_token"] - profile.cost_out / 1_000_000) < 1e-12,
            f"output cost matches for {model}",
        )
        check(
            info.get("supports_vision", False) == profile.vision,
            f"vision flag matches for {model}",
        )

    print("3. plugin resolves")
    try:
        from litellm.proxy.types_utils.utils import get_instance_fn
        from litellm.types.router import RoutingPlugin

        configured = config["router_settings"]["plugins"]
        check(bool(configured), f"router_settings.plugins = {configured}")
        # The proxy loads plugin modules without registering them in sys.modules,
        # so pop our own registration first or this check would mask a load-time
        # failure that only shows up when the proxy boots.
        saved = sys.modules.pop("routing_policy", None)
        try:
            resolved = get_instance_fn(value=configured[0], config_file_path=CONFIG)
        finally:
            if saved is not None:
                sys.modules["routing_policy"] = saved
        check(isinstance(resolved, RoutingPlugin), f"{configured[0]} implements RoutingPlugin")
        check(not isinstance(resolved, type), "plugin path names an instance, not a class")
        check(inspect.iscoroutinefunction(resolved.run), "run() is async")
    except ImportError as error:
        print(f"  skip litellm not importable here ({error}); the proxy still checks at startup")

    print("4. routing behaviour")

    class Ctx:
        def __init__(self, text: str) -> None:
            self.raw_messages = [{"role": "user", "content": text}]
            self.structured_messages = self.raw_messages
            self.candidate_models = list(distinct)
            self.metadata = {}
            self.signals = {}

    def route(text: str) -> tuple[str, str]:
        context = Ctx(text)
        asyncio.run(policy.CursorAutoPolicy().run(context))
        decision = context.signals["policy"]
        return decision["kind"], decision["chosen"]

    kind, chosen = route("fix the failing test in report_parser.py")
    check(kind == "debug" and chosen == "openai/kimi-k2.7-code", f"debug -> {chosen}")

    kind, chosen = route("write a script that renames every file across the repo")
    check(
        kind == "bulk" and policy.PROFILES[chosen].cost_in <= 0.05,
        f"bulk -> {chosen} (cost_in {policy.PROFILES[chosen].cost_in})",
    )

    kind, chosen = route("design the offline sync layer and its trade-offs")
    check(kind == "design" and policy.PROFILES[chosen].cap["design"] >= 0.85, f"design -> {chosen}")

    print("---")
    if failures:
        print(f"{len(failures)} check(s) failed")
        return 1
    print("config, profiles and policy agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
