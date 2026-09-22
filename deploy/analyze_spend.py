#!/usr/bin/env python3
"""Spend audit over the routing decision log (run inside the container).

Reconstructs per-turn cost for pre-importance lines (est_cost_hkd absent)
from est_tokens x model prices, and breaks the log down by time window,
session, and model. Output is an ESTIMATE: output tokens are assumed
(ask_tokens*3, else 800) except where est_cost_hkd exists.
"""
import json
from datetime import datetime, timedelta

PRICES_HKD = {  # (in, out) per 1M tokens
    "openai/qwen3-vl-flash": (0.1256, 1.38),
    "openai/glm-5.3-flash": (0.754, 2.638),
    "openai/qwen3.8-flash": (0.942, 2.9516),
    "openai/deepseek-v4.1-flash": (2.36, 9.42),
    "openai/kimi-k2.7-code": (5.97, 25.12),
    "openai/glm-5.3": (8.79, 27.632),
    "openai/claude-haiku-4-5": (7.85, 39.25),
}
GLM53_IN = 8.79

rows = []
for line in open("/app/logs/routing.jsonl", encoding="utf8"):
    try:
        r = json.loads(line)
    except ValueError:
        continue
    price = PRICES_HKD.get(r["chosen"], (8.79, 27.632))
    if r.get("est_cost_hkd") is not None:
        cost = r["est_cost_hkd"]
        est_out = None
    else:
        est_out = min(4000, max(64, (r.get("ask_tokens") or 0) * 3 or 800))
        cost = (r["est_tokens"] / 1e6) * price[0] + (est_out / 1e6) * price[1]
    rows.append({
        "ts": r["ts"], "chosen": r["chosen"], "kind": r.get("kind"),
        "session": r.get("session", "?")[:8], "reason": r.get("reason"),
        "imp": r.get("importance"), "est_tokens": r.get("est_tokens", 0),
        "cost": cost, "in_cost": (r.get("est_tokens", 0) / 1e6) * price[0],
        "in_cost_glm53": (r.get("est_tokens", 0) / 1e6) * GLM53_IN,
    })

HKT = timedelta(hours=8)
def hkt(ts):
    return datetime.utcfromtimestamp(ts) + HKT

print(f"log span: {hkt(rows[0]['ts']):%H:%M} -> {hkt(rows[-1]['ts']):%H:%M} HKT, {len(rows)} decisions")
win = [r for r in rows if 17 <= hkt(r["ts"]).hour < 18]
after = [r for r in rows if hkt(r["ts"]).hour >= 18]
for label, group in (("17:00-18:00", win), ("18:00-now", after), ("ALL", rows)):
    if not group:
        continue
    total = sum(r["cost"] for r in group)
    in_tot = sum(r["in_cost"] for r in group)
    cf = sum(r["in_cost_glm53"] for r in group)
    print(f"\n== {label}: {len(group)} turns, est {total:.2f} HKD "
          f"(input side {in_tot:.2f}; counterfactual all-glm5.3 input {cf:.2f})")
    per_model = {}
    for r in group:
        per_model.setdefault(r["chosen"].replace("openai/", ""), [0, 0.0])
        per_model[r["chosen"].replace("openai/", "")][0] += 1
        per_model[r["chosen"].replace("openai/", "")][1] += r["cost"]
    for m, (n, c) in sorted(per_model.items(), key=lambda kv: -kv[1][1]):
        print(f"   {m:20s} {n:3d} turns  {c:7.2f} HKD")

print("\n== per session (all log)")
sessions = {}
for r in rows:
    s = sessions.setdefault(r["session"], {"turns": 0, "cost": 0.0, "models": set(),
                                           "max_tok": 0, "first": r["ts"], "kinds": {}})
    s["turns"] += 1
    s["cost"] += r["cost"]
    s["models"].add(r["chosen"].replace("openai/", ""))
    s["max_tok"] = max(s["max_tok"], r["est_tokens"])
    s["kinds"][r["kind"]] = s["kinds"].get(r["kind"], 0) + 1
for sid, s in sorted(sessions.items(), key=lambda kv: -kv[1]["cost"]):
    kinds = ",".join(f"{k}x{n}" for k, n in sorted(s["kinds"].items(), key=lambda kv: -kv[1])[:3])
    print(f"   {sid}  {s['turns']:3d} turns  {s['cost']:7.2f} HKD  peak={s['max_tok']//1000}k tok  "
          f"models={'+'.join(sorted(s['models']))}  kinds={kinds}")

imps = [r for r in rows if r["imp"] is not None]
print(f"\ndecisions with importance field: {len(imps)} "
      f"(low={sum(1 for r in imps if r['imp']==0)}, normal={sum(1 for r in imps if r['imp']==1)}, "
      f"high={sum(1 for r in imps if r['imp']==2)})")
