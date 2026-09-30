import json, sys
from datetime import datetime, timezone

with open(sys.argv[1]) as f:
    d = json.load(f)['summary']
print('=== by_model ===')
for m, r in d['by_model'].items():
    print(f"{m}: real={r['real_hkd']:.3f} est={r['est_hkd']:.3f} calls={r['calls']} p={r.get('prompt_tokens',0)} c={r.get('completion_tokens',0)} cache={r.get('cached_tokens',0)} dur={r.get('duration_s')} dcalls={r.get('duration_calls')}")
print('\n=== by_task_model ===')
for t, models in d.get('by_task_model', {}).items():
    for m, r in models.items():
        print(f"{t} | {m}: real={r['real_hkd']:.4f} calls={r['calls']}")
print('\n=== by_task ===')
for t, clients in d.get('by_task', {}).items():
    if not isinstance(clients, dict):
        continue
    for c, r in clients.items():
        print(f"{t} [{c}]: real={r['real_hkd']:.4f} calls={r['calls']}")
print('\n=== task_redo ===', d.get('task_redo'))
print('=== switch_backs ===', d.get('switch_backs'))

# Trial-window cut from raw receipts (JSONL), if provided.
if len(sys.argv) > 2:
    receipts = []
    with open(sys.argv[2]) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    receipts.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    start = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    win = [r for r in receipts if float(r.get('ts', 0) or 0) >= start]

    def agg(rows):
        p = sum(r.get('prompt_tokens', 0) or 0 for r in rows)
        c = sum(r.get('completion_tokens', 0) or 0 for r in rows)
        ca = sum(r.get('cached_tokens', 0) or 0 for r in rows)
        real = sum(r.get('real_cost_hkd', 0) or 0 for r in rows)
        durs = sorted(r['duration_s'] for r in rows if isinstance(r.get('duration_s'), (int, float)))
        calls = len(rows)
        avg = sum(durs) / len(durs) if durs else None
        p50 = durs[len(durs) // 2] if durs else None
        p90 = durs[int(len(durs) * 0.9)] if durs else None
        return dict(calls=calls, real=round(real, 3), p=p, c=c, cache=ca,
                    cache_ratio=round(ca / p, 3) if p else None,
                    avg_s=round(avg, 1) if avg else None,
                    p50_s=round(p50, 1) if p50 else None,
                    p90_s=round(p90, 1) if p90 else None,
                    dur_calls=len(durs))

    print(f"\n=== trial window (ts>={start:.0f}, {datetime.fromtimestamp(start, tz=timezone.utc).isoformat()}) ===")
    by_model = {}
    for r in win:
        by_model.setdefault(r.get('model'), []).append(r)
    for m, rows in sorted(by_model.items()):
        a = agg(rows)
        print(f"{m}: {a}")
    sb = [r for r in win if r.get('switch_back_from')]
    print(f"switch-back receipts in window: {len(sb)} -> {sorted({r['switch_back_from'] for r in sb})}")

    # mimo spend by session: find the outlier (code-package recheck loop).
    print('\n=== mimo sessions in window (top 12 by real) ===')
    sess = {}
    for r in win:
        if r.get('model') == 'openai/mimo-v2.6-pro':
            key = r.get('session') or 'none'
            s = sess.setdefault(key, dict(calls=0, real=0.0, c=0, p=0, dur=0.0, task=r.get('task'), ts0=r.get('ts')))
            s['calls'] += 1
            s['real'] += r.get('real_cost_hkd', 0) or 0
            s['c'] += r.get('completion_tokens', 0) or 0
            s['p'] += r.get('prompt_tokens', 0) or 0
            if isinstance(r.get('duration_s'), (int, float)):
                s['dur'] += r['duration_s']
    for key, s in sorted(sess.items(), key=lambda kv: -kv[1]['real'])[:12]:
        t0 = datetime.fromtimestamp(s['ts0'], tz=timezone.utc).strftime('%m-%d %H:%M') if s.get('ts0') else '?'
        print(f"{key[:10]} start={t0} task={s['task']} calls={s['calls']} real={s['real']:.3f} p={s['p']} c={s['c']} dur={s['dur']:.0f}s")
