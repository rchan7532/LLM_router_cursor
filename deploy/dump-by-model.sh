#!/bin/bash
cd /opt/llm-router
TOKEN=$(docker exec llm-router-control sh -c 'echo -n "$CONTROL_TOKEN"')
curl -s -m 30 "http://127.0.0.1:4010/usage?n=2000" -H "Authorization: Bearer $TOKEN" | python3 -c "
import json,sys
d=json.load(sys.stdin)['summary']
print('count', d['count'], 'real_hkd', d['real_hkd'], 'est_hkd', d['est_hkd'])
print('prompt', d['prompt_tokens'], 'completion', d['completion_tokens'], 'cached', d['cached_tokens'])
print('--- by_model ---')
for m,v in sorted(d['by_model'].items()):
    print(m, v)
print('--- by_client ---')
for c,v in sorted(d['by_client'].items()):
    print(c, v)
print('--- by_kind ---')
for k,v in sorted(d['by_kind'].items()):
    print(k, v)
print('--- by_task ---')
print(json.dumps(d['by_task'], indent=1))
print('--- task_redo ---', d['task_redo'])
"
