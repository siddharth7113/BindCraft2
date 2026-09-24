#!/usr/bin/env bash
# Campaign progress at a glance.  Usage: ./progress.sh [results/rbp4_cyclic]
# Note: stage tables are named "!_*.csv" -- always quote or glob them, never leave the ! bare.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
D="${1:-$(ls -td results/*/ 2>/dev/null | head -1)}"; D="${D%/}"
[ -d "$D" ] || { echo "no campaign folder at $D"; exit 1; }

echo "=== campaign: $D ==="
squeue -u "$USER" --format='  job %.6i  %.9T  elapsed %.9M  limit %.9l' 2>/dev/null | tail -n +1
echo
python3 - "$D" <<'PY'
import json, os, sys, csv
d = sys.argv[1]
st = os.path.join(d, '.campaign_state.json')
if os.path.exists(st):
    s = json.load(open(st))
    want = json.load(open(os.path.join(d,'workers','campaign_settings.json'))).get('number_of_final_designs','?') \
           if os.path.exists(os.path.join(d,'workers','campaign_settings.json')) else '?'
    r = s.get('rejections', {})
    print(f"  trajectories run : {s.get('trajectories',0)}")
    print(f"  accepted designs : {s.get('accepted',0)} / {want}")
    print(f"  terminated early : {r.get('terminated',{})}")
    print(f"  candidates scored: {r.get('candidates_scored',0)}   rejected: {r.get('candidates_rejected',0)}")
    if r.get('failed_filters'): print(f"  failed filters   : {r['failed_filters']}")
for stage,name in (('1_Trajectories','!_Trajectories.csv'),
                   ('2_Refolded','!_Refolded.csv'),
                   ('3_Ranked','!_Ranked.csv')):
    p = os.path.join(d, stage, name)
    if os.path.exists(p):
        n = sum(1 for _ in open(p)) - 1
        print(f"  {stage:<16}: {n} rows")
    else:
        print(f"  {stage:<16}: -")
PY
echo
echo "=== screen outcomes (pLDDT gate = 0.60) ==="
grep -h "rejected at screen" "$D"/workers/*.log 2>/dev/null | sed 's/^ */  /' | tail -12
echo
echo "=== in flight ==="
for w in "$D"/workers/worker_*.log; do tail -3 "$w" | grep -q "rejected" || tail -1 "$w" | sed 's/^/  /'; done
echo
echo "=== GPUs ==="
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  /'
