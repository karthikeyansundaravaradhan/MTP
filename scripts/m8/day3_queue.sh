#!/bin/bash
# Queued behind day2_queue.sh: C2 two-view (L3D train+test with R3D) run. Run with:
#   nohup caffeinate -ims bash scripts/m8/day3_queue.sh <pid-of-day2-queue> > cache/logs/day3_queue.log 2>&1 & disown
cd "/Users/karthikeyansundaravaradhan/Desktop/Masters Project"
PY="/opt/anaconda3/envs/ddd-repo/bin/python -W ignore -u"
while kill -0 "$1" 2>/dev/null; do sleep 60; done
echo "[queue3] $(date) day2 queue finished, starting C2 TTA"
$PY scripts/m8/run_single_stream.py --stream l3d_168 --tta-stream r3d_168 --head mil || echo "[queue3] FAILED l3d+r3d tta"
echo "[queue3] $(date) ALL DONE"
