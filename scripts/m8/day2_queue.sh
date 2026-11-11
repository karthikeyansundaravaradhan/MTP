#!/bin/bash
# Queued behind day1_overnight.sh: C1-C4 single-stream runs (3 seeds + inner CV), then
# linear-probe (mean-pool) controls. Run with:
#   nohup caffeinate -ims bash scripts/m8/day2_queue.sh <pid-of-day1-chain> > cache/logs/day2_queue.log 2>&1 & disown
cd "/Users/karthikeyansundaravaradhan/Desktop/Masters Project"
PY="/opt/anaconda3/envs/ddd-repo/bin/python -W ignore -u"
while kill -0 "$1" 2>/dev/null; do sleep 60; done
echo "[queue] $(date) day1 chain finished, starting C1-C4"
for s in ir_112 ir_168 l3d_168 pose_224 tab; do
  $PY scripts/m8/run_single_stream.py --stream $s --head mil || echo "[queue] FAILED $s"
done
for s in ir_112 l3d_168 pose_224 tab; do
  $PY scripts/m8/run_single_stream.py --stream $s --head mean || echo "[queue] FAILED $s mean"
done
echo "[queue] $(date) ALL DONE"
