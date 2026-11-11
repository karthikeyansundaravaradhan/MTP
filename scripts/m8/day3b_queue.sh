#!/bin/bash
# Decision-independent runs while V* is open: binary B2 (C11) for the two candidate streams, and
# the plain telemetry expert (C7 baseline) on the telemetry cohort. Run with:
#   nohup caffeinate -ims bash scripts/m8/day3b_queue.sh > cache/logs/day3b_queue.log 2>&1 & disown
cd "/Users/karthikeyansundaravaradhan/Desktop/Masters Project"
PY="/opt/anaconda3/envs/ddd-repo/bin/python -W ignore -u"
echo "[q3b] $(date) starting"
$PY scripts/m8/run_single_stream.py --stream ir_112 --head mean --y y_b2 || echo "[q3b] FAILED ir_112 b2"
$PY scripts/m8/run_single_stream.py --stream pose_224 --head mil --y y_b2 || echo "[q3b] FAILED pose b2"
for h in mean mil; do
  $PY scripts/m8/run_single_stream.py --stream tel --head $h --cohort tel || echo "[q3b] FAILED tel $h"
done
echo "[q3b] $(date) ALL DONE"
