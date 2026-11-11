#!/bin/bash
# Day 1 -> 2 overnight chain: frozen DINOv2 extraction for all video streams, then the
# honest M5 retrain (R4, 2 seeds), then window-level scoring. Run with:
#   nohup caffeinate -ims bash scripts/m8/day1_overnight.sh > cache/logs/day1_overnight.log 2>&1 & disown
set -e
cd "/Users/karthikeyansundaravaradhan/Desktop/Masters Project"
PY="/opt/anaconda3/envs/ddd-repo/bin/python -W ignore -u"

echo "[chain] $(date) extraction start"
$PY -m m8.extract --stream ir   --px 168 --fps 2
$PY -m m8.extract --stream ir   --px 112 --fps 2
$PY -m m8.extract --stream l3d  --px 168 --fps 2
$PY -m m8.extract --stream r3d  --px 168 --fps 2
$PY -m m8.extract --stream pose --px 224 --fps 1
echo "[chain] $(date) extraction done"

$PY scripts/m8/r4_m5_honest.py --mode retrain --seeds 0 1 2 --epochs 10
for s in 0 1 2; do $PY scripts/m8/score_frame_run.py R4_seed$s; done
echo "[chain] $(date) ALL DONE"
