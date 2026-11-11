#!/bin/bash
# Round-2 R2-2: IR both-eye crops at 10 fps (DINOv2-S, 112 px), then mean-pool and MIL heads.
cd "/Users/karthikeyansundaravaradhan/Desktop/Masters Project"
PY="/opt/anaconda3/envs/ddd-repo/bin/python -W ignore -u"
echo "[r2e] $(date) extracting"
$PY -m m8.extract --stream ireye --px 112 --fps 10 || echo "[r2e] FAILED extract"
for h in mean mil; do
  $PY scripts/m8/run_single_stream.py --stream ireye_112 --head $h || echo "[r2e] FAILED $h"
done
echo "[r2e] $(date) ALL DONE"
