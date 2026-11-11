#!/bin/bash
# Round-2 R2-12: DINOv3-S/16 IR face crops at 112 px / 2 fps, then the V* head on them.
cd "/Users/karthikeyansundaravaradhan/Desktop/Masters Project"
PY="/opt/anaconda3/envs/ddd-repo/bin/python -W ignore -u"
while kill -0 "$1" 2>/dev/null; do sleep 30; done
echo "[r2d] $(date) extracting"
$PY -m m8.extract --stream ir --px 112 --fps 2 --encoder vit_small_patch16_dinov3.lvd1689m || echo "[r2d] FAILED extract"
$PY scripts/m8/run_single_stream.py --stream ir_112_dinov3 --head mean || echo "[r2d] FAILED head"
echo "[r2d] $(date) ALL DONE"
