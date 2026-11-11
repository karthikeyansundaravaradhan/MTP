#!/bin/bash
# Round-2 V1: LOSO (16 folds, 3 seeds, no inner) for V*, Alert-calibrated V*, and pose MIL.
cd "/Users/karthikeyansundaravaradhan/Desktop/Masters Project"
PY="/opt/anaconda3/envs/ddd-repo/bin/python -W ignore -u"
while kill -0 "$1" 2>/dev/null; do sleep 30; done
echo "[r2l] $(date) starting"
$PY scripts/m8/run_single_stream.py --stream ir_112 --head mean --protocol loso || echo "[r2l] FAILED vstar"
$PY scripts/m8/run_single_stream.py --stream ir_112 --head mean --protocol loso --ref 240 --ref-session A || echo "[r2l] FAILED refA"
$PY scripts/m8/run_single_stream.py --stream pose_224 --head mil --protocol loso || echo "[r2l] FAILED pose"
echo "[r2l] $(date) ALL DONE"
