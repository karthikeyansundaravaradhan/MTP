#!/bin/bash
# C12 start-of-drive normalisation (same-session first 4 min) on the two candidate streams.
cd "/Users/karthikeyansundaravaradhan/Desktop/Masters Project"
PY="/opt/anaconda3/envs/ddd-repo/bin/python -W ignore -u"
echo "[c12] $(date) starting"
$PY scripts/m8/run_single_stream.py --stream ir_112 --head mean --ref 240 || echo "[c12] FAILED ir"
$PY scripts/m8/run_single_stream.py --stream pose_224 --head mil --ref 240 || echo "[c12] FAILED pose"
echo "[c12] $(date) ALL DONE"
