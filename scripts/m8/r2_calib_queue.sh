#!/bin/bash
# Round-2 R2-7 / R2-8 on V* (IR-112 mean-pool): Alert-session calibration, and absolute + delta.
cd "/Users/karthikeyansundaravaradhan/Desktop/Masters Project"
PY="/opt/anaconda3/envs/ddd-repo/bin/python -W ignore -u"
echo "[r2c] $(date) starting"
$PY scripts/m8/run_single_stream.py --stream ir_112 --head mean --ref 240 --ref-session A || echo "[r2c] FAILED refA"
$PY scripts/m8/run_single_stream.py --stream ir_112 --head mean --ref 240 --ref-session A --ref-concat || echo "[r2c] FAILED refAcat"
echo "[r2c] $(date) ALL DONE"
