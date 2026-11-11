#!/bin/bash
# C10 ordinal objective on V* (IR-112 mean-pool), queued behind the M8 gate run.
cd "/Users/karthikeyansundaravaradhan/Desktop/Masters Project"
PY="/opt/anaconda3/envs/ddd-repo/bin/python -W ignore -u"
while kill -0 "$1" 2>/dev/null; do sleep 30; done
echo "[c10] $(date) starting"
$PY scripts/m8/run_single_stream.py --stream ir_112 --head mean --loss ordinal || echo "[c10] FAILED"
echo "[c10] $(date) ALL DONE"
