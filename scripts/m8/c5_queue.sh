#!/bin/bash
# C5 on the selected stream (ir_112, by inner-OOF): temporal transformer head, and 4-min causal
# context at equal step budget (240 s at 0.5 fps = 120 steps, same as 60 s at 2 fps). Run with:
#   nohup caffeinate -ims bash scripts/m8/c5_queue.sh <pid-to-wait-for> > cache/logs/c5_queue.log 2>&1 & disown
cd "/Users/karthikeyansundaravaradhan/Desktop/Masters Project"
PY="/opt/anaconda3/envs/ddd-repo/bin/python -W ignore -u"
while kill -0 "$1" 2>/dev/null; do sleep 60; done
echo "[c5] $(date) starting"
$PY scripts/m8/run_single_stream.py --stream ir_112 --head tf || echo "[c5] FAILED tf"
for h in mean mil tf; do
  $PY scripts/m8/run_single_stream.py --stream ir_112 --head $h --context 240 --ctx-stride 4 || echo "[c5] FAILED $h ctx240"
done
echo "[c5] $(date) ALL DONE"
