#!/bin/bash
# Subject-dependent splits (rerun after the shortcut_X string-array fix); waits for train_final.
cd "$(dirname "$0")/../.."
while pgrep -f scripts/m8/train_final.py >/dev/null; do sleep 30; done
echo "[sd] train_final finished"
for s in random epoch drive; do
  /opt/anaconda3/envs/ddd-repo/bin/python -W ignore -u scripts/m8/subject_dependent.py --split $s || echo "FAILED SD_$s"
done
echo "[sd] ALL DONE"
