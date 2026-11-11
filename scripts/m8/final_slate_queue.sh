#!/bin/bash
# Items 1-4 of the post-round-2 slate + the subject-dependent split question (2026-11-07).
cd "$(dirname "$0")/../.."
PY=/opt/anaconda3/envs/ddd-repo/bin/python
R="$PY -W ignore -u scripts/m8/run_single_stream.py --stream ir_112 --head mean"
set -x
$R --ref 240 --ref-session A --y y_b2 || echo FAILED calA_b2
$R --ref 240 --ref-session A --y y_b2 --protocol loso || echo FAILED calA_b2_loso
$R --y y_b2 --protocol loso || echo FAILED vstar_b2_loso
$PY -W ignore -u scripts/m8/run_vigil.py --modes moe_telclock vision --gate-lr-mult 10 --y y_b2 --name C9d_b2 || echo FAILED m8_b2
for s in random epoch drive; do $PY -W ignore -u scripts/m8/subject_dependent.py --split $s || echo FAILED SD_$s; done
$PY -W ignore -u scripts/m8/train_final.py || echo FAILED train_final
echo "[final] $(date) ALL DONE"
