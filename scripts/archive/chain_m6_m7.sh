#!/bin/bash
# ARCHIVED -- superseded by chain_m6_m7_resume.sh (this version crashed partway
# through and needed two follow-up bugfixes to m6_train.py/m6_validate.py before
# training could actually run -- see docs/story_so_far.md). Kept for historical
# reference only; hardcodes a PID and a session-specific scratchpad log path that
# no longer exist. Use scripts/m6/train_m6.sh + scripts/m7/ for a runnable version.
set -e
REPO="/Users/karthikeyansundaravaradhan/Desktop/Masters Project/driver-drowsiness-detection-system"
SCRIPTS="/Users/karthikeyansundaravaradhan/Desktop/Masters Project/scripts"
PY="/opt/anaconda3/envs/ddd-repo/bin/python"
EXTRACT_LOG="/private/tmp/claude-501/-Users-karthikeyansundaravaradhan-Desktop-Masters-Project/a9c96569-4853-4a70-ae28-8f7816e31960/scratchpad/m6_extract.log"

echo "[chain] waiting for M6 embedding extraction (pid 4315) to finish..."
while kill -0 4315 2>/dev/null; do sleep 30; done
echo "[chain] extraction finished."
tail -5 "$EXTRACT_LOG"

cd "$REPO"

echo "[chain] Starting M6 lite training (5-fold CV)..."
$PY -m src.models.m6_train --variant lite --epochs 50

echo "[chain] Starting M6 full training (5-fold CV)..."
$PY -m src.models.m6_train --variant full --epochs 50

echo "[chain] M6 done. Starting M7 feature extraction..."
$PY "$SCRIPTS/m7_extract_features.py"

echo "[chain] M7 features ready. Starting M7 training..."
$PY "$SCRIPTS/m7_train.py"

echo "[chain] ALL DONE (M6 lite + full, M7)."
