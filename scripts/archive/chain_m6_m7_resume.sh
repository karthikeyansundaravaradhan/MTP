#!/bin/bash
# ARCHIVED -- historical record only. This is the version that actually succeeded
# (all 3 m6_train.py/m6_validate.py bugfixes applied, MPS enabled) and produced the
# M6 lite/full + M7 results documented in docs/story_so_far.md. It references the
# old flat scripts/ layout (m7_extract_features.py, m7_train.py) which has since
# moved to scripts/m7/. Use scripts/m6/train_m6.sh + scripts/m7/ to re-run.
set -e
REPO="/Users/karthikeyansundaravaradhan/Desktop/Masters Project/driver-drowsiness-detection-system"
SCRIPTS="/Users/karthikeyansundaravaradhan/Desktop/Masters Project/scripts"
PY="/opt/anaconda3/envs/ddd-repo/bin/python -u"

cd "$REPO"

echo "[chain] (resume) M6 embeddings already extracted -- starting M6 lite training..."
$PY -m src.models.m6_train --variant lite --epochs 50

echo "[chain] Starting M6 full training (5-fold CV)..."
$PY -m src.models.m6_train --variant full --epochs 50

echo "[chain] M6 done. Starting M7 feature extraction..."
$PY "$SCRIPTS/m7_extract_features.py"

echo "[chain] M7 features ready. Starting M7 training..."
$PY "$SCRIPTS/m7_train.py"

echo "[chain] ALL DONE (M6 lite + full, M7)."
