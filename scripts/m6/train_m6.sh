#!/bin/bash
# M6 -- Visual-CAN fusion (Lite + Full variants), with the alignment fix applied.
#
# M6 has no dedicated script of ours: training runs Kavipriya's own
# src/models/m6_train.py module directly via its CLI. This wrapper documents the
# exact commands and order that were actually run for this project's results
# (see docs/story_so_far.md and docs/repo_understanding.md section 7.1 for the
# three local bugfixes to m6_train.py/m6_validate.py this required, and why both
# variants ended up *worse* than Kavipriya's original broken-alignment numbers).
#
# Prerequisite: 60fps-aligned visual embeddings must exist at
# driver-drowsiness-detection-system/models/embeddings_uldd/ -- generate them with:
#
#   cd "<repo>/driver-drowsiness-detection-system"
#   /opt/anaconda3/envs/ddd-repo/bin/python -m src.models.m6_extractor_from_video \
#     --ckpt models/checkpoints_regen/M5_fold0.pt \
#     --video-root "<dataset>/DriverDrowsinessDataset/Video_Data/Video_Data" \
#     --out-dir models/embeddings_uldd \
#     --fps 60 --batch-size 64 --num-workers 8
#
set -e
REPO="/Users/karthikeyansundaravaradhan/Desktop/Masters Project/driver-drowsiness-detection-system"
PY="/opt/anaconda3/envs/ddd-repo/bin/python -u"

cd "$REPO"

echo "[M6] Training lite variant (5-fold CV)..."
$PY -m src.models.m6_train --variant lite --epochs 50

echo "[M6] Training full variant (5-fold CV)..."
$PY -m src.models.m6_train --variant full --epochs 50

echo "[M6] Done. Results in results/reports/M6_{lite,full}_results.json"
