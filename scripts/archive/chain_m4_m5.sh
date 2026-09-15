#!/bin/zsh
# ARCHIVED -- historical record only. M4/M5 training already completed successfully
# using this script; it is not meant to be re-run. Paths below point at a session-
# specific scratchpad and the old flat scripts/ layout (pre-reorganization), both of
# which are now stale. See scripts/m4/, scripts/m5/, scripts/common/ for the current,
# runnable equivalents, and docs/story_so_far.md for what this run actually did.
set -e
SCRATCH="/private/tmp/claude-501/-Users-karthikeyansundaravaradhan-Desktop-Masters-Project/a9c96569-4853-4a70-ae28-8f7816e31960/scratchpad"
EXTRACT_LOG="$SCRATCH/m4_m5_extract.log"
DDD_PY="/opt/anaconda3/envs/ddd-repo/bin/python"

echo "[chain] waiting for extraction to finish..."
while ! grep -q "^Done\.$" "$EXTRACT_LOG" 2>/dev/null; do
  if ! pgrep -f "m4_m5_extract.py" > /dev/null; then
    echo "[chain] extraction process died without reaching 'Done.' — aborting chain."
    exit 1
  fi
  sleep 60
done
echo "[chain] extraction finished. Starting M4 training..."
"$DDD_PY" "$SCRATCH/m4_train.py"

echo "[chain] M4 training finished. Starting M5 training..."
"$DDD_PY" "$SCRATCH/m5_train.py"

echo "[chain] ALL DONE"
