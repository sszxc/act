#!/usr/bin/env bash
# Follow-ups, after run_all_20260904.sh finishes. Priority order: the two questions that were
# asked and are still unresolved come first, then a final replay-eval pass over everything.
# Fully self-contained (waits for its own prerequisites) so it survives detached.
set -uo pipefail
cd "$(dirname "$0")"
DATA=data/real_pick_yellow_bottle

echo "[$(date '+%F %T')] waiting for run_all_20260904 to finish"
until ! pgrep -f run_all_20260904 >/dev/null; do sleep 300; done

echo "[$(date '+%F %T')] === camera seed replication ==="
bash run_camera_seeds_20260904.sh

echo "[$(date '+%F %T')] waiting for clean_pauses on the scripted batch"
until [[ $(ls "$DATA"/scripted_0904_c19_clean/*.hdf5 2>/dev/null | wc -l) -eq 19 ]] \
      && ! pgrep -f clean_pauses >/dev/null; do sleep 120; done
/home/asu/miniconda3/envs/aloha/bin/python build_clean_mixes.py

echo "[$(date '+%F %T')] === mixing with pause-stripped scripted data ==="
bash run_cleanmix_20260904.sh

echo "[$(date '+%F %T')] === final replay eval over everything ==="
bash run_sweep_20260904.sh 4

echo "[$(date '+%F %T')] === follow-ups done ==="
