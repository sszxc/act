#!/usr/bin/env bash
# Remaining work, in priority order. Each step is individually resumable (runs with a
# deploy_metrics.json are skipped), so this can be re-run after any interruption.
#   stages 2,3 : the dataset and camera questions that were asked
#   mix        : human/scripted mixing ladder that was asked for
#   stage 5    : hyperparameter re-tune (my own addition -- last, so it is what gets dropped)
#   stage 4    : replay eval over every checkpoint + summary tables
set -uo pipefail
cd "$(dirname "$0")"
echo "[$(date '+%F %T')] === stages 2 and 3 ==="
bash run_sweep_20260904.sh 2 3
echo "[$(date '+%F %T')] === human/scripted mixing ==="
bash run_mix_20260904.sh
echo "[$(date '+%F %T')] === stage 5 ==="
bash run_sweep_20260904.sh 5
echo "[$(date '+%F %T')] === stage 4: replay eval ==="
bash run_sweep_20260904.sh 4
echo "[$(date '+%F %T')] === all done ==="
