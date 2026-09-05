#!/usr/bin/env bash
# real_pick_yellow_bottle — human-teleop vs scripted (state-machine, fixed-waypoint) data mixing.
#
# The scripted batch (~/data/data_0904/success, 19 successes) is converted to
# data/real_pick_yellow_bottle/scripted_0904_c19 and combined with the 41 human episodes into
# mix_h<H>_s<S> dirs. Every mix dir puts the SAME 9 held-out human episodes at indices 0..8, and
# every run here passes val_episode_ids=[0..8], so all mixes are validated on one identical
# human-only set and their losses are directly comparable despite different dataset sizes.
# mix_h32_s0 is the no-scripted-data control.
#
#   bash run_mix_20260904.sh
# Safe to re-run: any run with deploy_metrics.json is skipped. Do not run this at the same time
# as run_sweep_20260904.sh -- the GPU is already saturated by that script's 3 concurrent jobs.
set -uo pipefail
cd "$(dirname "$0")"

PY=/home/asu/miniconda3/envs/aloha/bin/python
SWEEP=${SWEEP:-results/mix_20260904}
MAIN=results/sweep_20260904
LOGS=$SWEEP/logs
MAXJOBS=${MAXJOBS:-3}
DATA=data/real_pick_yellow_bottle
EPOCHS=${EPOCHS:-12000}
VAL='val_episode_ids=[0,1,2,3,4,5,6,7,8]'
COMMON="task_name=real_pick_yellow_bottle policy_class=ACT lr=1e-4 kl_weight=1 chunk_size=50
        batch_size=8 hidden_dim=512 dim_feedforward=3200 num_epochs=$EPOCHS image_size=[240,320]
        num_workers=5 deploy_every=25 save_every=0"

mkdir -p "$LOGS"

# Inherit the action representation and camera set the main sweep settled on; fall back to the
# defaults this investigation started from if it hasn't got that far.
REPR="action_repr=delta action_offset=0 qpos_dropout=0.0"
[[ -f $MAIN/winner.env ]] && source "$MAIN/winner.env" && \
  REPR="action_repr=$ACTION_REPR action_offset=$ACTION_OFFSET qpos_dropout=$QPOS_DROPOUT"
CAMS='[left,top]'
if [[ -f $MAIN/stage3.csv ]]; then
  best=$("$PY" -c "
import sys
sys.path.insert(0, '.')
from summarize_sweep import collect
rows = [r for r in collect('$MAIN', 's3_') if r['status'] == 'done']
print('[' + rows[0]['camera_names'].replace('+', ',') + ']' if rows else '')")
  [[ -n $best ]] && CAMS=$best
fi
echo "=== representation: $REPR"
echo "=== cameras:        $CAMS"

run() {
  local name=$1; shift
  if [[ -f "$SWEEP/$name/deploy_metrics.json" ]]; then
    echo "[$(date '+%F %T')] SKIP $name (already finished)"; return 0
  fi
  rm -rf "${SWEEP:?}/$name"
  local t0=$SECONDS
  echo "[$(date '+%F %T')] START $name: $*" | tee "$LOGS/$name.log"
  "$PY" imitate_episodes.py --ckpt_dir="$SWEEP/$name" $COMMON "$@" >>"$LOGS/$name.log" 2>&1
  local st=$?
  echo "[$(date '+%F %T')] DONE $name exit=$st $((SECONDS-t0))s" | tee -a "$LOGS/$name.log"
}

queue() {
  while (( $(jobs -rp | wc -l) >= MAXJOBS )); do wait -n; done
  run "$@" &
}

# dir -> total episode count (9 val human + the rest)
declare -a MIXES=(
  "mix_h32_s0|41"    # control: no scripted data
  "mix_h32_s5|46"
  "mix_h32_s10|51"
  "mix_h32_s19|60"
  "mix_h16_s19|44"   # ~50/50 human:scripted
  "mix_h0_s19|28"    # scripted only -- the reference that says whether it is usable alone
)

for seed in 0 1; do
  for spec in "${MIXES[@]}"; do
    IFS='|' read -r dir n <<<"$spec"
    if [[ ! -d $DATA/$dir ]]; then echo "MISSING $DATA/$dir -- skipping"; continue; fi
    queue "${dir}_s$seed" camera_names="$CAMS" seed=$seed $REPR $VAL \
          dataset_dir=$DATA/$dir num_episodes=$n
  done
done
wait

"$PY" summarize_sweep.py "$SWEEP" --csv "$SWEEP/summary.csv" | tee "$SWEEP/summary.txt"

echo "=== open-loop replay eval ==="
for run_dir in "$SWEEP"/*/; do
  name=$(basename "$run_dir")
  [[ -f "$run_dir/policy_best_deploy.ckpt" ]] || continue
  [[ -f "$run_dir/replay_eval.json" ]] && continue
  readarray -t meta < <("$PY" -c "
import yaml, sys
c = yaml.safe_load(open(sys.argv[1]))
print(' '.join(c['camera_names']))
print(c['dataset_dir'])
print(c['num_episodes'])" "$run_dir/config_hydra_resolved.yaml")
  # Replay on the 9 shared human val episodes (indices 0..8 of every mix dir).
  "$PY" replay_eval.py --ckpt_dir "$run_dir" --ckpt_name policy_best_deploy.ckpt \
      --dataset_dir "${meta[1]}" --num_episodes "${meta[2]}" --camera_names ${meta[0]} \
      --image_size 240 320 --chunk_size 50 --val_episode_ids 0 1 2 3 4 5 6 7 8 \
      >>"$LOGS/$name.replay.log" 2>&1 || echo "  replay FAILED for $name"
done
"$PY" summarize_sweep.py "$SWEEP" --csv "$SWEEP/summary.csv" | tee "$SWEEP/summary.txt"
echo "[$(date '+%F %T')] mix sweep finished"
