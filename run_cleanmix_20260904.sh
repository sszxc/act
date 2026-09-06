#!/usr/bin/env bash
# Mixing ladder re-run with pause-stripped scripted data (cmix_* dirs).
# Same 6 ratios, same fixed 9-human val set, same cameras/representation as run_mix_20260904.sh,
# so results drop straight into the same table. The only variable changed is raw vs
# pause-stripped scripted episodes.
set -uo pipefail
cd "$(dirname "$0")"
PY=/home/asu/miniconda3/envs/aloha/bin/python
SWEEP=${SWEEP:-results/mix_20260904}
LOGS=$SWEEP/logs; MAXJOBS=${MAXJOBS:-3}; EPOCHS=${EPOCHS:-12000}
DATA=data/real_pick_yellow_bottle
VAL='val_episode_ids=[0,1,2,3,4,5,6,7,8]'
CAMS='[top,side,wrist]'
REPR="action_repr=delta action_offset=0 qpos_dropout=0.5"
COMMON="task_name=real_pick_yellow_bottle policy_class=ACT lr=1e-4 kl_weight=1 chunk_size=50
        batch_size=8 hidden_dim=512 dim_feedforward=3200 num_epochs=$EPOCHS image_size=[240,320]
        num_workers=5 deploy_every=25 save_every=0"
mkdir -p "$LOGS"

run() {
  local name=$1; shift
  [[ -f "$SWEEP/$name/deploy_metrics.json" ]] && { echo "SKIP $name"; return 0; }
  rm -rf "${SWEEP:?}/$name"; local t0=$SECONDS
  echo "[$(date '+%F %T')] START $name: $*" | tee "$LOGS/$name.log"
  "$PY" imitate_episodes.py --ckpt_dir="$SWEEP/$name" $COMMON "$@" >>"$LOGS/$name.log" 2>&1
  echo "[$(date '+%F %T')] DONE $name exit=$? $((SECONDS-t0))s" | tee -a "$LOGS/$name.log"
}
queue() { while (( $(jobs -rp | wc -l) >= MAXJOBS )); do wait -n; done; run "$@" & }

declare -a MIXES=("cmix_h32_s0|41" "cmix_h32_s5|46" "cmix_h32_s10|51"
                  "cmix_h32_s19|60" "cmix_h16_s19|44" "cmix_h0_s19|28")
for seed in 0 1; do
  for spec in "${MIXES[@]}"; do
    IFS='|' read -r dir n <<<"$spec"
    [[ -d $DATA/$dir ]] || { echo "MISSING $DATA/$dir"; continue; }
    queue "${dir}_s$seed" camera_names="$CAMS" seed=$seed $REPR $VAL \
          dataset_dir=$DATA/$dir num_episodes=$n
  done
done
wait
"$PY" summarize_sweep.py "$SWEEP" --csv "$SWEEP/summary.csv" | tee "$SWEEP/summary.txt"
echo "[$(date '+%F %T')] clean-scripted mixing finished"
