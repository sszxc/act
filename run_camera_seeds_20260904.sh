#!/usr/bin/env bash
# Stage 3 follow-up: paired seed replication of the informative camera sets.
#
# Stage 3 ranked 20 camera sets at one seed each. The total spread across all 20 was 7.1%,
# while the seed-to-seed spread of a SINGLE config is ~24% -- so that ranking is not resolvable.
# Comparisons within a seed are paired (same 80/20 split, same fixed deploy set), so replicating
# a few sets across 5 seeds and comparing rank-by-seed is far more powerful than adding more
# camera sets. If one set wins at 5/5 seeds that is a sign test at p=0.03; 3/5 is noise.
#
# Sets: the incumbent (left+top), the Stage 3 winner (top+side+wrist), all-RealSense, and
# the top+wrist / top+wrist+F pair which isolates the fingertip cameras.
set -uo pipefail
cd "$(dirname "$0")"
PY=/home/asu/miniconda3/envs/aloha/bin/python
SWEEP=${SWEEP:-results/sweep_20260904}
LOGS=$SWEEP/logs; MAXJOBS=${MAXJOBS:-3}; EPOCHS=${EPOCHS:-12000}
DS=data/real_pick_yellow_bottle/good_41
COMMON="task_name=real_pick_yellow_bottle policy_class=ACT lr=1e-4 kl_weight=1 chunk_size=50
        batch_size=8 hidden_dim=512 dim_feedforward=3200 num_epochs=$EPOCHS image_size=[240,320]
        num_workers=5 deploy_every=25 save_every=0"
REPR="action_repr=delta action_offset=0 qpos_dropout=0.5"
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

F=thumb,index,middle,ring
declare -a SETS=(
  "left_top|[left,top]"
  "top_side_wrist|[top,side,wrist]"
  "rs_all|[left,right,side,top,wrist]"
  # all9 dropped from the replication: at 3.75h/run it is the most expensive arm by far, and
  # the questions it would answer are already covered more cheaply -- rs_all covers "many
  # cameras", and top_wrist vs top_wrist_F isolates the fingertip cameras. Its 2 existing seeds
  # stay in the table.
  "top_wrist|[top,wrist]"
  "top_wrist_F|[top,wrist,$F]"
)
for seed in 0 1 2 3 4; do
  for spec in "${SETS[@]}"; do
    IFS='|' read -r name cams <<<"$spec"
    queue "s3_${name}_s$seed" camera_names="$cams" seed=$seed $REPR dataset_dir=$DS
  done
done
wait
"$PY" summarize_sweep.py "$SWEEP" --prefix s3_ --csv "$SWEEP/stage3.csv" | tee "$SWEEP/stage3.txt"
echo "[$(date '+%F %T')] camera seed replication finished"
