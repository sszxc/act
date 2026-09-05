#!/usr/bin/env bash
# real_pick_yellow_bottle — 80h sweep, 2026-09-04. Four questions, in dependency order:
#
#   Stage 1  action representation. This dataset has action[t] == qpos[t+1] exactly, so copying
#            qpos is a near-optimal solution to the training loss and a useless policy on the
#            robot. Tests delta actions and proprio dropout against the current absolute setup.
#   Stage 2  good_41 vs good_41_clean (teleop pauses stripped), under Stage 1's winner.
#   Stage 3  camera combinations, under the Stage 1 winner and Stage 2's dataset.
#   Stage 5  re-tune chunk_size / lr / training length under the winning representation
#            (the existing tuning was done under the copycat regime, so it may not transfer).
#   Stage 4  open-loop replay eval of every checkpoint + summary table. Runs last.
#
# Ranking metric is deploy/l1_rad: chunk L1 in radians on the *deployed* path (CVAE prior,
# z=0, no ground-truth actions), on a fixed val sample set. The ordinary val loss runs the
# posterior and is not comparable across action representations. See
# imitate_episodes.deploy_metrics(). Comparisons that change chunk_size must use replay_eval's
# per-timestep cmd_l1 instead -- l1_rad averages over the chunk, so it grows with chunk length.
#
#   bash run_sweep_20260904.sh            # all stages
#   bash run_sweep_20260904.sh 3 4        # only these (reuses winner.env from a prior run)
# Re-running is safe: any run with deploy_metrics.json is skipped.
set -uo pipefail
cd "$(dirname "$0")"

PY=/home/asu/miniconda3/envs/aloha/bin/python
SWEEP=${SWEEP:-results/sweep_20260904}
LOGS=$SWEEP/logs
MAXJOBS=${MAXJOBS:-3}
DATA=data/real_pick_yellow_bottle
EPOCHS=${EPOCHS:-12000}   # env-overridable, so the driver can be smoke-tested cheaply
COMMON="task_name=real_pick_yellow_bottle policy_class=ACT lr=1e-4 kl_weight=1 chunk_size=50
        batch_size=8 hidden_dim=512 dim_feedforward=3200 num_epochs=$EPOCHS image_size=[240,320]
        num_workers=5 deploy_every=25 save_every=0"

mkdir -p "$LOGS"
WANT=("$@")
stage_wanted() { [[ ${#WANT[@]} -eq 0 || " ${WANT[*]} " == *" $1 "* ]]; }

run() {  # run <name> <hydra overrides...>
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

queue() {  # respects MAXJOBS
  while (( $(jobs -rp | wc -l) >= MAXJOBS )); do wait -n; done
  run "$@" &
}

top_n() {  # top_n <prefix> <n> -> "<run> <action_repr> <action_offset> <qpos_dropout> <cams>"
  "$PY" - "$SWEEP" "$1" "$2" <<'EOF'
import sys
from summarize_sweep import collect
sweep, prefix, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
for r in [r for r in collect(sweep, prefix) if r['status'] == 'done'][:n]:
    print(r['run'], r['action_repr'], r['action_offset'], r['qpos_dropout'],
          r['camera_names'].replace('+', ','))
EOF
}

# ============================ Stage 1: action representation =================
if stage_wanted 1; then
  echo "=== Stage 1: action representation (cameras fixed at left+top) ==="
  S1="dataset_dir=$DATA/good_41 camera_names=[left,top] seed=0"
  queue s1_abs_legacy    $S1 action_repr=absolute action_offset=-1 qpos_dropout=0.0
  queue s1_abs           $S1 action_repr=absolute action_offset=0  qpos_dropout=0.0
  queue s1_delta         $S1 action_repr=delta    action_offset=0  qpos_dropout=0.0
  queue s1_abs_drop50    $S1 action_repr=absolute action_offset=0  qpos_dropout=0.5
  queue s1_delta_drop50  $S1 action_repr=delta    action_offset=0  qpos_dropout=0.5
  queue s1_delta_drop100 $S1 action_repr=delta    action_offset=0  qpos_dropout=1.0
  wait
  # Reseed the top two, so the pick isn't a one-seed accident.
  while read -r name repr off drop _; do
    for s in 1 2; do
      queue "${name}_s$s" $S1 seed=$s action_repr=$repr action_offset=$off qpos_dropout=$drop
    done
  done < <(top_n s1_ 2)
  wait
  "$PY" summarize_sweep.py "$SWEEP" --prefix s1_ --csv "$SWEEP/stage1.csv" \
        --pick "$SWEEP/winner.env" | tee "$SWEEP/stage1.txt"
fi

[[ -f $SWEEP/winner.env ]] && source "$SWEEP/winner.env"
REPR="action_repr=${ACTION_REPR:-delta} action_offset=${ACTION_OFFSET:-0} qpos_dropout=${QPOS_DROPOUT:-0.5}"
echo "=== representation for stages 2-5: $REPR ==="

# ============================ Stage 2: dataset ===============================
if stage_wanted 2; then
  echo "=== Stage 2: good_41 vs good_41_clean ==="
  for s in 0 1 2; do
    queue "s2_raw_s$s"   camera_names=[left,top] seed=$s $REPR dataset_dir=$DATA/good_41
    queue "s2_clean_s$s" camera_names=[left,top] seed=$s $REPR dataset_dir=$DATA/good_41_clean
  done
  # Same A/B under the original absolute-action setup, so the dataset question is answered
  # independently of whether Stage 1 changed the representation.
  for s in 0 1; do
    queue "s2_absraw_s$s"   camera_names=[left,top] seed=$s dataset_dir=$DATA/good_41 \
          action_repr=absolute action_offset=-1 qpos_dropout=0.0
    queue "s2_absclean_s$s" camera_names=[left,top] seed=$s dataset_dir=$DATA/good_41_clean \
          action_repr=absolute action_offset=-1 qpos_dropout=0.0
  done
  wait
  "$PY" summarize_sweep.py "$SWEEP" --prefix s2_ --csv "$SWEEP/stage2.csv" | tee "$SWEEP/stage2.txt"
  # Whichever dataset wins on the mean of its 3 seeds carries into Stage 3.
  "$PY" - "$SWEEP" "$DATA" > "$SWEEP/stage2_dataset.env" <<'EOF'
import statistics, sys
from summarize_sweep import collect
sweep, data = sys.argv[1], sys.argv[2]
rows = [r for r in collect(sweep, 's2_') if r['status'] == 'done']
mean = lambda p: statistics.mean([r['l1_rad'] for r in rows if r['run'].startswith(p)] or [9e9])
print(f"STAGE2_DATASET={data}/good_41_clean" if mean('s2_clean') < mean('s2_raw')
      else f"STAGE2_DATASET={data}/good_41")
EOF
  cat "$SWEEP/stage2_dataset.env"
fi

DS=$DATA/good_41
[[ -f $SWEEP/stage2_dataset.env ]] && source "$SWEEP/stage2_dataset.env" && DS=$STAGE2_DATASET
echo "=== dataset for stages 3-5: $DS ==="

# ============================ Stage 3: cameras ===============================
if stage_wanted 3; then
  echo "=== Stage 3: camera combinations ==="
  F=thumb,index,middle,ring
  declare -a SETS=(
    # 2-3 RealSense cameras (the range asked for)
    "left_top|[left,top]"
    "top_wrist|[top,wrist]"
    "side_wrist|[side,wrist]"
    "left_right|[left,right]"
    "top_side|[top,side]"
    "left_wrist|[left,wrist]"
    "left_top_wrist|[left,top,wrist]"
    "top_side_wrist|[top,side,wrist]"
    "left_right_wrist|[left,right,wrist]"
    "left_top_side|[left,top,side]"
    # all RealSense + leave-one-out from it -> per-camera importance, which pairwise
    # comparisons alone can't give (a camera can be redundant in one pair and vital in another)
    "rs_all|[left,right,side,top,wrist]"
    "rs_no_left|[right,side,top,wrist]"
    "rs_no_right|[left,side,top,wrist]"
    "rs_no_side|[left,right,top,wrist]"
    "rs_no_top|[left,right,side,wrist]"
    "rs_no_wrist|[left,right,side,top]"
    # the 4 fingertip cameras added to the most plausible bases
    "left_top_F|[left,top,$F]"
    "top_wrist_F|[top,wrist,$F]"
    "left_top_wrist_F|[left,top,wrist,$F]"
    "all9|[left,right,side,top,wrist,$F]"
  )
  for spec in "${SETS[@]}"; do
    IFS='|' read -r name cams <<<"$spec"
    queue "s3_${name}_s0" camera_names="$cams" seed=0 $REPR dataset_dir=$DS
  done
  wait
  while read -r name _ _ _ cams; do
    queue "${name%_s0}_s1" camera_names="[$cams]" seed=1 $REPR dataset_dir=$DS
  done < <(top_n s3_ 6)
  wait
  "$PY" summarize_sweep.py "$SWEEP" --prefix s3_ --csv "$SWEEP/stage3.csv" | tee "$SWEEP/stage3.txt"
fi

# ============================ Stage 5: re-tune ===============================
if stage_wanted 5; then
  echo "=== Stage 5: hyperparameters under the winning representation ==="
  CAMS='[left,top]'
  if [[ -f $SWEEP/stage3.csv ]]; then
    read -r _ _ _ _ best_cams < <(top_n s3_ 1); [[ -n ${best_cams:-} ]] && CAMS="[$best_cams]"
  fi
  echo "cameras for stage 5: $CAMS"
  B="camera_names=$CAMS seed=0 $REPR dataset_dir=$DS"
  queue s5_cs25    $B chunk_size=25
  queue s5_cs100   $B chunk_size=100
  queue s5_lr3e-5  $B lr=3e-5
  queue s5_lr3e-4  $B lr=3e-4
  queue s5_kl10    $B kl_weight=10
  # Every delta variant commands only ~25% of the true motion (motion_ratio), the signature of
  # L1 regressing to the conditional median. The CVAE latent is meant to absorb that
  # multimodality but its KL collapses to ~0, leaving a deterministic median predictor. Lower
  # kl_weight is the direct test of whether keeping the latent alive restores motion amplitude.
  queue s5_kl0.1   $B kl_weight=0.1
  queue s5_kl0.01  $B kl_weight=0.01
  queue s5_bpe16   $B batches_per_epoch=16 num_epochs=3000   # 4x the gradient steps per epoch
  queue s5_long    $B num_epochs=30000
  wait
  "$PY" summarize_sweep.py "$SWEEP" --prefix s5_ --csv "$SWEEP/stage5.csv" | tee "$SWEEP/stage5.txt"
fi

# ============================ Stage 4: replay eval ===========================
if stage_wanted 4; then
  echo "=== Stage 4: open-loop replay eval ==="
  for run_dir in "$SWEEP"/*/; do
    name=$(basename "$run_dir")
    [[ -f "$run_dir/policy_best_deploy.ckpt" ]] || continue
    [[ -f "$run_dir/replay_eval.json" ]] && continue
    readarray -t meta < <("$PY" -c "
import yaml, sys
c = yaml.safe_load(open(sys.argv[1]))
print(' '.join(c['camera_names'] or ['left', 'top']))
print(c['dataset_dir'] or 'data/real_pick_yellow_bottle/good_41')
print(c['chunk_size'])" "$run_dir/config_hydra_resolved.yaml")
    echo "[$(date '+%F %T')] replay $name (${meta[0]})"
    "$PY" replay_eval.py --ckpt_dir "$run_dir" --ckpt_name policy_best_deploy.ckpt \
        --dataset_dir "${meta[1]}" --camera_names ${meta[0]} --image_size 240 320 \
        --chunk_size "${meta[2]}" >>"$LOGS/$name.replay.log" 2>&1 \
        || echo "  replay FAILED for $name (see $LOGS/$name.replay.log)"
  done
  "$PY" summarize_sweep.py "$SWEEP" --csv "$SWEEP/summary.csv" | tee "$SWEEP/summary.txt"
fi

echo "[$(date '+%F %T')] sweep script finished"
