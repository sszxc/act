#!/usr/bin/env bash
# Closed-loop-instability follow-up (see session notes 2026-09-10): the 4 `action_repr=delta`
# runs under results/resynced_longrun_datasets_20260909/ all collapse under closed-loop replay
# (replay_eval.py --qpos_source policy), regardless of dataset. Two single-variable ablations
# vs. each of those 4 runs' exact config, changing ONLY action_repr or ONLY qpos_dropout (never
# both in the same run, so any closed-loop improvement is attributable) -- everything else
# (dataset_dir, camera_names, chunk_size, hidden_dim, dim_feedforward, joint_ids,
# val_episode_ids, action_offset, kl_weight, batch/epoch sizing) copied verbatim from each run's
# own config_hydra_resolved.yaml. task_space runs (taskspace, taskspace_armonly) are untouched --
# they're not action_repr=delta, not in scope here.
#
# Priority 1 (always runs): action_repr=absolute, qpos_dropout unchanged (0.5).
# Priority 2 (only if the 16h budget allows after priority 1 finishes): qpos_dropout=0.0,
# action_repr back to each run's own original (delta) -- isolates the dropout axis alone.
#
# Per CLAUDE.md instructions: validation-set loss during training is NOT a decision signal here
# (this script never picks/prunes on it) -- every epoch checkpoint + best_val/best_track/
# best_deploy are all just saved as-is for real-robot testing tomorrow.
#
# Jobs are tracked by polling `pgrep -f "ckpt_dir <dir> "` rather than bash job control, so a
# job already running under a matching ckpt_dir (e.g. the first launch of this script, which had
# a `pids+=($(fn))` word-splitting bug and got restarted -- d1_good41_absolute/d2_scripted50_absolute
# were already correctly running and were left alone rather than killed) is picked up and waited
# on the same way as one this invocation launches itself.
set -u
cd "$(dirname "$0")"
PY=/home/asu/miniconda3/envs/aloha/bin/python
LOG=run_absolute_dropout_ablation_20260910.orchestration.log
DEADLINE=$(( $(date +%s) + 16*3600 - 5*60 ))   # ~16h budget from the ORIGINAL launch (~5min already spent on the restart)

exec >> "$LOG" 2>&1
echo "=== orchestration (re)start $(date), deadline $(date -d @$DEADLINE) ==="

COMMON="task_name=real_pick_yellow_bottle camera_names=[top,wrist] \
action_offset=0 chunk_size=30 hidden_dim=512 dim_feedforward=3200 no_encoder=false kl_weight=10 \
amp=true num_workers=8 batch_size=128 batches_per_epoch=16 lr=1e-4 num_epochs=2000 \
save_every=200 deploy_every=50 resume_every=90"

VAL='val_episode_ids=[0,3,6,9,19,21,23,24,39]'

launch () {  # launch <out_dir> <name> <dataset_dir> <num_episodes> <action_repr> <qpos_dropout> <extra overrides...>
  local out=$1; local name=$2; local dset=$3; local nep=$4; local repr=$5; local drop=$6; shift 6
  mkdir -p "$out/logs"
  if pgrep -f "ckpt_dir $out/$name " > /dev/null; then
    echo "$name already running (from an earlier attempt) -- attaching, not relaunching"
    return
  fi
  local R=""
  [ -f "$out/$name/train_state.pt" ] && R="resume=true" && echo "resuming $name"
  nohup $PY imitate_episodes.py --ckpt_dir "$out/$name" $COMMON \
      dataset_dir=$dset num_episodes=$nep action_repr=$repr qpos_dropout=$drop $R "$@" \
      >> "$out/logs/$name.log" 2>&1 &
  disown
  echo "launched $name (action_repr=$repr qpos_dropout=$drop)"
}

wait_for () {  # wait_for <out_dir> <name> -- poll until no process matches this ckpt_dir
  local out=$1; local name=$2
  while pgrep -f "ckpt_dir $out/$name " > /dev/null; do sleep 30; done
  echo "$name done $(date)"
}

run_tier () {  # run_tier <out_dir> <suffix> <action_repr> <qpos_dropout>
  local out=$1; local suffix=$2; local repr=$3; local drop=$4
  launch "$out" "d1_good41_$suffix"     data/real_pick_yellow_bottle/resynced_good_41_tw240      41 "$repr" "$drop" $VAL
  sleep 90   # stagger: norm-stats + deploy-set construction are CPU-heavy, don't overlap them
  launch "$out" "d2_scripted50_$suffix" data/real_pick_yellow_bottle/resynced_scripted_50c_tw240 50 "$repr" "$drop"
  sleep 90
  launch "$out" "d2_armonly_$suffix"    data/real_pick_yellow_bottle/resynced_scripted_50c_tw240 50 "$repr" "$drop" 'joint_ids=[0,1,2,3,4,5,6,7]'
  sleep 90
  launch "$out" "d3_mix91_$suffix"      data/real_pick_yellow_bottle/resynced_h41_s50c_tw240     91 "$repr" "$drop" $VAL
  echo "waiting on tier '$suffix'..."
  wait_for "$out" "d1_good41_$suffix"
  wait_for "$out" "d2_scripted50_$suffix"
  wait_for "$out" "d2_armonly_$suffix"
  wait_for "$out" "d3_mix91_$suffix"
  echo "=== tier '$suffix' finished $(date) ==="
}

T1_START=$(date +%s)
run_tier results/resynced_longrun_datasets_20260910_absolute absolute absolute 0.5
T1_END=$(date +%s)
T1_DUR=$(( T1_END - T1_START ))
echo "priority-1 (absolute) took $((T1_DUR/3600))h$((T1_DUR%3600/60))m"

MARGIN=1800  # 30min safety buffer
if (( T1_END + T1_DUR + MARGIN <= DEADLINE )); then
  echo "budget allows priority-2 (qpos_dropout=0.0) -- launching"
  run_tier results/resynced_longrun_datasets_20260910_dropout0 dropout0 delta 0.0
else
  remaining=$(( DEADLINE - T1_END ))
  echo "SKIPPING priority-2: est. ${T1_DUR}s needed but only ${remaining}s left in the 16h budget"
fi

echo "=== orchestration done $(date) ==="
