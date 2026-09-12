#!/usr/bin/env bash
# Follow-up to run_absolute_dropout_ablation_20260910.sh: that script isolated action_repr and
# qpos_dropout as two SEPARATE single-variable changes (one experiment each, never together).
# This one stacks both on the same run -- action_repr=absolute AND qpos_dropout=0.0 -- for all
# 4 of the action_repr=delta datasets (good41, scripted50, armonly, mix91), confirmed with the
# user (2026-09-11) to mean all 4, not 3. Everything else copied verbatim from each dataset's own
# results/resynced_longrun_datasets_20260909/*/config_hydra_resolved.yaml, same as before.
# Uses the same pgrep-based job tracking as the fixed version of the previous script (a
# `pids+=($(fn))` word-splitting bug there -- see that script's header comment -- is why this one
# never routes launch()'s pid through command substitution at all).
set -u
cd "$(dirname "$0")"
PY=/home/asu/miniconda3/envs/aloha/bin/python
LOG=run_absolute_dropout0_combo_20260911.orchestration.log
OUT=results/resynced_longrun_datasets_20260911_absolute_dropout0

exec >> "$LOG" 2>&1
echo "=== orchestration start $(date) ==="

COMMON="task_name=real_pick_yellow_bottle camera_names=[top,wrist] \
action_offset=0 chunk_size=30 hidden_dim=512 dim_feedforward=3200 no_encoder=false kl_weight=10 \
amp=true num_workers=8 batch_size=128 batches_per_epoch=16 lr=1e-4 num_epochs=2000 \
save_every=200 deploy_every=50 resume_every=90 action_repr=absolute qpos_dropout=0.0"

VAL='val_episode_ids=[0,3,6,9,19,21,23,24,39]'

launch () {  # launch <name> <dataset_dir> <num_episodes> <extra overrides...>
  local name=$1; local dset=$2; local nep=$3; shift 3
  mkdir -p "$OUT/logs"
  if pgrep -f "ckpt_dir $OUT/$name " > /dev/null; then
    echo "$name already running -- attaching, not relaunching"
    return
  fi
  local R=""
  [ -f "$OUT/$name/train_state.pt" ] && R="resume=true" && echo "resuming $name"
  nohup $PY imitate_episodes.py --ckpt_dir "$OUT/$name" $COMMON \
      dataset_dir=$dset num_episodes=$nep $R "$@" \
      >> "$OUT/logs/$name.log" 2>&1 &
  disown
  echo "launched $name pid-unused"
}

wait_for () {  # wait_for <name>
  local name=$1
  while pgrep -f "ckpt_dir $OUT/$name " > /dev/null; do sleep 30; done
  echo "$name done $(date)"
}

launch d1_good41_absolute_dropout0     data/real_pick_yellow_bottle/resynced_good_41_tw240      41 $VAL
sleep 90   # stagger: norm-stats + deploy-set construction are CPU-heavy, don't overlap them
launch d2_scripted50_absolute_dropout0 data/real_pick_yellow_bottle/resynced_scripted_50c_tw240 50
sleep 90
launch d2_armonly_absolute_dropout0    data/real_pick_yellow_bottle/resynced_scripted_50c_tw240 50 'joint_ids=[0,1,2,3,4,5,6,7]'
sleep 90
launch d3_mix91_absolute_dropout0      data/real_pick_yellow_bottle/resynced_h41_s50c_tw240     91 $VAL

echo "waiting on all 4..."
wait_for d1_good41_absolute_dropout0
wait_for d2_scripted50_absolute_dropout0
wait_for d2_armonly_absolute_dropout0
wait_for d3_mix91_absolute_dropout0
echo "=== orchestration done $(date) ==="
