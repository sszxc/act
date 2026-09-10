#!/usr/bin/env bash
# Follow-up to run_longrun_datasets_20260909.sh's d2_scripted50_kl10_bs128: same scripted_50c_tw240
# dataset and hyperparameters, varying only the action/state representation --
#   d2_taskspace_kl10_bs128         action_repr=task_space (27-dim: palm pos+rot6d + 18 joints)
#   d2_armonly_kl10_bs128           joint_ids=[0..7] (8-dim: 6 arm + 2 wrist, delta as before)
#   d2_taskspace_armonly_kl10_bs128  both: task_space trimmed to arm-only (11-dim: palm pos+rot6d
#                                    + wrist yaw/pitch; the 16 finger dims are dropped from the
#                                    tail of the 27-dim task-space vector, not from the FK input,
#                                    which always needs the full 24 -- see utils._task_space_keep_idx)
#
# Requires imitate_episodes.py's task_space+joint_ids support (utils.py, this commit) and the
# ~/Honda_proto5_description symlink for forward_kinematics.py's FK to resolve on the host the
# same way it does inside the docker container.
#
# Idempotent: re-running resumes each run from its train_state.pt. Safe to kill and restart.
set -u
cd "$(dirname "$0")"
PY=/home/asu/miniconda3/envs/aloha/bin/python
OUT=results/longrun_datasets_20260909
mkdir -p $OUT/logs
EPOCHS=${EPOCHS:-6000}

COMMON="task_name=real_pick_yellow_bottle camera_names=[top,wrist] \
action_offset=0 qpos_dropout=0.5 chunk_size=30 no_encoder=false kl_weight=10 \
amp=true num_workers=8 batch_size=128 batches_per_epoch=16 lr=1e-4 num_epochs=$EPOCHS \
save_every=200 deploy_every=50 resume_every=200 \
dataset_dir=data/real_pick_yellow_bottle/scripted_50c_tw240 num_episodes=50"

launch () {  # launch <name> <extra overrides...>
  local name=$1; shift 1
  local R=""
  [ -f "$OUT/$name/train_state.pt" ] && R="resume=true" && echo "resuming $name"
  nohup $PY imitate_episodes.py --ckpt_dir "$OUT/$name" $COMMON $R "$@" \
      >> "$OUT/logs/$name.log" 2>&1 &
  echo "launched $name pid $!"
}

launch d2_taskspace_kl10_bs128        action_repr=task_space
sleep 90   # stagger: norm-stats (task_space's FK pass) + deploy-set construction are CPU-heavy
launch d2_armonly_kl10_bs128          action_repr=delta joint_ids=[0,1,2,3,4,5,6,7]
sleep 90
launch d2_taskspace_armonly_kl10_bs128 action_repr=task_space joint_ids=[0,1,2,3,4,5,6,7]
