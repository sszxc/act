#!/usr/bin/env bash
# Re-run of run_longrun_taskspace_20260909.sh's 3 variants on resynced_scripted_50c_tw240.
# Fresh checkpoints, "resynced_" names; run after run_longrun_datasets_resynced_20260909.sh
# finishes (2nd wave of 3 -- one GPU, 12h budget split into two ~6h waves).
#
# EPOCHS default (170) targets ~6h at 3-way concurrency; slightly more conservative than the
# datasets wave to leave headroom for task_space's per-sample FK cost. Override with EPOCHS=N.
set -u
cd "$(dirname "$0")"
PY=/home/asu/miniconda3/envs/aloha/bin/python
OUT=results/resynced_longrun_datasets_20260909
mkdir -p $OUT/logs
EPOCHS=${EPOCHS:-170}

COMMON="task_name=real_pick_yellow_bottle camera_names=[top,wrist] \
action_offset=0 qpos_dropout=0.5 chunk_size=30 no_encoder=false kl_weight=10 \
amp=true num_workers=8 batch_size=128 batches_per_epoch=16 lr=1e-4 num_epochs=$EPOCHS \
save_every=60 deploy_every=50 resume_every=90 \
dataset_dir=data/real_pick_yellow_bottle/resynced_scripted_50c_tw240 num_episodes=50"

launch () {  # launch <name> <extra overrides...>
  local name=$1; shift 1
  local R=""
  [ -f "$OUT/$name/train_state.pt" ] && R="resume=true" && echo "resuming $name"
  nohup $PY imitate_episodes.py --ckpt_dir "$OUT/$name" $COMMON $R "$@" \
      >> "$OUT/logs/$name.log" 2>&1 &
  echo "launched $name pid $!"
}

launch resynced_d2_taskspace_kl10_bs128        action_repr=task_space
sleep 90   # stagger: norm-stats (task_space's FK pass) + deploy-set construction are CPU-heavy
launch resynced_d2_armonly_kl10_bs128          action_repr=delta joint_ids=[0,1,2,3,4,5,6,7]
sleep 90
launch resynced_d2_taskspace_armonly_kl10_bs128 action_repr=task_space joint_ids=[0,1,2,3,4,5,6,7]
