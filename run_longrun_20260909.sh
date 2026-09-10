#!/usr/bin/env bash
# Extra-long full-24-joint runs, 6000 epochs x 2048 samples = 12.3M samples / 96k steps
# (5x the 2026-09-08 sweep). Two variants, because the sweep left the top group unresolved:
# 24 joints without the CVAE vs 24 joints with it, both at batch 128 / lr 1e-4 -- the only
# setting measured to keep improving late instead of decaying toward "hold still".
#
# Idempotent: re-running resumes each run from its train_state.pt. Safe to kill and restart.
set -u
cd "$(dirname "$0")"
PY=/home/asu/miniconda3/envs/aloha/bin/python
OUT=results/longrun_20260909
mkdir -p $OUT/logs
EPOCHS=${EPOCHS:-6000}

COMMON="task_name=real_pick_yellow_bottle camera_names=[top,wrist] \
dataset_dir=data/real_pick_yellow_bottle/good_41_tw240 num_episodes=41 \
action_repr=delta action_offset=0 qpos_dropout=0.5 chunk_size=30 amp=true num_workers=8 \
batch_size=128 batches_per_epoch=16 lr=1e-4 num_epochs=$EPOCHS \
save_every=0 deploy_every=50 resume_every=200 \
val_episode_ids=[0,3,6,9,19,21,23,24,39]"

launch () {  # launch <name> <overrides...>
  local name=$1; shift
  local R=""
  [ -f "$OUT/$name/train_state.pt" ] && R="resume=true" && echo "resuming $name"
  nohup $PY imitate_episodes.py --ckpt_dir "$OUT/$name" $COMMON $R "$@" \
      >> "$OUT/logs/$name.log" 2>&1 &
  echo "launched $name pid $!"
}

launch xl1_full24_noenc_bs128 no_encoder=true
sleep 90   # stagger: norm-stats + deploy-set construction are CPU-heavy, don't overlap them
launch xl2_full24_kl10_bs128  no_encoder=false kl_weight=10
