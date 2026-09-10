#!/usr/bin/env bash
# Re-run of run_longrun_datasets_20260909.sh's d1/d2/d3 on the resynced datasets (the
# pre-resync copies hit a real bug found 2026-09-09; see prepare_resynced_data.sh).
# Fresh checkpoints (new ckpt_dir, "resynced_" names) -- old results/longrun_datasets_20260909
# is left untouched, not resumed from (its weights were fit on the bad data).
#
# EPOCHS default (180) targets ~6h wall time at 3-way concurrency on 1 GPU, measured from the
# pre-resync run: ~6.7-7.2s/it, 16 it/epoch -> ~110-115s/epoch. Override with EPOCHS=N env var.
set -u
cd "$(dirname "$0")"
PY=/home/asu/miniconda3/envs/aloha/bin/python
OUT=results/resynced_longrun_datasets_20260909
mkdir -p $OUT/logs
EPOCHS=${EPOCHS:-180}

COMMON="task_name=real_pick_yellow_bottle camera_names=[top,wrist] \
action_repr=delta action_offset=0 qpos_dropout=0.5 chunk_size=30 no_encoder=false kl_weight=10 \
amp=true num_workers=8 batch_size=128 batches_per_epoch=16 lr=1e-4 num_epochs=$EPOCHS \
save_every=60 deploy_every=50 resume_every=90"

VAL='val_episode_ids=[0,3,6,9,19,21,23,24,39]'

launch () {  # launch <name> <dataset_dir> <num_episodes> <extra overrides...>
  local name=$1; local dset=$2; local nep=$3; shift 3
  local R=""
  [ -f "$OUT/$name/train_state.pt" ] && R="resume=true" && echo "resuming $name"
  nohup $PY imitate_episodes.py --ckpt_dir "$OUT/$name" $COMMON \
      dataset_dir=$dset num_episodes=$nep $R "$@" \
      >> "$OUT/logs/$name.log" 2>&1 &
  echo "launched $name pid $!"
}

launch resynced_d1_good41_kl10_bs128     data/real_pick_yellow_bottle/resynced_good_41_tw240      41 $VAL
sleep 90   # stagger: norm-stats + deploy-set construction are CPU-heavy, don't overlap them
launch resynced_d2_scripted50_kl10_bs128 data/real_pick_yellow_bottle/resynced_scripted_50c_tw240 50
sleep 90
launch resynced_d3_mix91_kl10_bs128      data/real_pick_yellow_bottle/resynced_h41_s50c_tw240     91 $VAL
