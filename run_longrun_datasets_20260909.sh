#!/usr/bin/env bash
# Long runs of the established best config -- chunk 30 (1.0 s), kl_weight=10 (CVAE kept),
# batch 128 / lr 1e-4 -- across three real-robot datasets in parallel: scripted-only,
# human-only, and the human+scripted mix. Same hyperparameters as results/longrun_20260909's
# xl2 (full 24 joints + CVAE + batch 128), the only setting the 2026-09-08 sweep measured to
# keep improving late instead of decaying toward "hold still" (see results/sweep_20260908/REPORT.md §Q4/Q5).
#
# 6000 epochs x 2048 samples/epoch = 12.3M samples / 96k steps -- 10x r5_arm8_c30_noenc_bs8's
# exposure (600 epochs x 256 batches/epoch x 8 = 1.23M samples / 153.6k steps) and well past the
# point (sweep epoch 1200 = 2.46M samples) where batch 128 was still gaining, not decaying.
#
# Datasets use the pause-stripped/pre-resized (top+wrist, 240x320) caches rather than the raw
# good_41/scripted_50 dirs: raw scripted data collapses commanded motion 18x (7ddd849), and the
# tw240 caches are bit-identical content otherwise (build_image_cache.py), just faster to read.
#
# save_every=200: keep periodic policy_epoch_N.ckpt snapshots (~336MB each) across the run, not
# just best/last. 30 snapshots/run x 3 runs x 336MB =~ 30GB, well inside free disk.
#
# Idempotent: re-running resumes each run from its train_state.pt. Safe to kill and restart.
set -u
cd "$(dirname "$0")"
PY=/home/asu/miniconda3/envs/aloha/bin/python
OUT=results/longrun_datasets_20260909
mkdir -p $OUT/logs
EPOCHS=${EPOCHS:-6000}

COMMON="task_name=real_pick_yellow_bottle camera_names=[top,wrist] \
action_repr=delta action_offset=0 qpos_dropout=0.5 chunk_size=30 no_encoder=false kl_weight=10 \
amp=true num_workers=8 batch_size=128 batches_per_epoch=16 lr=1e-4 num_epochs=$EPOCHS \
save_every=200 deploy_every=50 resume_every=200"

# 9 held-out human episodes (indices 0,3,6,9,19,21,23,24,39 of good_41); also valid inside the
# mix, whose first 41 entries are good_41 in the same order (verified via manifest symlinks).
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

launch d1_good41_kl10_bs128     data/real_pick_yellow_bottle/good_41_tw240      41 $VAL
sleep 90   # stagger: norm-stats + deploy-set construction are CPU-heavy, don't overlap them
launch d2_scripted50_kl10_bs128 data/real_pick_yellow_bottle/scripted_50c_tw240 50
sleep 90
launch d3_mix91_kl10_bs128      data/real_pick_yellow_bottle/h41_s50c_tw240     91 $VAL
