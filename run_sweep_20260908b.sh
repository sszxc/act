#!/usr/bin/env bash
# Phase B of the 2026-09-08 sweep.
#
# b1: the small-batch control the user asked for (batch 8 / lr 1e-5). At 256 batches/epoch it
#     runs 3.2x slower per epoch than batch 64, so it gets 600 epochs, not 1200 -- r0 plateaus
#     by epoch 450, and r0's own history supplies the batch-64 value at epoch 600 for an
#     equal-samples comparison. The wall-clock cost of each is reported alongside.
# b2/b3: action_stride -- the same 1 s chunk, but commanded at 10 Hz instead of 30 Hz. This is
#     the lead the 2026-09-04 report left untested (its §7): per-step arm motion is 0.0028 rad
#     while the best command error is 0.003-0.005 rad, so at 30 Hz the signal sits under the
#     error floor. stride 3 makes every commanded step 3x larger for the same wall-clock span.
#     Deploying one of these means holding each target for 3 frames (100 ms).
set -u
cd "$(dirname "$0")"
PY=/home/asu/miniconda3/envs/aloha/bin/python
OUT=results/sweep_20260908
mkdir -p $OUT/logs
H=data/real_pick_yellow_bottle/good_41_tw240
ARM='joint_ids=[0,1,2,3,4,5,6,7]'
COMMON="task_name=real_pick_yellow_bottle camera_names=[top,wrist] \
action_repr=delta action_offset=0 qpos_dropout=0.5 amp=true num_workers=12 \
save_every=0 deploy_every=25 val_episode_ids=[0,3,6,9,19,21,23,24,39] \
no_encoder=true dataset_dir=$H num_episodes=41 $ARM chunk_size=30 \
batch_size=64 batches_per_epoch=32 lr=5e-5"

run () {
  local name=$1; shift
  if [ -d "$OUT/$name" ]; then echo "== skip $name (exists)"; return; fi
  echo "== $(date +%H:%M:%S) start $name"
  $PY imitate_episodes.py --ckpt_dir "$OUT/$name" $COMMON "$@" > "$OUT/logs/$name.log" 2>&1
  local rc=$?
  echo "== $(date +%H:%M:%S) end   $name rc=$rc  $(grep -h 'Best track ckpt' $OUT/logs/$name.log | tail -1)"
  [ $rc -ne 0 ] && tail -20 "$OUT/logs/$name.log"
  return 0
}

run r5_arm8_c30_noenc_bs8   batch_size=8 batches_per_epoch=256 lr=1e-5 num_epochs=600
run rs1_arm8_c10_st3_noenc  chunk_size=10 action_stride=3 num_epochs=1200   # 1.0 s span, 10 Hz
# Traded in for seed 2 of the centre point. r0/r9 already pin the noise floor on the LOW side
# of the main effect (0.7% apart); what is still n=1 is the HIGH side, so replicate the leading
# condition instead of the one already replicated.
run r3b_arm8_c30_kl10_s1    no_encoder=false kl_weight=10 seed=1 num_epochs=1200
# Only if the clock allows: are 24 joints and the CVAE additive, or is 0.227 a ceiling?
run r8_full24_c30_kl10      joint_ids=null no_encoder=false kl_weight=10 num_epochs=1200
echo "== $(date +%H:%M:%S) PHASE B DONE"
