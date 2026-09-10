#!/usr/bin/env bash
# 2026-09-08 sweep: finger-free control, 1-second chunks, CVAE strength, batch/lr scaling,
# and (above all) training long past plateau. Design notes: results/sweep_20260908/PLAN.md
#
# Sequential by design: the GPU sits at 97% util on a single run, so two concurrent runs just
# split the same throughput. Resumable -- any run whose ckpt_dir already exists is skipped.
set -u
cd "$(dirname "$0")"
PY=/home/asu/miniconda3/envs/aloha/bin/python
OUT=results/sweep_20260908
mkdir -p $OUT/logs

H=data/real_pick_yellow_bottle/good_41_tw240          # 41 human teleop
HS=data/real_pick_yellow_bottle/h41_s50c_tw240        # + 50 pause-stripped scripted
HSR=data/real_pick_yellow_bottle/h41_s50raw_tw240     # + 50 raw scripted

# Every epoch is 2048 samples regardless of batch size (batches_per_epoch is set per run), so
# num_epochs is directly comparable across the batch-size axis.
EPOCHS=${EPOCHS:-1200}
ARM='joint_ids=[0,1,2,3,4,5,6,7]'

COMMON="task_name=real_pick_yellow_bottle camera_names=[top,wrist] \
action_repr=delta action_offset=0 qpos_dropout=0.5 amp=true num_workers=12 \
save_every=0 deploy_every=25 val_episode_ids=[0,3,6,9,19,21,23,24,39] \
num_epochs=$EPOCHS"

# Centre point: arm+wrist only, 1.0 s chunk, no CVAE, batch 64 / lr 5e-5, human data.
CENTRE="$ARM chunk_size=30 no_encoder=true batch_size=64 batches_per_epoch=32 lr=5e-5 \
dataset_dir=$H num_episodes=41"

run () {  # run <name> <overrides...>
  local name=$1; shift
  if [ -d "$OUT/$name" ]; then echo "== skip $name (exists)"; return; fi
  echo "== $(date +%H:%M:%S) start $name"
  $PY imitate_episodes.py --ckpt_dir "$OUT/$name" $COMMON "$@" \
      > "$OUT/logs/$name.log" 2>&1
  local rc=$?
  echo "== $(date +%H:%M:%S) end   $name rc=$rc  $(grep -h 'Best track ckpt' $OUT/logs/$name.log | tail -1)"
  [ $rc -ne 0 ] && tail -20 "$OUT/logs/$name.log"
  return 0
}

# ---- ordered by information value: if the budget runs out, the tail is what is lost ----
run r0_arm8_c30_noenc        $CENTRE
run r1_full24_c30_noenc      $CENTRE joint_ids=null
run r3_arm8_c30_kl10         $CENTRE no_encoder=false kl_weight=10
run r5_arm8_c30_noenc_bs8    $ARM chunk_size=30 no_encoder=true batch_size=8 batches_per_epoch=256 lr=1e-5 dataset_dir=$H num_episodes=41
run r2_arm8_c50_noenc        $CENTRE chunk_size=50
run r9_arm8_c30_noenc_s1     $CENTRE seed=1
run r4_arm8_c30_kl100        $CENTRE no_encoder=false kl_weight=100
run r7_arm8_c30_noenc_mixc   $ARM chunk_size=30 no_encoder=true batch_size=64 batches_per_epoch=32 lr=5e-5 dataset_dir=$HS num_episodes=91
run r6_arm8_c30_noenc_bs128  $ARM chunk_size=30 no_encoder=true batch_size=128 batches_per_epoch=16 lr=1e-4 dataset_dir=$H num_episodes=41
run r10_arm8_c30_noenc_s2    $CENTRE seed=2
run r8_full24_c30_kl10       $CENTRE joint_ids=null no_encoder=false kl_weight=10
run r11_arm8_c30_noenc_nodrop $CENTRE qpos_dropout=0.0
run r12_arm8_c15_noenc       $CENTRE chunk_size=15
run r13_arm8_c30_noenc_mixraw $ARM chunk_size=30 no_encoder=true batch_size=64 batches_per_epoch=32 lr=5e-5 dataset_dir=$HSR num_episodes=91
echo "== $(date +%H:%M:%S) SWEEP DONE"
