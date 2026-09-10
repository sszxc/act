#!/usr/bin/env bash
# Rebuild good_41/scripted_50c/h41_s50c (raw -> clean -> tw240) once the data-migration agent
# delivers resynced_good_41 (41ep) and resynced_scripted_50 (50ep) directly. Falls back to
# building those two from their resynced_* components (mirroring good_41/scripted_50's existing
# symlink mapping -- see mirror_symlink_dir.py) if the merged dirs aren't there yet.
set -euo pipefail
cd "$(dirname "$0")"
PY=/home/asu/miniconda3/envs/aloha/bin/python
ROOT=data/real_pick_yellow_bottle

count () { ls "$ROOT/$1"/episode_*.hdf5 2>/dev/null | wc -l; }
check_count () { local n; n=$(count "$1"); [ "$n" = "$2" ] || { echo "ABORT: $ROOT/$1 has $n episodes, expected $2"; exit 1; }; }

if [ "$(count resynced_good_41)" = 41 ]; then
  echo "resynced_good_41 already present (41ep), using as-is"
else
  check_count resynced_good_0901_c20 20
  check_count resynced_good_0902_c21 21
  $PY mirror_symlink_dir.py --src $ROOT/good_41 --out $ROOT/resynced_good_41
  check_count resynced_good_41 41
fi

if [ "$(count resynced_scripted_50)" = 50 ]; then
  echo "resynced_scripted_50 already present (50ep), using as-is"
else
  check_count resynced_scripted_31 31
  check_count resynced_scripted_0904_c19 19
  $PY mirror_symlink_dir.py --src $ROOT/scripted_50 --out $ROOT/resynced_scripted_50
  check_count resynced_scripted_50 50
fi

$PY clean_pauses.py --data_dir $ROOT/resynced_scripted_50 --out_dir $ROOT/resynced_scripted_50_clean
check_count resynced_scripted_50_clean 50

$PY build_image_cache.py --src $ROOT/resynced_good_41 --out $ROOT/resynced_good_41_tw240 \
    --cams top wrist --size 240 320
$PY build_image_cache.py --src $ROOT/resynced_scripted_50_clean --out $ROOT/resynced_scripted_50c_tw240 \
    --cams top wrist --size 240 320
check_count resynced_good_41_tw240 41
check_count resynced_scripted_50c_tw240 50

$PY mirror_symlink_dir.py --src $ROOT/h41_s50c_tw240 --out $ROOT/resynced_h41_s50c_tw240
check_count resynced_h41_s50c_tw240 91

echo "All resynced_* datasets ready."
