#!/usr/bin/env bash
# 为 k=5 的 AE sweep 预先 fit AE checkpoint：(ae_hidden=0 线性 / 64 非线性) x (visual/memory/hs)，
# 共 6 个。AE 的 latent 宽度在 fit 时就固定死了（不像 PCA 能在搜索时切片），所以 k=1 时 fit 的
# checkpoint 不能拿来给 k=5 用，必须重新 fit。跑完这个脚本之后再跑 run_experiments_dim5.sh。
#
# 用法：
#   ./fit_film_ae_dim5.sh
#   nohup ./fit_film_ae_dim5.sh > logs/fit_film_ae_dim5.out 2>&1 &

set -uo pipefail

MAX_PARALLEL=3   # 同时 fit 几个；单卡+32核，够用

LOG_DIR="logs/fit_film_ae_dim5_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR" tmp/film_ae

PY=/home/lab/miniforge3/envs/aloha/bin/python
CKPT=results/sim_hmf_proto5_grasp_red_box/policy_best.ckpt
TASK=sim_hmf_proto5_grasp_red_box

FITS=(
  "visual-h0|$PY fit_film_ae.py --ckpt $CKPT --task_name $TASK --target visual --k 5 --ae_hidden 0  --output tmp/film_ae/${TASK}_visual_k5_h0.pt"
  "visual-h64|$PY fit_film_ae.py --ckpt $CKPT --task_name $TASK --target visual --k 5 --ae_hidden 64 --output tmp/film_ae/${TASK}_visual_k5_h64.pt"
  "memory-h0|$PY fit_film_ae.py --ckpt $CKPT --task_name $TASK --target memory --k 5 --ae_hidden 0  --output tmp/film_ae/${TASK}_memory_k5_h0.pt"
  "memory-h64|$PY fit_film_ae.py --ckpt $CKPT --task_name $TASK --target memory --k 5 --ae_hidden 64 --output tmp/film_ae/${TASK}_memory_k5_h64.pt"
  "hs-h0|$PY fit_film_ae.py --ckpt $CKPT --task_name $TASK --target hs --k 5 --ae_hidden 0  --output tmp/film_ae/${TASK}_hs_k5_h0.pt"
  "hs-h64|$PY fit_film_ae.py --ckpt $CKPT --task_name $TASK --target hs --k 5 --ae_hidden 64 --output tmp/film_ae/${TASK}_hs_k5_h64.pt"
)

run_one() {
  local name="$1" cmd="$2"
  local log_file="$LOG_DIR/${name}.log"
  echo "命令: $cmd" >"$log_file"
  bash -c "$cmd" >>"$log_file" 2>&1
  echo $? >"$LOG_DIR/${name}.status"
}

active=0
for entry in "${FITS[@]}"; do
  IFS='|' read -r name cmd <<<"$entry"
  echo "==== [$(date '+%F %T')] 开始 fit: $name ===="
  run_one "$name" "$cmd" &
  active=$((active + 1))
  if (( active >= MAX_PARALLEL )); then
    wait -n
    active=$((active - 1))
  fi
done
wait

PASSED=()
FAILED=()
for entry in "${FITS[@]}"; do
  IFS='|' read -r name _ <<<"$entry"
  status=$(cat "$LOG_DIR/${name}.status" 2>/dev/null || echo "?")
  if [[ "$status" == "0" ]]; then
    PASSED+=("$name")
  else
    FAILED+=("$name (exit=$status)")
  fi
done

echo "================ 汇总 ================"
echo "成功 (${#PASSED[@]}): ${PASSED[*]:-无}"
echo "失败 (${#FAILED[@]}): ${FAILED[*]:-无}"
echo "日志目录: $LOG_DIR"
