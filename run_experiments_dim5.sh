#!/usr/bin/env bash
# 前 5 个 FiLM 瓶颈维度 (dim 0-4) 的 sweep：PCA (3 个 target) + AE (3 个 target x h0/h64)，
# 共 9 组。跟之前 k=1 的 sweep 不同，这次不对 object position 做 sweep（不传
# --object_sweep_x_values/--object_sweep_y_values，默认 "0"，物体固定在 --fixed_object_pose）。
# k=5 时每组 10 个 FiLM 维度 (gamma_0..4, beta_0..4) x 5 个 sweep_values = 50 次 rollout。
#
# 跑之前先跑 fit_film_ae_dim5.sh 生成 k=5 的 AE checkpoint（AE 的 latent 宽度是 fit 时定死的，
# 不能像 PCA 一样在搜索时切片，k=1 fit 的 checkpoint 不能给 k=5 用）。
#
# 用法：
#   ./run_experiments_dim5.sh
#   nohup ./run_experiments_dim5.sh > logs/run_experiments_dim5.out 2>&1 &
#
# 某一组失败会跳过继续跑下一组；不想跑某组时把对应行注释掉即可。

set -uo pipefail

MAX_PARALLEL=3   # 同时跑几组；单卡+32核，rollout 是 CPU(mujoco)瓶颈不是 GPU 瓶颈，3 组并发够用

LOG_DIR="logs/experiments_dim5_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

PY=/home/lab/miniforge3/envs/aloha/bin/python
CKPT=results/sim_hmf_proto5_grasp_red_box/policy_best.ckpt
TASK=sim_hmf_proto5_grasp_red_box
OUT_TS=$(date +%Y%m%d-%H%M%S)
COMMON="--ckpt $CKPT --task_name $TASK --film_bottleneck_dim 5 --fixed_object_shape box --fixed_object_size 0.03,0.03,0.05 --fixed_object_pose=-0.1,0.7,0.685 --temporal_agg --method sweep --sweep_values [-2,-1,0,1,2] --save_videos"

EXPERIMENTS=(
  # PCA sweep, k=5, target visual/memory/hs (max_k=64 in the existing .npz files, no re-fit needed)
  "pca-visual|$PY optimize_film_params.py $COMMON --film_pca_path tmp/film_pca/${TASK}_visual.npz --film_target visual --output_dir results/${TASK}/icl_${OUT_TS}_dim5_sweep_pca_visual"
  "pca-memory|$PY optimize_film_params.py $COMMON --film_pca_path tmp/film_pca/${TASK}_memory.npz --film_target memory --output_dir results/${TASK}/icl_${OUT_TS}_dim5_sweep_pca_memory"
  "pca-hs|$PY optimize_film_params.py $COMMON --film_pca_path tmp/film_pca/${TASK}_hs.npz --film_target hs --output_dir results/${TASK}/icl_${OUT_TS}_dim5_sweep_pca_hs"

  # AE sweep, k=5, target visual/memory/hs x ae_hidden 0(线性)/64(非线性)
  # (requires fit_film_ae_dim5.sh to have produced these checkpoints first)
  "ae-visual-h0|$PY optimize_film_params.py $COMMON --film_bottleneck_method ae --film_ae_path tmp/film_ae/${TASK}_visual_k5_h0.pt --film_target visual --output_dir results/${TASK}/icl_${OUT_TS}_dim5_sweep_ae_visual_h0"
  "ae-visual-h64|$PY optimize_film_params.py $COMMON --film_bottleneck_method ae --film_ae_path tmp/film_ae/${TASK}_visual_k5_h64.pt --film_target visual --output_dir results/${TASK}/icl_${OUT_TS}_dim5_sweep_ae_visual_h64"
  "ae-memory-h0|$PY optimize_film_params.py $COMMON --film_bottleneck_method ae --film_ae_path tmp/film_ae/${TASK}_memory_k5_h0.pt --film_target memory --output_dir results/${TASK}/icl_${OUT_TS}_dim5_sweep_ae_memory_h0"
  "ae-memory-h64|$PY optimize_film_params.py $COMMON --film_bottleneck_method ae --film_ae_path tmp/film_ae/${TASK}_memory_k5_h64.pt --film_target memory --output_dir results/${TASK}/icl_${OUT_TS}_dim5_sweep_ae_memory_h64"
  "ae-hs-h0|$PY optimize_film_params.py $COMMON --film_bottleneck_method ae --film_ae_path tmp/film_ae/${TASK}_hs_k5_h0.pt --film_target hs --output_dir results/${TASK}/icl_${OUT_TS}_dim5_sweep_ae_hs_h0"
  "ae-hs-h64|$PY optimize_film_params.py $COMMON --film_bottleneck_method ae --film_ae_path tmp/film_ae/${TASK}_hs_k5_h64.pt --film_target hs --output_dir results/${TASK}/icl_${OUT_TS}_dim5_sweep_ae_hs_h64"
)

run_one() {
  local name="$1" cmd="$2"
  local log_file="$LOG_DIR/${name}.log"
  echo "命令: $cmd" >"$log_file"
  bash -c "$cmd" >>"$log_file" 2>&1
  echo $? >"$LOG_DIR/${name}.status"
}

active=0
for entry in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r name cmd <<<"$entry"
  echo "==== [$(date '+%F %T')] 开始实验: $name (log: $LOG_DIR/${name}.log) ===="
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
for entry in "${EXPERIMENTS[@]}"; do
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
