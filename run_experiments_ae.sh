#!/usr/bin/env bash
# AE-bottleneck FiLM 版本的 sweep 实验，跟 run_experiments.sh 里 grasp_red_box 那 3 组 PCA sweep
# 参数完全一致（同一个 fixed_object_pose/shape/size、同一个 object grid、同一个 sweep_values），
# 只是把 --film_pca_path 换成 --film_bottleneck_method ae --film_ae_path，film_ae_path 由
# fit_film_ae.py 预先在 k=1 下各 fit 好（visual/memory/hs x ae_hidden=0/64 共 6 个 .pt）。
# 用法同 run_experiments.sh：
#   ./run_experiments_ae.sh
#   nohup ./run_experiments_ae.sh > logs/run_experiments_ae.out 2>&1 &

set -uo pipefail

LOG_DIR="logs/experiments_ae_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

TIMEOUT_SECS=0

PY=/home/lab/miniforge3/envs/aloha/bin/python

EXPERIMENTS=(
  "sweep-grasp_red_box-ae-visual-h0|1|$PY optimize_film_params.py   --ckpt results/sim_hmf_proto5_grasp_red_box/policy_best.ckpt   --task_name sim_hmf_proto5_grasp_red_box   --film_bottleneck_method ae --film_ae_path tmp/film_ae/sim_hmf_proto5_grasp_red_box_visual_k1_h0.pt   --film_target visual --film_bottleneck_dim 1   --fixed_object_shape box --fixed_object_size 0.03,0.03,0.05 --fixed_object_pose=-0.1,0.7,0.685  --object_sweep_x_values "[-0.05,-0.025,0,0.025,0.05]" --object_sweep_y_values "[-0.05,-0.025,0,0.025,0.05]" --temporal_agg   --method sweep --sweep_values "[-2,-1,0,1,2]"   --output_dir results/sim_hmf_proto5_grasp_red_box/icl_20260908-ae_pos_sweep_visual_h0 --save_videos"
  "sweep-grasp_red_box-ae-visual-h64|1|$PY optimize_film_params.py   --ckpt results/sim_hmf_proto5_grasp_red_box/policy_best.ckpt   --task_name sim_hmf_proto5_grasp_red_box   --film_bottleneck_method ae --film_ae_path tmp/film_ae/sim_hmf_proto5_grasp_red_box_visual_k1_h64.pt   --film_target visual --film_bottleneck_dim 1   --fixed_object_shape box --fixed_object_size 0.03,0.03,0.05 --fixed_object_pose=-0.1,0.7,0.685  --object_sweep_x_values "[-0.05,-0.025,0,0.025,0.05]" --object_sweep_y_values "[-0.05,-0.025,0,0.025,0.05]" --temporal_agg   --method sweep --sweep_values "[-2,-1,0,1,2]"   --output_dir results/sim_hmf_proto5_grasp_red_box/icl_20260908-ae_pos_sweep_visual_h64 --save_videos"
  "sweep-grasp_red_box-ae-memory-h0|1|$PY optimize_film_params.py   --ckpt results/sim_hmf_proto5_grasp_red_box/policy_best.ckpt   --task_name sim_hmf_proto5_grasp_red_box   --film_bottleneck_method ae --film_ae_path tmp/film_ae/sim_hmf_proto5_grasp_red_box_memory_k1_h0.pt   --film_target memory --film_bottleneck_dim 1   --fixed_object_shape box --fixed_object_size 0.03,0.03,0.05 --fixed_object_pose=-0.1,0.7,0.685  --object_sweep_x_values "[-0.05,-0.025,0,0.025,0.05]" --object_sweep_y_values "[-0.05,-0.025,0,0.025,0.05]" --temporal_agg   --method sweep --sweep_values "[-2,-1,0,1,2]"   --output_dir results/sim_hmf_proto5_grasp_red_box/icl_20260908-ae_pos_sweep_memory_h0 --save_videos"
  "sweep-grasp_red_box-ae-memory-h64|1|$PY optimize_film_params.py   --ckpt results/sim_hmf_proto5_grasp_red_box/policy_best.ckpt   --task_name sim_hmf_proto5_grasp_red_box   --film_bottleneck_method ae --film_ae_path tmp/film_ae/sim_hmf_proto5_grasp_red_box_memory_k1_h64.pt   --film_target memory --film_bottleneck_dim 1   --fixed_object_shape box --fixed_object_size 0.03,0.03,0.05 --fixed_object_pose=-0.1,0.7,0.685  --object_sweep_x_values "[-0.05,-0.025,0,0.025,0.05]" --object_sweep_y_values "[-0.05,-0.025,0,0.025,0.05]" --temporal_agg   --method sweep --sweep_values "[-2,-1,0,1,2]"   --output_dir results/sim_hmf_proto5_grasp_red_box/icl_20260908-ae_pos_sweep_memory_h64 --save_videos"
  "sweep-grasp_red_box-ae-hs-h0|1|$PY optimize_film_params.py   --ckpt results/sim_hmf_proto5_grasp_red_box/policy_best.ckpt   --task_name sim_hmf_proto5_grasp_red_box   --film_bottleneck_method ae --film_ae_path tmp/film_ae/sim_hmf_proto5_grasp_red_box_hs_k1_h0.pt   --film_target hs --film_bottleneck_dim 1   --fixed_object_shape box --fixed_object_size 0.03,0.03,0.05 --fixed_object_pose=-0.1,0.7,0.685  --object_sweep_x_values "[-0.05,-0.025,0,0.025,0.05]" --object_sweep_y_values "[-0.05,-0.025,0,0.025,0.05]" --temporal_agg   --method sweep --sweep_values "[-2,-1,0,1,2]"   --output_dir results/sim_hmf_proto5_grasp_red_box/icl_20260908-ae_pos_sweep_hs_h0 --save_videos"
  "sweep-grasp_red_box-ae-hs-h64|1|$PY optimize_film_params.py   --ckpt results/sim_hmf_proto5_grasp_red_box/policy_best.ckpt   --task_name sim_hmf_proto5_grasp_red_box   --film_bottleneck_method ae --film_ae_path tmp/film_ae/sim_hmf_proto5_grasp_red_box_hs_k1_h64.pt   --film_target hs --film_bottleneck_dim 1   --fixed_object_shape box --fixed_object_size 0.03,0.03,0.05 --fixed_object_pose=-0.1,0.7,0.685  --object_sweep_x_values "[-0.05,-0.025,0,0.025,0.05]" --object_sweep_y_values "[-0.05,-0.025,0,0.025,0.05]" --temporal_agg   --method sweep --sweep_values "[-2,-1,0,1,2]"   --output_dir results/sim_hmf_proto5_grasp_red_box/icl_20260908-ae_pos_sweep_hs_h64 --save_videos"
)

PASSED=()
FAILED=()

for entry in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r name repeats cmd <<<"$entry"
  repeats="${repeats:-1}"

  for run in $(seq 1 "$repeats"); do
    if [[ "$repeats" -gt 1 ]]; then
      run_name="${name}_run${run}"
    else
      run_name="$name"
    fi
    log_file="$LOG_DIR/${run_name}.log"

    echo "==== [$(date '+%F %T')] 开始实验: $run_name (log: $log_file) ===="
    echo "命令: $cmd" | tee "$log_file"

    if [[ "$TIMEOUT_SECS" -gt 0 ]]; then
      timeout "$TIMEOUT_SECS" bash -c "$cmd" >>"$log_file" 2>&1
    else
      bash -c "$cmd" >>"$log_file" 2>&1
    fi
    status=$?

    if [[ $status -eq 0 ]]; then
      echo "==== [$(date '+%F %T')] 实验成功: $run_name ===="
      PASSED+=("$run_name")
    else
      echo "==== [$(date '+%F %T')] 实验失败 (exit=$status)，跳过并继续下一组: $run_name ===="
      FAILED+=("$run_name (exit=$status)")
    fi
    echo
  done
done

echo "================ 汇总 ================"
echo "成功 (${#PASSED[@]}): ${PASSED[*]:-无}"
echo "失败 (${#FAILED[@]}): ${FAILED[*]:-无}"
echo "日志目录: $LOG_DIR"
