#!/usr/bin/env bash
# 顺序跑多组 optimize_film_params.py 实验；某一组报错/崩溃时自动跳过，继续跑下一组。
#
# 用法：
#   ./run_experiments.sh                                          # 前台跑
#   nohup ./run_experiments.sh > logs/run_experiments.out 2>&1 &   # 无人值守，挂后台
#
# 加实验的方法：在下面 EXPERIMENTS 数组里加一行，格式：
#   "名字|重复次数|完整命令"
# - 名字：只用于日志文件名，不要带空格、不要带 |
# - 重复次数：同一条命令原样重跑几次（不改 seed，用来观察随机性/不确定性），不需要重复写 1
# - 完整命令：整条可执行命令，直接照抄你要跑的命令即可，不用拆成 common/extra 参数
# 不想跑某组时，把那一整行注释掉（行首加 #）即可，不用删除。

set -uo pipefail  # 注意不能用 -e：某组失败要跳过而不是终止整个脚本

LOG_DIR="logs/experiments_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# 单组实验超时时间（秒），0 = 不限制。防止某次 LLM/网络调用卡死导致后面全部排队等待。
TIMEOUT_SECS=0

PY=/home/lab/miniforge3/envs/aloha/bin/python

# ---------------- 实验列表 ----------------
# 格式："名字|重复次数|完整命令"
EXPERIMENTS=(
  # qwen-vl LLM 优化，重复 5 次看 LLM 输出的不确定性
  # "qwen3-vl-32b-thinking|5|$PY optimize_film_params.py --ckpt results/sim_hmf_proto5_pick_place_v3/policy_best.ckpt --task_name sim_hmf_proto5_pick_place_v3_eval --film_pca_path tmp/film_pca/sim_hmf_proto5_pick_place_v3.npz --film_bottleneck_dim 8 --fixed_object_pose 0.1,0.65,0.66,0.1,0.65,0.8 --temporal_agg --method llm --llm_prompt_template prompts/num_optim_Pratyush_vlm.py --llm_maxiter 50 --save_videos --llm_model qwen3-vl-32b-thinking"
  
  # llm
  "qwen36-27b|5|$PY optimize_film_params.py   --ckpt results/sim_hmf_proto5_pick_place_v3/policy_best.ckpt   --task_name sim_hmf_proto5_pick_place_v3_eval   --film_pca_path tmp/film_pca/sim_hmf_proto5_pick_place_v3.npz --film_bottleneck_dim 8   --fixed_object_pose "0.1,0.65,0.66,0.1,0.65,0.8"  --temporal_agg  --method llm --llm_model qwen36-27b --llm_maxiter  50 --save_videos"
  
  # CMA-ES 优化，重复 5 次
  # "cma-sigma1.0|5|$PY optimize_film_params.py --ckpt results/sim_hmf_proto5_pick_place_v3/policy_best.ckpt --task_name sim_hmf_proto5_pick_place_v3_eval --film_pca_path tmp/film_pca/sim_hmf_proto5_pick_place_v3.npz --film_bottleneck_dim 8 --fixed_object_pose 0.1,0.65,0.66,0.1,0.65,0.8 --temporal_agg --method cma --cma_sigma0 1.0 --cma_maxiter 50"
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
