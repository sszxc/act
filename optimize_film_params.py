#!/usr/bin/env python3
"""
在固定物体初始位姿下，用 CMA-ES 或 ARS 搜索 ACT 策略中 FiLM（visual gamma/beta）参数，以最大化单局 episode_return。

依赖：
  - 必选：numpy, torch, matplotlib
  - CMA-ES：pip install cma（仅 --method cma 时需要）

示例：
  python optimize_film_params.py --ckpt /path/to/policy_best.ckpt --task_name sim_transfer_cube_human \\
    --fixed_object_pose "0.1,0.5,0.05,1,0,0,0" --method ars --ars_iters 3 --ars_pairs 2 --output_dir tmp/film_search

  python optimize_film_params.py --ckpt ... --method cma --cma_maxiter 5 --cma_popsize 8 ...
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

# 与 imitate_episodes 一致：无显示器时 EGL
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

from constants import DT, SIM_TASK_CONFIGS, DEFAULT_STATE_DIM
from policy import ACTPolicy
from imitate_episodes import make_policy, set_seed
from sim_env import make_sim_env


def _parse_float_list(s: str) -> np.ndarray:
    s = str(s).strip()
    if s.startswith("["):
        arr = json.loads(s)
    else:
        arr = [float(x) for x in s.split(",")]
    return np.asarray(arr, dtype=np.float64)


def _parse_latent_z(s: str | None, latent_z_dim: int):
    if s is None or str(s).strip() == "" or str(s).lower() in ("none", "null"):
        return None
    arr = _parse_float_list(s)
    if arr.size != latent_z_dim:
        raise ValueError(f"latent_z_sample 维度 {arr.size} != {latent_z_dim}")
    return torch.tensor(arr, dtype=torch.float32).cuda()


def _load_policy_and_stats(
    ckpt_path: Path,
    stats_path: Path | None,
    policy_class: str,
    policy_config: dict,
):
    ckpt_path = Path(ckpt_path).resolve()
    if stats_path is None:
        stats_path = ckpt_path.parent / "dataset_stats.pkl"
    else:
        stats_path = Path(stats_path).resolve()
    policy = make_policy(policy_class, policy_config)
    state_dict = torch.load(ckpt_path, map_location="cuda")
    loading_status = policy.load_state_dict(state_dict, strict=False)
    allowed_missing = {"model.visual_film_gamma", "model.visual_film_beta"}
    missing = set(getattr(loading_status, "missing_keys", []))
    unexpected = set(getattr(loading_status, "unexpected_keys", []))
    if unexpected or (missing - allowed_missing):
        raise RuntimeError(f"load_state_dict failed: {loading_status}")
    policy.cuda()
    policy.eval()
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    return policy, stats, str(ckpt_path)


def _film_theta_from_policy(policy: ACTPolicy) -> np.ndarray:
    g = policy.model.visual_film_gamma.detach().float().cpu().numpy()
    b = policy.model.visual_film_beta.detach().float().cpu().numpy()
    return np.concatenate([g, b], axis=0)


def _apply_film_theta(policy: ACTPolicy, theta: np.ndarray, hidden_dim: int):
    g = torch.from_numpy(theta[:hidden_dim]).to(
        device=policy.model.visual_film_gamma.device,
        dtype=policy.model.visual_film_gamma.dtype,
    )
    b = torch.from_numpy(theta[hidden_dim : 2 * hidden_dim]).to(
        device=policy.model.visual_film_beta.device,
        dtype=policy.model.visual_film_beta.dtype,
    )
    with torch.no_grad():
        policy.model.visual_film_gamma.copy_(g)
        policy.model.visual_film_beta.copy_(b)


def _split_film_theta_batch(theta_batch: np.ndarray, hidden_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    theta_batch: (B, 2*hidden_dim) float64/float32 numpy.
    returns: (film_gamma, film_beta) as torch tensors on CUDA of shape (B, hidden_dim)
    """
    tb = np.asarray(theta_batch, dtype=np.float32)
    if tb.ndim != 2 or tb.shape[1] != 2 * hidden_dim:
        raise ValueError(f"theta_batch shape {tb.shape} expected (B, {2*hidden_dim})")
    g = torch.from_numpy(tb[:, :hidden_dim]).cuda()
    b = torch.from_numpy(tb[:, hidden_dim:]).cuda()
    return g, b


def rollout_batch_episode_returns(
    policy: ACTPolicy,
    envs: list,
    eval_cfg: dict,
    pre_process,
    post_process,
    thetas: np.ndarray,
    *,
    latent_z,
    fixed_object_pose: np.ndarray,
    fixed_init_qpos: np.ndarray | None,
    init_qpos_from_dataset: bool,
    dataset_dir: str | None,
    num_episodes: int | None,
) -> np.ndarray:
    """
    同步推进多个 env（每个 env 对应一个 theta），按时间步将观测拼 batch 在单 GPU 上推理。
    返回每条轨迹的 episode_return（shape: (B,)）。
    """
    B = len(envs)
    if thetas.shape[0] != B:
        raise ValueError(f"thetas batch {thetas.shape[0]} != envs {B}")

    hidden_dim = int(policy.model.visual_film_gamma.numel())
    film_gamma, film_beta = _split_film_theta_batch(thetas, hidden_dim)

    # reset all envs with fixed object pose
    from imitate_episodes import apply_object_pose_for_reset, overwrite_sim_qpos_from_dataset, sample_dataset_start_qpos

    ts_list = []
    for i, env in enumerate(envs):
        apply_object_pose_for_reset(eval_cfg["task_name"], fixed_object_pose)
        ts = env.reset()
        if fixed_init_qpos is not None:
            overwrite_sim_qpos_from_dataset(env, eval_cfg["task_name"], fixed_init_qpos)
            new_obs = env._task.get_observation(env._physics)
            ts = ts._replace(observation=new_obs)
        elif init_qpos_from_dataset:
            if dataset_dir is None or num_episodes is None:
                raise ValueError("init_qpos_from_dataset 需要 dataset_dir/num_episodes")
            qpos0 = sample_dataset_start_qpos(dataset_dir, num_episodes)
            overwrite_sim_qpos_from_dataset(env, eval_cfg["task_name"], qpos0)
            new_obs = env._task.get_observation(env._physics)
            ts = ts._replace(observation=new_obs)
        ts_list.append(ts)

    max_timesteps = int(eval_cfg["episode_len"] * 2)
    state_dim = int(eval_cfg["state_dim"])
    camera_names = eval_cfg["camera_names"]
    policy_config = eval_cfg["policy_config"]
    query_frequency = int(policy_config["num_queries"])
    temporal_agg = bool(eval_cfg["temporal_agg"])
    action_dim = int(eval_cfg.get("action_dim", state_dim))
    if temporal_agg:
        query_frequency = 1
        num_queries = int(policy_config["num_queries"])
        all_time_actions = torch.zeros([B, max_timesteps, max_timesteps + num_queries, action_dim], device="cuda")
    else:
        all_time_actions = None

    rewards_sum = np.zeros(B, dtype=np.float64)

    # imitation_episodes eval 同样用 inference_mode：避免 temporal_agg 把多步 policy 输出拼进
    # all_time_actions 时保留整段 episode 的 autograd 图导致显存爆炸
    with torch.inference_mode():
        for t in range(max_timesteps):
            # build batch obs
            qpos_batch = []
            img_batch = []
            for ts in ts_list:
                obs = ts.observation
                qpos_np = np.asarray(obs["qpos"], dtype=np.float32)[:state_dim]
                qpos_batch.append(pre_process(qpos_np))

                # images: (num_cam, C, H, W) normalized to 0..1
                cams = []
                for cam_name in camera_names:
                    im = obs["images"][cam_name]  # H,W,C uint8
                    im = np.transpose(im, (2, 0, 1)).astype(np.float32) / 255.0
                    cams.append(im)
                img_batch.append(np.stack(cams, axis=0))

            qpos_t = torch.from_numpy(np.stack(qpos_batch, axis=0)).float().cuda()
            img_t = torch.from_numpy(np.stack(img_batch, axis=0)).float().cuda()

            # query policy
            if t % query_frequency == 0:
                all_actions = policy(
                    qpos_t,
                    img_t,
                    latent_z_sample=latent_z,
                    film_gamma=film_gamma,
                    film_beta=film_beta,
                )

            if temporal_agg:
                assert all_time_actions is not None
                all_time_actions[:, t, t : t + num_queries] = all_actions
                actions_for_curr_step = all_time_actions[:, :, t]  # (B, T, action_dim)
                # populated mask
                populated = torch.all(actions_for_curr_step != 0, dim=2)  # (B, T)
                raw_action_list = []
                for i in range(B):
                    acts_i = actions_for_curr_step[i][populated[i]]
                    if acts_i.numel() == 0:
                        raw_action_list.append(torch.zeros((1, action_dim), device="cuda"))
                        continue
                    k = 0.01
                    w = np.exp(-k * np.arange(len(acts_i)))
                    w = w / w.sum()
                    w = torch.from_numpy(w).float().cuda().unsqueeze(1)
                    raw_action_list.append((acts_i * w).sum(dim=0, keepdim=True))
                raw_action = torch.cat(raw_action_list, dim=0)  # (B, action_dim)
            else:
                raw_action = all_actions[:, t % query_frequency]  # (B, action_dim)

            raw_np = raw_action.detach().cpu().numpy()
            target_qpos_batch = post_process(raw_np)  # (B, action_dim)

            # step envs
            new_ts_list = []
            for i, env in enumerate(envs):
                ts = env.step(np.asarray(target_qpos_batch[i], dtype=np.float64))
                r = ts.reward
                if r is not None:
                    rewards_sum[i] += float(r)
                new_ts_list.append(ts)
            ts_list = new_ts_list

    return rewards_sum


def _build_eval_config(
    task_name: str,
    task_cfg: dict,
    policy_class: str,
    policy_config: dict,
    seed: int,
    temporal_agg: bool,
    latent_z_sample,
    fixed_object_pose: np.ndarray | None,
    fixed_init_qpos: np.ndarray | None,
    init_qpos_from_dataset: bool,
):
    state_dim = task_cfg.get("state_dim", DEFAULT_STATE_DIM)
    action_dim = task_cfg.get("action_dim", state_dim)
    return {
        "real_robot": False,
        "policy_class": policy_class,
        "policy_config": policy_config,
        "task_name": task_name,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "episode_len": task_cfg["episode_len"],
        "camera_names": task_cfg["camera_names"],
        "temporal_agg": temporal_agg,
        "seed": seed,
        "latent_z_sample": None,
        "dataset_dir": task_cfg.get("dataset_dir"),
        "num_episodes": task_cfg.get("num_episodes"),
        "init_qpos_from_dataset": init_qpos_from_dataset,
        "direct_replay": False,
        "fixed_object_pose": fixed_object_pose,
        "fixed_init_qpos": fixed_init_qpos,
    }


def run_ars(
    fitness_fn,
    theta0: np.ndarray,
    *,
    n_iters: int,
    n_pairs: int,
    sigma: float,
    alpha: float,
    seed: int,
    log_path: Path,
):
    rng = np.random.default_rng(seed)
    theta = theta0.astype(np.float64).copy()
    dim = theta.size
    history_best = []
    history_iter_max = []
    best_so_far = -np.inf
    best_theta = theta.copy()

    with open(log_path, "w", encoding="utf-8") as flog:
        flog.write(f"# ARS dim={dim} sigma={sigma} alpha={alpha} n_pairs={n_pairs}\n")

    for it in range(n_iters):
        grad = np.zeros(dim, dtype=np.float64)
        iter_best = -np.inf
        snap = theta.copy()
        for _ in range(n_pairs):
            eps = rng.standard_normal(dim)
            t_p = theta + sigma * eps
            t_m = theta - sigma * eps
            r_plus = fitness_fn(t_p)
            r_minus = fitness_fn(t_m)
            if r_plus > iter_best:
                iter_best = r_plus
                snap = t_p.copy()
            if r_minus > iter_best:
                iter_best = r_minus
                snap = t_m.copy()
            grad += (r_plus - r_minus) / (2.0 * sigma) * eps
        grad /= max(n_pairs, 1)
        theta = theta + alpha * grad
        r_end = fitness_fn(theta)
        if r_end > iter_best:
            iter_best = r_end
            snap = theta.copy()
        if iter_best > best_so_far:
            best_so_far = float(iter_best)
            best_theta = snap.copy()
        history_best.append(best_so_far)
        history_iter_max.append(float(iter_best))
        with open(log_path, "a", encoding="utf-8") as flog:
            flog.write(json.dumps({"iter": it, "best_so_far": best_so_far, "iter_best": iter_best}) + "\n")
        print(f"ARS iter {it}: iter_best={iter_best:.4f} best_so_far={best_so_far:.4f}")

    return best_theta, np.array(history_best), np.array(history_iter_max)


def run_ars_batched(
    fitness_batch_fn,
    theta0: np.ndarray,
    *,
    n_iters: int,
    n_pairs: int,
    sigma: float,
    alpha: float,
    seed: int,
    log_path: Path,
    batch_size: int,
):
    """
    ARS，但用 batch fitness 一次评估多个候选（单 GPU 前向批处理）。
    fitness_batch_fn: (N,dim)->(N,) episode_return
    """
    rng = np.random.default_rng(seed)
    theta = theta0.astype(np.float64).copy()
    dim = theta.size
    best_so_far = -np.inf
    best_theta = theta.copy()
    history_best = []
    history_iter_best = []

    with open(log_path, "w", encoding="utf-8") as flog:
        flog.write(
            f"# ARS(batched) dim={dim} sigma={sigma} alpha={alpha} n_pairs={n_pairs} batch_size={batch_size}\n"
        )

    for it in range(n_iters):
        eps_list = [rng.standard_normal(dim) for _ in range(n_pairs)]
        cand = []
        for eps in eps_list:
            cand.append(theta + sigma * eps)
            cand.append(theta - sigma * eps)
        # evaluate all perturbations
        rewards = []
        for i in range(0, len(cand), batch_size):
            tb = np.stack(cand[i : i + batch_size], axis=0)
            rewards.extend(list(map(float, fitness_batch_fn(tb))))
        rewards = np.asarray(rewards, dtype=np.float64)

        iter_best = float(np.max(rewards))
        best_idx = int(np.argmax(rewards))
        snap = cand[best_idx].copy()

        # grad estimate
        grad = np.zeros(dim, dtype=np.float64)
        for k, eps in enumerate(eps_list):
            r_plus = rewards[2 * k]
            r_minus = rewards[2 * k + 1]
            grad += (r_plus - r_minus) / (2.0 * sigma) * eps
        grad /= max(n_pairs, 1)

        theta = theta + alpha * grad
        r_end = float(fitness_batch_fn(theta.reshape(1, -1))[0])
        if r_end > iter_best:
            iter_best = r_end
            snap = theta.copy()

        if iter_best > best_so_far:
            best_so_far = iter_best
            best_theta = snap.copy()

        history_best.append(best_so_far)
        history_iter_best.append(iter_best)
        with open(log_path, "a", encoding="utf-8") as flog:
            flog.write(json.dumps({"iter": it, "best_so_far": best_so_far, "iter_best": iter_best}) + "\n")
        print(f"ARS(batched) iter {it}: iter_best={iter_best:.4f} best_so_far={best_so_far:.4f}")

    return best_theta, np.asarray(history_best), np.asarray(history_iter_best)


def run_cma(
    fitness_fn,
    theta0: np.ndarray,
    *,
    sigma0: float,
    maxiter: int,
    popsize: int | None,
    seed: int,
    log_path: Path,
):
    import cma

    x0 = theta0.astype(np.float64).copy()
    opts: dict = {
        "seed": seed,
        "maxiter": maxiter,
        "verb_disp": 1,
    }
    if popsize is not None:
        opts["popsize"] = popsize
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    history_best = []
    history_gen_max = []
    best_so_far = -np.inf
    best_x = x0.copy()

    with open(log_path, "w", encoding="utf-8") as flog:
        flog.write(
            f"# CMA-ES dim={x0.size} sigma0={sigma0} maxiter={maxiter} popsize={getattr(es, 'popsize', None)}\n"
        )

    while not es.stop():
        xs = es.ask()
        rewards = [fitness_fn(np.asarray(x, dtype=np.float64)) for x in xs]
        es.tell(xs, [-float(r) for r in rewards])
        gen_max = float(np.max(rewards))
        gen = int(es.countiter)
        ib = int(np.argmax(rewards))
        if rewards[ib] > best_so_far:
            best_so_far = float(rewards[ib])
            best_x = np.asarray(xs[ib], dtype=np.float64).copy()
        history_best.append(best_so_far)
        history_gen_max.append(gen_max)
        with open(log_path, "a", encoding="utf-8") as flog:
            flog.write(
                json.dumps(
                    {
                        "generation": gen,
                        "gen_max": gen_max,
                        "best_so_far": best_so_far,
                    }
                )
                + "\n"
            )
        print(f"CMA gen {gen}: gen_max={gen_max:.4f} best_so_far={best_so_far:.4f}")

    return best_x, np.array(history_best), np.array(history_gen_max)


def run_cma_batched(
    fitness_batch_fn,
    theta0: np.ndarray,
    *,
    sigma0: float,
    maxiter: int,
    popsize: int | None,
    seed: int,
    log_path: Path,
    batch_size: int,
):
    import cma

    x0 = theta0.astype(np.float64).copy()
    opts: dict = {"seed": seed, "maxiter": maxiter, "verb_disp": 1}
    if popsize is not None:
        opts["popsize"] = popsize
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    best_so_far = -np.inf
    best_x = x0.copy()
    history_best = []
    history_gen_max = []

    with open(log_path, "w", encoding="utf-8") as flog:
        flog.write(
            f"# CMA-ES(batched) dim={x0.size} sigma0={sigma0} maxiter={maxiter} popsize={getattr(es,'popsize',None)} batch_size={batch_size}\n"
        )

    while not es.stop():
        xs = es.ask()
        rewards = []
        for i in range(0, len(xs), batch_size):
            tb = np.stack([np.asarray(x, dtype=np.float64) for x in xs[i : i + batch_size]], axis=0)
            rewards.extend(list(map(float, fitness_batch_fn(tb))))
        rewards = np.asarray(rewards, dtype=np.float64)
        es.tell(xs, list((-rewards).astype(float)))

        gen_max = float(np.max(rewards))
        gen = int(es.countiter)
        ib = int(np.argmax(rewards))
        if rewards[ib] > best_so_far:
            best_so_far = float(rewards[ib])
            best_x = np.asarray(xs[ib], dtype=np.float64).copy()
        history_best.append(best_so_far)
        history_gen_max.append(gen_max)
        with open(log_path, "a", encoding="utf-8") as flog:
            flog.write(json.dumps({"generation": gen, "gen_max": gen_max, "best_so_far": best_so_far}) + "\n")
        print(f"CMA(batched) gen {gen}: gen_max={gen_max:.4f} best_so_far={best_so_far:.4f}")

    return best_x, np.asarray(history_best), np.asarray(history_gen_max)


def _save_curve_png(path: Path, y1: np.ndarray, y1_label: str, y2: np.ndarray | None, y2_label: str | None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(len(y1)), y1, label=y1_label)
    if y2 is not None:
        ax.plot(np.arange(len(y2)), y2, alpha=0.6, label=y2_label or "iter/gen max")
    ax.set_xlabel("iteration / generation")
    ax.set_ylabel("episode_return")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="FiLM gamma/beta 搜索（CMA-ES 或 ARS）")
    p.add_argument("--ckpt", type=str, required=True, help="policy .ckpt 路径")
    p.add_argument("--stats_path", type=str, default=None, help="dataset_stats.pkl，默认与 ckpt 同目录")
    p.add_argument("--task_name", type=str, required=True, help="SIM_TASK_CONFIGS 中的任务名")
    p.add_argument("--output_dir", type=str, default="tmp/film_param_search")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temporal_agg", action="store_true")
    p.add_argument("--latent_z_sample", type=str, default=None)
    p.add_argument(
        "--fixed_object_pose",
        type=str,
        required=True,
        help="固定物体位姿：逗号分隔或 JSON 列表（transfer 7 维 / insertion 14 维 / dex 7 维）",
    )
    p.add_argument(
        "--fixed_init_qpos",
        type=str,
        default=None,
        help="可选：固定机械臂初始 qpos（与 eval 中数据集格式一致）",
    )
    p.add_argument(
        "--init_qpos_from_dataset",
        action="store_true",
        help="从数据集随机一条轨迹的起点初始化 qpos（与固定物体可同时用）",
    )
    p.add_argument("--method", type=str, choices=("cma", "ars"), required=True)
    # policy architecture (must match training)
    p.add_argument("--policy_class", type=str, default="ACT")
    p.add_argument("--chunk_size", type=int, default=100)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--dim_feedforward", type=int, default=3200)
    p.add_argument("--latent_z_dim", type=int, default=32)
    p.add_argument("--kl_weight", type=float, default=10.0)
    # ARS
    p.add_argument("--ars_iters", type=int, default=50)
    p.add_argument("--ars_pairs", type=int, default=4, help="每轮对称扰动对数（每轮 2*pairs 次仿真）")
    p.add_argument("--ars_sigma", type=float, default=0.05)
    p.add_argument("--ars_alpha", type=float, default=0.1)
    # CMA
    p.add_argument("--cma_sigma0", type=float, default=0.3)
    p.add_argument("--cma_maxiter", type=int, default=50)
    p.add_argument("--cma_popsize", type=int, default=None)
    p.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="并行评估候选数（一次 batch 同步推进多少条轨迹；单 GPU 批量推理）",
    )

    args = p.parse_args()
    if args.policy_class != "ACT":
        print("FiLM 仅存在于 ACT (DETRVAE)，请使用 --policy_class ACT", file=sys.stderr)
        sys.exit(1)

    task_name = args.task_name
    if task_name not in SIM_TASK_CONFIGS:
        print(f"未知 task_name: {task_name}. 可选: {list(SIM_TASK_CONFIGS.keys())}", file=sys.stderr)
        sys.exit(1)
    task_cfg = SIM_TASK_CONFIGS[task_name]

    set_seed(args.seed)
    fixed_object_pose = _parse_float_list(args.fixed_object_pose)
    fixed_init_qpos = _parse_float_list(args.fixed_init_qpos) if args.fixed_init_qpos else None

    policy_config = {
        "lr": 1e-5,
        "num_queries": args.chunk_size,
        "kl_weight": args.kl_weight,
        "hidden_dim": args.hidden_dim,
        "dim_feedforward": args.dim_feedforward,
        "latent_z_dim": args.latent_z_dim,
        "lr_backbone": 1e-5,
        "backbone": "resnet18",
        "enc_layers": 4,
        "dec_layers": 7,
        "nheads": 8,
        "camera_names": task_cfg["camera_names"],
        "state_dim": task_cfg.get("state_dim", DEFAULT_STATE_DIM),
        "action_dim": task_cfg.get("action_dim", task_cfg.get("state_dim", DEFAULT_STATE_DIM)),
    }

    policy, stats, ckpt_loaded = _load_policy_and_stats(
        Path(args.ckpt),
        Path(args.stats_path) if args.stats_path else None,
        args.policy_class,
        policy_config,
    )

    hidden_dim = int(policy.model.visual_film_gamma.numel())
    film_dim = 2 * hidden_dim
    print(f"Loaded {ckpt_loaded}, FiLM 参数维度 = {film_dim} (hidden_dim={hidden_dim})")

    latent_z = _parse_latent_z(args.latent_z_sample, args.latent_z_dim)

    eval_cfg = _build_eval_config(
        task_name,
        task_cfg,
        args.policy_class,
        policy_config,
        args.seed,
        args.temporal_agg,
        latent_z,
        fixed_object_pose,
        fixed_init_qpos,
        args.init_qpos_from_dataset,
    )

    pre = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
    post = lambda a: a * stats["action_std"] + stats["action_mean"]

    max_timesteps = int(task_cfg["episode_len"] * 2)
    envs = []
    env_max_reward = None
    for _ in range(max(1, int(args.parallel))):
        env_i = make_sim_env(task_name, time_limit=max_timesteps * DT)
        envs.append(env_i)
        if env_max_reward is None:
            env_max_reward = env_i.task.max_reward
    print(
        f"env_max_reward={env_max_reward}, episode_len*2 steps cap ≈ {max_timesteps}, parallel={len(envs)}"
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def fitness_batch(theta_batch: np.ndarray) -> np.ndarray:
        return rollout_batch_episode_returns(
            policy,
            envs[: theta_batch.shape[0]],
            eval_cfg,
            pre,
            post,
            theta_batch,
            latent_z=latent_z,
            fixed_object_pose=fixed_object_pose,
            fixed_init_qpos=fixed_init_qpos,
            init_qpos_from_dataset=args.init_qpos_from_dataset,
            dataset_dir=task_cfg.get("dataset_dir"),
            num_episodes=task_cfg.get("num_episodes"),
        )

    def fitness(theta: np.ndarray) -> float:
        return float(fitness_batch(np.asarray(theta, dtype=np.float64).reshape(1, -1))[0])

    theta0 = _film_theta_from_policy(policy)
    meta = {
        "ckpt": ckpt_loaded,
        "task_name": task_name,
        "method": args.method,
        "film_dim": film_dim,
        "fixed_object_pose": fixed_object_pose.tolist(),
        "env_max_reward": env_max_reward,
    }
    with open(out_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if args.method == "ars":
        log_path = out_dir / "ars_history.jsonl"
        if int(args.parallel) > 1:
            best_theta, h_best, h_iter = run_ars_batched(
                fitness_batch,
                theta0,
                n_iters=args.ars_iters,
                n_pairs=args.ars_pairs,
                sigma=args.ars_sigma,
                alpha=args.ars_alpha,
                seed=args.seed,
                log_path=log_path,
                batch_size=int(args.parallel),
            )
        else:
            best_theta, h_best, h_iter = run_ars(
                fitness,
                theta0,
                n_iters=args.ars_iters,
                n_pairs=args.ars_pairs,
                sigma=args.ars_sigma,
                alpha=args.ars_alpha,
                seed=args.seed,
                log_path=log_path,
            )
        np.savez(out_dir / "ars_curves.npz", best_so_far=h_best, iter_max=h_iter)
        _save_curve_png(out_dir / "reward_curve.png", h_best, "best_so_far", h_iter, "iter_max")
    else:
        log_path = out_dir / "cma_history.jsonl"
        if int(args.parallel) > 1:
            best_theta, h_best, h_gen = run_cma_batched(
                fitness_batch,
                theta0,
                sigma0=args.cma_sigma0,
                maxiter=args.cma_maxiter,
                popsize=args.cma_popsize,
                seed=args.seed,
                log_path=log_path,
                batch_size=int(args.parallel),
            )
        else:
            best_theta, h_best, h_gen = run_cma(
                fitness,
                theta0,
                sigma0=args.cma_sigma0,
                maxiter=args.cma_maxiter,
                popsize=args.cma_popsize,
                seed=args.seed,
                log_path=log_path,
            )
        np.savez(out_dir / "cma_curves.npz", best_so_far=h_best, gen_max=h_gen)
        _save_curve_png(out_dir / "reward_curve.png", h_best, "best_so_far", h_gen, "gen_max")

    _apply_film_theta(policy, best_theta, hidden_dim)
    film_ckpt = {
        "visual_film_gamma": policy.model.visual_film_gamma.cpu(),
        "visual_film_beta": policy.model.visual_film_beta.cpu(),
        "best_theta": torch.from_numpy(best_theta.astype(np.float32)),
    }
    torch.save(film_ckpt, out_dir / "best_film_only.pt")
    best_logged = float(np.max(h_best)) if len(h_best) else float("nan")
    print(f"Done. Best episode_return in log ≈ {best_logged}")
    print(f"Saved: {out_dir / 'best_film_only.pt'}, {out_dir / 'reward_curve.png'}")


if __name__ == "__main__":
    main()
