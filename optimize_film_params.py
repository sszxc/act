#!/usr/bin/env python3
"""
Search PCA-bottleneck FiLM parameters in ACT with CMA-ES, ARS, or an LLM optimizer, under a fixed
object pose, to maximize episode_return.

The search space is theta = [gamma, beta], each k-dim, where k = --film_bottleneck_dim. These are
NOT raw per-channel FiLM values: they modulate a k-dim PCA subspace of a hidden_dim-wide tensor
(see detr_vae.py's load_film_pca()/forward()), fit offline from recorded activations by
fit_film_pca.py. --film_pca_path points at that fit's .npz output; k is sliced from its stored
basis (k <= max_k used when fitting).

--film_target selects WHERE in the network that tensor is (must match --target used when fitting
--film_pca_path):
  - "visual" (default): before the Transformer encoder (the original mode).
  - "memory": the encoder-decoder boundary (encoder's output, right before the decoder reads it).
  - "hs": right after the decoder, before action_head/is_pad_head.

Requires:
  - numpy, torch, matplotlib
  - CMA-ES: pip install cma (only for --method cma)

Examples:
  # 1) Fit the PCA basis once per ckpt/task/target (no sim rollout needed):
  python fit_film_pca.py --ckpt /path/to/policy_best.ckpt --task_name sim_transfer_cube_human \\
    --target visual --max_k 64 --output tmp/film_pca/sim_transfer_cube_human_visual.npz

  # 2) Search theta (gamma, beta) in the resulting k-dim bottleneck (--film_target must match):
  python optimize_film_params.py --ckpt /path/to/policy_best.ckpt --task_name sim_transfer_cube_human \\
    --film_pca_path tmp/film_pca/sim_transfer_cube_human_visual.npz --film_bottleneck_dim 8 --film_target visual \\
    --fixed_object_pose "0.1,0.5,0.05,1,0,0,0" --method ars --ars_iters 3 --ars_pairs 2 --output_dir tmp/film_search

  python optimize_film_params.py --ckpt ... --film_pca_path ... --film_bottleneck_dim 8 --method cma --cma_maxiter 5 --cma_popsize 8 ...

  # candidate 4 / 5 (fit + search must use the same --target / --film_target):
  python fit_film_pca.py --ckpt ... --task_name ... --target memory --output tmp/film_pca/..._memory.npz
  python optimize_film_params.py --ckpt ... --film_pca_path tmp/film_pca/..._memory.npz --film_target memory \\
    --film_bottleneck_dim 8 --fixed_object_pose "..." --method ars --output_dir tmp/film_search_memory

This file is the CLI entry point (argparse + orchestration) only; the actual algorithms live under
src/: film_utils.py (policy/FiLM theta), rollout.py (sim rollout), optimizers.py (ARS/CMA/sweep),
llm_optimizer.py (LLM optimizer), logging_utils.py / cli_utils.py (shared small helpers).
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

# Match imitate_episodes: EGL when headless
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

from constants import DT, SIM_TASK_CONFIGS, DEFAULT_STATE_DIM
from imitate_episodes import set_seed
from sim_env import make_sim_env
from visualize_episodes import _video_path_for_cam

from src.cli_utils import _parse_float_list, _parse_latent_z
from src.film_utils import _FILM_TARGETS, _film_pca_attr, _load_policy_and_stats, _film_theta_from_policy, _apply_film_theta
from src.rollout import rollout_batch_episode_returns, _build_eval_config
from src.logging_utils import _format_duration, _save_curve_png
from src.optimizers import run_ars, run_ars_batched, run_cma, run_cma_batched, run_sweep
from src.llm_optimizer import _prompt_template_is_vlm, _prompt_template_uses_reward_predictor, run_llm


def main():
    p = argparse.ArgumentParser(description="FiLM gamma/beta search (CMA-ES or ARS)")
    p.add_argument("--ckpt", type=str, required=True, help="path to policy .ckpt")
    p.add_argument("--stats_path", type=str, default=None, help="dataset_stats.pkl; defaults next to ckpt")
    p.add_argument("--task_name", type=str, required=True, help="task name in SIM_TASK_CONFIGS")
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Parent folder for this run's outputs; a new "
        "icl_<timestamp>_<model_or_method> subfolder is always created under it "
        "(defaults to the ckpt's own directory, alongside eval_* folders)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--save_videos",
        action="store_true",
        help="Render and save an mp4 per evaluated candidate per round to <output_dir>/videos/ "
        "(off by default: adds per-step frame capture + encoding overhead)",
    )
    p.add_argument(
        "--video_layout",
        type=str,
        choices=("combined", "separate"),
        default="combined",
        help="When --save_videos: 'combined' writes one side-by-side mp4 per env (default); "
        "'separate' writes one mp4 per camera (e.g. round_0001_env0_default_cam.mp4)",
    )
    p.add_argument("--temporal_agg", action="store_true")
    p.add_argument("--latent_z_sample", type=str, default=None)
    p.add_argument(
        "--fixed_object_pose",
        type=str,
        required=True,
        help="Fixed object pose: comma or JSON list (transfer 7 / insertion 14 / dex 7)",
    )
    p.add_argument(
        "--fixed_object_shape",
        type=str,
        default=None,
        choices=("box", "cylinder", "sphere"),
        help="HMF proto5 only: pin the target object's shape (size = mid-point of the task's "
        "train_shapes size_ranges; z auto-derived from table_z + half-extent, overriding the z "
        "in --fixed_object_pose). Omit to keep the task's default shape.",
    )
    p.add_argument(
        "--fixed_object_size",
        type=str,
        default=None,
        help="Requires --fixed_object_shape. Comma or JSON list overriding the default "
        "mid-range size (box: half-extents x,y,z / cylinder: radius,half-height / sphere: "
        "radius). Not clamped to the task's train_shapes size_ranges.",
    )
    p.add_argument(
        "--fixed_init_qpos",
        type=str,
        default=None,
        help="Optional fixed arm init qpos (same format as eval dataset)",
    )
    p.add_argument(
        "--init_qpos_from_dataset",
        action="store_true",
        help="Init qpos from a random trajectory start in the dataset (combines with fixed object)",
    )
    p.add_argument("--method", type=str, choices=("cma", "ars", "llm", "sweep"), required=True)
    # policy architecture (must match training)
    p.add_argument("--policy_class", type=str, default="ACT")
    p.add_argument("--chunk_size", type=int, default=100)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--dim_feedforward", type=int, default=3200)
    p.add_argument("--latent_z_dim", type=int, default=32)
    p.add_argument("--kl_weight", type=float, default=10.0)
    # ARS
    p.add_argument("--ars_iters", type=int, default=50)
    p.add_argument("--ars_pairs", type=int, default=4, help="Symmetric perturbation pairs per iter (2*pairs sim calls per iter)")
    p.add_argument("--ars_sigma", type=float, default=0.05)
    p.add_argument("--ars_alpha", type=float, default=0.1)
    # CMA
    p.add_argument("--cma_sigma0", type=float, default=0.3)
    p.add_argument("--cma_maxiter", type=int, default=50)
    p.add_argument("--cma_popsize", type=int, default=None)
    # sweep (manual one-at-a-time grid search)
    p.add_argument(
        "--sweep_values",
        type=str,
        default="-10,-5,-3,-1,-0.5,0,0.5,1,3,5,10",
        help="--method sweep: comma/JSON list of raw theta values. Swept one dim at a time "
        "(gamma_0..gamma_{k-1}, beta_0..beta_{k-1}) with every other dim held at theta_base "
        "(the policy's currently loaded FiLM identity: gamma=1, beta=0) — n_dims * n_values "
        "rollouts total. Always records the end-effector (mocap) xyz trajectory per point (see "
        "film_sweep_trajectories.npz); pass --save_videos to also render a video per point "
        "(off by default, same as every other --method — video encoding is ~15-20%% of a "
        "rollout's wall time, measured ~4-5s of ~20-30s).",
    )
    p.add_argument(
        "--object_sweep_x_values",
        type=str,
        default="0",
        help="--method sweep: comma/JSON list of x offsets (meters) from --fixed_object_pose's x, "
        "crossed with the FiLM dim/value grid (for every offset, the whole one-at-a-time FiLM "
        "sweep is re-run at that object position) -- rollouts multiply by len(x) * len(y). "
        "Default '0' keeps the original single fixed-position behavior.",
    )
    p.add_argument(
        "--object_sweep_y_values",
        type=str,
        default="0",
        help="--method sweep: same as --object_sweep_x_values but for the object's y offset.",
    )
    p.add_argument(
        "--show_rollout_progress",
        action="store_true",
        help="Show tqdm-style progress bar for each simulation rollout (requires tqdm; otherwise no-op)",
    )
    p.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Parallel candidates to evaluate (batch rollout count; single-GPU batched inference)",
    )
    p.add_argument(
        "--film_pca_path",
        type=str,
        required=True,
        help="Path to a fit_film_pca.py .npz output (W, mu, explained_variance_ratio, meta). "
        "Must have been fit with --target matching --film_target below.",
    )
    p.add_argument(
        "--film_bottleneck_dim",
        type=int,
        required=True,
        help="k: PCA-bottleneck FiLM dim to search (theta = [gamma, beta], each k-dim). "
        "Must be <= max_k stored in --film_pca_path; W is sliced to W[:, :k]",
    )
    p.add_argument(
        "--film_target",
        type=str,
        choices=_FILM_TARGETS,
        default="visual",
        help="Which encoder-FiLM-decoder insertion point to search: 'visual' (default, before "
        "the Transformer encoder), 'memory' (candidate 4: encoder-decoder boundary), or 'hs' "
        "(candidate 5: pre-action_head, after the decoder). See detr_vae.py's "
        "load_film_pca(..., target=...).",
    )
    # LLM optimizer
    p.add_argument("--llm_model", type=str, default="llama4-scout-17b", help="LLM model name")
    p.add_argument("--llm_maxiter", type=int, default=50, help="LLM optimization iterations")
    p.add_argument("--llm_temperature", type=float, default=0.2, help="LLM sampling temperature")
    p.add_argument("--llm_max_retries", type=int, default=3, help="LLM retries per iteration for valid unseen params")
    p.add_argument(
        "--llm_max_tokens",
        type=int,
        default=8192,
        help="Explicit max_tokens per LLM call. 'Thinking' models can burn the whole "
        "generation budget on their <think> block and return empty content if this is too "
        "small (or left to a stingy backend default); pass 0 to omit max_tokens entirely.",
    )
    p.add_argument(
        "--llm_retry_temperature_bump",
        type=float,
        default=0.2,
        help="Added to --llm_temperature on each retry within an iteration (capped at 2.0), so "
        "a retry has a real chance of escaping a repeated duplicate/empty response instead of "
        "resampling the same near-deterministic low-temperature output.",
    )
    p.add_argument("--llm_history_window", type=int, default=40, help="How many past samples to include in each prompt")
    p.add_argument("--llm_step_size_hint", type=float, default=0.5, help="Exploration step-size hint in prompt")
    p.add_argument(
        "--llm_optimum_hint",
        type=float,
        default=None,
        help="Approx best episode return R for the prompt hint (default: env task max_reward)",
    )
    p.add_argument(
        "--llm_prompt_template",
        type=str,
        default=None,
        help="Prompt template python file (defaults to prompts/num_optim_Pratyush.py)",
    )
    p.add_argument(
        "--llm_vlm_num_frames",
        type=int,
        default=5,
        help="Frames sampled from the previous round's rollout for VLM visual feedback, "
        "stacked into one composite image (only used when the prompt template sets IS_VLM = True, "
        "e.g. prompts/num_optim_Pratyush_vlm.py)",
    )
    p.add_argument(
        "--llm_rp_instruction",
        type=str,
        default=None,
        help="Natural-language task instruction passed to the local reward_predictor for "
        "subgoal generation (required when the prompt template sets IS_REWARD_PREDICTOR = True, "
        "e.g. prompts/num_optim_Pratyush_reward_predictor.py)",
    )
    p.add_argument(
        "--llm_rp_camera_labels",
        type=str,
        default=None,
        help="Comma-separated camera labels, one per task camera name (same order as the task's "
        "camera_names), describing each recorded rollout video for reward_predictor's "
        "camera_labels. Reward-predictor mode renders/scores one video per camera (never merged "
        "side-by-side). Default: the task's own camera_names.",
    )
    p.add_argument(
        "--llm_rp_repo_path",
        type=str,
        default="/home/lab/Documents/reward_predictor",
        help="Path to the reward_predictor checkout; added to sys.path to import its ipc.client",
    )
    p.add_argument(
        "--llm_rp_socket_path",
        type=str,
        default="/tmp/reward_predictor_ipc.sock",
        help="Unix socket path of a running `python -m ipc.server` in the reward_predictor conda env",
    )
    p.add_argument(
        "--llm_rp_timeout",
        type=float,
        default=3600.0,
        help="Timeout (seconds) per reward_predictor IPC call; the first call also pays for model load",
    )

    args = p.parse_args()
    if args.policy_class != "ACT":
        print("FiLM exists only on ACT (DETRVAE); use --policy_class ACT", file=sys.stderr)
        sys.exit(1)

    # --output_dir is the *parent* folder for this run; the actual run always lands one level
    # down in icl_<timestamp>_<model_or_method>/, so a script sweeping multiple runs can pass the
    # same --output_dir repeatedly without them clobbering each other.
    # Resolved to absolute here (not left relative to this process's cwd): reward-predictor mode
    # sends rp_video_dir paths over IPC to a *separate* server process with its own cwd, which
    # resolves relative paths against itself and silently looks in the wrong place.
    parent_dir = Path(args.output_dir).resolve() if args.output_dir is not None else Path(args.ckpt).resolve().parent
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = args.llm_model if args.method == "llm" else args.method
    args.output_dir = str(parent_dir / f"icl_{timestamp}_{model_name}")

    task_name = args.task_name
    if task_name not in SIM_TASK_CONFIGS:
        print(f"Unknown task_name: {task_name}. Options: {list(SIM_TASK_CONFIGS.keys())}", file=sys.stderr)
        sys.exit(1)
    task_cfg = SIM_TASK_CONFIGS[task_name]

    set_seed(args.seed)
    fixed_object_pose = _parse_float_list(args.fixed_object_pose)
    fixed_object_shape = args.fixed_object_shape
    fixed_object_size = _parse_float_list(args.fixed_object_size) if args.fixed_object_size else None
    if fixed_object_size is not None and fixed_object_shape is None:
        print("--fixed_object_size requires --fixed_object_shape", file=sys.stderr)
        sys.exit(1)
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

    film_pca_path = Path(args.film_pca_path).resolve()
    pca_npz = np.load(film_pca_path, allow_pickle=False)
    pca_W_full = pca_npz["W"]  # (hidden_dim, max_k)
    pca_mu = pca_npz["mu"]  # (hidden_dim,)
    max_k = int(pca_W_full.shape[1])
    if pca_W_full.shape[0] != hidden_dim:
        print(
            f"--film_pca_path hidden_dim={pca_W_full.shape[0]} != model hidden_dim={hidden_dim}; "
            "was it fit against a different --hidden_dim / architecture?",
            file=sys.stderr,
        )
        sys.exit(1)
    k = int(args.film_bottleneck_dim)
    if k <= 0 or k > max_k:
        print(f"--film_bottleneck_dim={k} must be in [1, {max_k}] (max_k stored in {film_pca_path})", file=sys.stderr)
        sys.exit(1)
    policy.model.load_film_pca(
        torch.from_numpy(np.ascontiguousarray(pca_W_full[:, :k])).float(),
        torch.from_numpy(np.ascontiguousarray(pca_mu)).float(),
        target=args.film_target,
    )
    film_dim = 2 * k
    print(
        f"Loaded {ckpt_loaded}, PCA-bottleneck FiLM target={args.film_target}: k={k} "
        f"(max_k={max_k} in {film_pca_path}), film_dim={film_dim}"
    )
    sweep_dim_names = [f"gamma_{i}" for i in range(k)] + [f"beta_{i}" for i in range(k)]

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
    with open(out_dir / "command.txt", "w", encoding="utf-8") as f:
        f.write("python " + " ".join(shlex.quote(a) for a in sys.argv) + "\n")
    video_dir = out_dir / "videos" if args.save_videos else None
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)
    _round_counter = {"n": 0}

    def eval_theta_batch(theta_batch: np.ndarray, capture_frames: bool = False, video_dir_override: Path | None = None):
        round_idx = _round_counter["n"]
        _round_counter["n"] += 1
        return rollout_batch_episode_returns(
            policy,
            envs[: theta_batch.shape[0]],
            eval_cfg,
            pre,
            post,
            theta_batch,
            latent_z=latent_z,
            fixed_object_pose=fixed_object_pose,
            fixed_object_shape=fixed_object_shape,
            fixed_object_size=fixed_object_size,
            fixed_init_qpos=fixed_init_qpos,
            init_qpos_from_dataset=args.init_qpos_from_dataset,
            dataset_dir=task_cfg.get("dataset_dir"),
            num_episodes=task_cfg.get("num_episodes"),
            show_rollout_progress=bool(args.show_rollout_progress),
            rollout_desc=f"sim rollout x{theta_batch.shape[0]}",
            video_dir=video_dir_override if video_dir_override is not None else video_dir,
            round_label=f"round_{round_idx:04d}",
            capture_frames=capture_frames,
            video_layout=args.video_layout,
            film_target=args.film_target,
        )

    theta_base = _film_theta_from_policy(policy, target=args.film_target).astype(np.float64, copy=False)

    def fitness_batch(theta_batch: np.ndarray) -> np.ndarray:
        return eval_theta_batch(theta_batch)

    def fitness(theta: np.ndarray) -> float:
        return float(fitness_batch(np.asarray(theta, dtype=np.float64).reshape(1, -1))[0])

    def fitness_and_frames(theta: np.ndarray) -> tuple[float, list[dict]]:
        theta_batch = np.asarray(theta, dtype=np.float64).reshape(1, -1)
        rewards, image_lists = eval_theta_batch(theta_batch, capture_frames=True)
        return float(rewards[0]), image_lists[0]

    rp_video_dir = out_dir / "rp_videos"

    def fitness_and_video(theta: np.ndarray) -> tuple[float, list[str]]:
        # video_layout is forced to "separate" for reward-predictor mode (see main()), so this
        # rollout writes one mp4 per camera_names entry; return those paths in that same order.
        theta_batch = np.asarray(theta, dtype=np.float64).reshape(1, -1)
        round_idx = _round_counter["n"]
        rewards = eval_theta_batch(theta_batch, video_dir_override=rp_video_dir)
        base_path = str(rp_video_dir / f"round_{round_idx:04d}_env0.mp4")
        video_paths = [_video_path_for_cam(base_path, cam) for cam in task_cfg["camera_names"]]
        return float(rewards[0]), video_paths

    opt_x0 = theta_base

    meta = {
        "ckpt": ckpt_loaded,
        "task_name": task_name,
        "method": args.method,
        "film_target": args.film_target,
        "film_pca_path": str(film_pca_path),
        "film_bottleneck_dim": k,
        "film_pca_max_k": max_k,
        "film_dim": film_dim,
        "theta_base": theta_base.tolist(),
        "fixed_object_pose": fixed_object_pose.tolist(),
        "fixed_object_shape": fixed_object_shape,
        "fixed_object_size": fixed_object_size.tolist() if fixed_object_size is not None else None,
        "env_max_reward": env_max_reward,
    }
    if args.save_videos:
        meta["videos_dir"] = "videos"
        meta["video_layout"] = args.video_layout
    if args.method == "sweep":
        sweep_values = _parse_float_list(args.sweep_values)
        meta["sweep_values"] = sweep_values.tolist()
        meta["sweep_dim_names"] = sweep_dim_names
        meta["sweep_trajectories_file"] = "film_sweep_trajectories.npz"
        # Object x/y grid crossed with the FiLM dim/value sweep (see run_sweep call below).
        # Offsets are relative to fixed_object_pose; stored here as absolute positions since
        # that's what the trajectories npz records per-point and what the viz plots directly.
        object_x_offsets = _parse_float_list(args.object_sweep_x_values)
        object_y_offsets = _parse_float_list(args.object_sweep_y_values)
        object_x_values = (fixed_object_pose[0] + object_x_offsets).tolist()
        object_y_values = (fixed_object_pose[1] + object_y_offsets).tolist()
        meta["object_x_values"] = object_x_values
        meta["object_y_values"] = object_y_values
    prompt_template_path: Path | None = None
    use_vlm = False
    use_reward_predictor = False
    rp_camera_labels: list[str] | None = None
    if args.method == "llm":
        prompt_template_path = (
            Path(args.llm_prompt_template).resolve()
            if args.llm_prompt_template
            else (Path(__file__).resolve().parent / "prompts" / "num_optim_Pratyush.py")
        )
        use_vlm = _prompt_template_is_vlm(prompt_template_path)
        use_reward_predictor = _prompt_template_uses_reward_predictor(prompt_template_path)
        if use_vlm and use_reward_predictor:
            print(
                f"{prompt_template_path} sets both IS_VLM and IS_REWARD_PREDICTOR; "
                "these video-feedback modes are mutually exclusive",
                file=sys.stderr,
            )
            sys.exit(1)
        if use_reward_predictor:
            if not args.llm_rp_instruction:
                print(
                    "--llm_rp_instruction is required when the prompt template sets "
                    "IS_REWARD_PREDICTOR = True",
                    file=sys.stderr,
                )
                sys.exit(1)
            if args.video_layout != "separate":
                print(
                    f"[LLM] reward-predictor mode scores one video per camera (never merged "
                    f"side-by-side); forcing --video_layout separate (was {args.video_layout!r})"
                )
                args.video_layout = "separate"
                if args.save_videos:
                    meta["video_layout"] = args.video_layout
            if args.llm_rp_camera_labels:
                rp_camera_labels = [s.strip() for s in args.llm_rp_camera_labels.split(",")]
                if len(rp_camera_labels) != len(task_cfg["camera_names"]):
                    print(
                        f"--llm_rp_camera_labels must have exactly {len(task_cfg['camera_names'])} "
                        f"comma-separated labels, one per camera {task_cfg['camera_names']}; "
                        f"got {len(rp_camera_labels)}",
                        file=sys.stderr,
                    )
                    sys.exit(1)
            else:
                rp_camera_labels = list(task_cfg["camera_names"])
        meta["llm_model"] = args.llm_model
        meta["llm_maxiter"] = int(args.llm_maxiter)
        meta["llm_temperature"] = float(args.llm_temperature)
        meta["llm_max_retries"] = int(args.llm_max_retries)
        meta["llm_max_tokens"] = int(args.llm_max_tokens) or None
        meta["llm_retry_temperature_bump"] = float(args.llm_retry_temperature_bump)
        meta["llm_history_window"] = int(args.llm_history_window)
        meta["llm_step_size_hint"] = float(args.llm_step_size_hint)
        # Under reward-predictor mode R(params) is progress_reward in [0, 1] (env_max_reward is on
        # the unrelated episode_return scale and would be a misleading default hint there).
        default_llm_optimum_hint = 1.0 if use_reward_predictor else float(env_max_reward)
        meta["llm_optimum_hint"] = (
            float(args.llm_optimum_hint) if args.llm_optimum_hint is not None else default_llm_optimum_hint
        )
        meta["llm_param_clip"] = [-6.0, 6.0]
        meta["llm_param_decimals"] = 1
        meta["llm_prompt_logs_dir"] = "llm_prompt_logs"
        meta["llm_prompt_template"] = str(prompt_template_path)
        meta["llm_vlm"] = use_vlm
        if use_vlm:
            meta["llm_vlm_num_frames"] = int(args.llm_vlm_num_frames)
            print(
                f"[LLM] VLM prompt detected ({prompt_template_path.name}); attaching a "
                f"{args.llm_vlm_num_frames}-frame visual feedback image (previous round only) "
                "from iteration 1 onward. Make sure --llm_model points at a vision-capable model."
            )
        meta["llm_reward_predictor"] = use_reward_predictor
        if use_reward_predictor:
            meta["llm_rp_instruction"] = args.llm_rp_instruction
            meta["llm_rp_camera_labels"] = rp_camera_labels
            meta["llm_rp_socket_path"] = args.llm_rp_socket_path
            sys.path.insert(0, str(Path(args.llm_rp_repo_path).resolve()))
            print(
                f"[LLM] reward-predictor prompt detected ({prompt_template_path.name}); scoring each "
                f"round's rollout video via the local reward_predictor IPC bridge "
                f"({args.llm_rp_socket_path}) with auto-generated subgoals for "
                f"instruction={args.llm_rp_instruction!r} from iteration 1 onward. Make sure "
                "`python -m ipc.server` is running in the reward_predictor conda env (with "
                "OPENAI_API_KEY sourced for subgoal generation)."
            )
    with open(out_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    interrupted_exc: BaseException | None = None
    elapsed_sec: float = 0.0
    if args.method == "ars":
        log_path = out_dir / "ars_history.jsonl"
        if int(args.parallel) > 1:
            best_x, h_best, h_iter, interrupted_exc, elapsed_sec = run_ars_batched(
                fitness_batch,
                opt_x0,
                n_iters=args.ars_iters,
                n_pairs=args.ars_pairs,
                sigma=args.ars_sigma,
                alpha=args.ars_alpha,
                seed=args.seed,
                log_path=log_path,
                batch_size=int(args.parallel),
            )
        else:
            best_x, h_best, h_iter, interrupted_exc, elapsed_sec = run_ars(
                fitness,
                opt_x0,
                n_iters=args.ars_iters,
                n_pairs=args.ars_pairs,
                sigma=args.ars_sigma,
                alpha=args.ars_alpha,
                seed=args.seed,
                log_path=log_path,
            )
        np.savez(out_dir / "ars_curves.npz", best_so_far=h_best, iter_max=h_iter)
        _save_curve_png(out_dir / "reward_curve.png", h_best, "best_so_far", h_iter, "iter_max", elapsed_sec=elapsed_sec)
    elif args.method == "cma":
        log_path = out_dir / "cma_history.jsonl"
        if int(args.parallel) > 1:
            best_x, h_best, h_gen, interrupted_exc, elapsed_sec = run_cma_batched(
                fitness_batch,
                opt_x0,
                sigma0=args.cma_sigma0,
                maxiter=args.cma_maxiter,
                popsize=args.cma_popsize,
                seed=args.seed,
                log_path=log_path,
                batch_size=int(args.parallel),
            )
        else:
            best_x, h_best, h_gen, interrupted_exc, elapsed_sec = run_cma(
                fitness,
                opt_x0,
                sigma0=args.cma_sigma0,
                maxiter=args.cma_maxiter,
                popsize=args.cma_popsize,
                seed=args.seed,
                log_path=log_path,
            )
        np.savez(out_dir / "cma_curves.npz", best_so_far=h_best, gen_max=h_gen)
        _save_curve_png(out_dir / "reward_curve.png", h_best, "best_so_far", h_gen, "gen_max", elapsed_sec=elapsed_sec)
    elif args.method == "sweep":
        if int(args.parallel) > 1:
            print("Warning: --method sweep evaluates one grid point per rollout; extra parallel envs stay idle.")
        sweep_records: list[dict] = []

        # Object (x, y) grid crossed with the FiLM dim/value grid: for every object position,
        # the whole one-at-a-time FiLM sweep is re-run there. n_obj == 1 (both --object_sweep_*
        # default to "0") reproduces the original single-fixed-position behavior exactly,
        # including unchanged log/video filenames.
        obj_grid = [(float(ox), float(oy)) for ox in object_x_offsets for oy in object_y_offsets]
        n_obj = len(obj_grid)
        n_total = n_obj * len(sweep_dim_names) * len(sweep_values)
        if n_obj > 1:
            print(
                f"[sweep] object grid: {len(object_x_offsets)} x-offsets x {len(object_y_offsets)} "
                f"y-offsets = {n_obj} positions, each with {len(sweep_dim_names)} FiLM dims x "
                f"{len(sweep_values)} values -> {n_total} rollouts total"
            )

        sweep_t_start = time.perf_counter()
        best_so_far_overall = -np.inf
        best_x = opt_x0.copy()
        all_point_rewards: list[float] = []
        for obj_idx, (ox_off, oy_off) in enumerate(obj_grid):
            fixed_object_pose_i = fixed_object_pose.copy()
            fixed_object_pose_i[0] += ox_off
            fixed_object_pose_i[1] += oy_off
            obj_tag = f"objx{ox_off:+.3f}_objy{oy_off:+.3f}" if n_obj > 1 else None
            if n_obj > 1:
                print(
                    f"[sweep] object position {obj_idx + 1}/{n_obj}: offset=({ox_off:+.3f},{oy_off:+.3f}) "
                    f"pose_xy=({fixed_object_pose_i[0]:.3f},{fixed_object_pose_i[1]:.3f})"
                )

            def eval_sweep_point(
                theta: np.ndarray, dim_name: str, value: float,
                _pose=fixed_object_pose_i, _tag=obj_tag,
            ) -> float:
                round_label = f"{_tag}_{dim_name}_val{value:g}" if _tag else f"{dim_name}_val{value:g}"
                reward_arr, mocap_pos_arr = rollout_batch_episode_returns(
                    policy,
                    envs[:1],
                    eval_cfg,
                    pre,
                    post,
                    theta.reshape(1, -1),
                    latent_z=latent_z,
                    fixed_object_pose=_pose,
                    fixed_object_shape=fixed_object_shape,
                    fixed_object_size=fixed_object_size,
                    fixed_init_qpos=fixed_init_qpos,
                    init_qpos_from_dataset=args.init_qpos_from_dataset,
                    dataset_dir=task_cfg.get("dataset_dir"),
                    num_episodes=task_cfg.get("num_episodes"),
                    show_rollout_progress=bool(args.show_rollout_progress),
                    rollout_desc=f"sweep {round_label}",
                    video_dir=video_dir,
                    round_label=round_label,
                    capture_mocap_pos=True,
                    video_layout=args.video_layout,
                    film_target=args.film_target,
                )
                sweep_records.append(
                    {
                        "dim_name": dim_name,
                        "value": float(value),
                        "episode_return": float(reward_arr[0]),
                        "mocap_pos": mocap_pos_arr[0],  # (T, 3)
                        "object_x": float(_pose[0]),
                        "object_y": float(_pose[1]),
                    }
                )
                return float(reward_arr[0])

            log_path = out_dir / (f"sweep_history_{obj_tag}.jsonl" if obj_tag else "sweep_history.jsonl")
            best_x_i, h_best_i, h_point_i, interrupted_exc, _elapsed_i = run_sweep(
                eval_sweep_point,
                opt_x0,
                dim_names=sweep_dim_names,
                sweep_values=sweep_values,
                log_path=log_path,
            )
            all_point_rewards.extend(h_point_i.tolist())
            if len(h_best_i) and float(h_best_i[-1]) > best_so_far_overall:
                best_so_far_overall = float(h_best_i[-1])
                best_x = best_x_i.copy()
            if interrupted_exc is not None:
                break  # stop sweeping further object positions too

        elapsed_sec = time.perf_counter() - sweep_t_start
        h_point = np.array(all_point_rewards)
        h_best = np.maximum.accumulate(h_point) if h_point.size else np.array([])
        np.savez(out_dir / "sweep_curves.npz", best_so_far=h_best, point_reward=h_point)
        _save_curve_png(out_dir / "reward_curve.png", h_best, "best_so_far", h_point, "point_reward", elapsed_sec=elapsed_sec)
        if sweep_records:
            np.savez(
                out_dir / "film_sweep_trajectories.npz",
                dim_name=np.array([r["dim_name"] for r in sweep_records]),
                value=np.array([r["value"] for r in sweep_records], dtype=np.float64),
                episode_return=np.array([r["episode_return"] for r in sweep_records], dtype=np.float64),
                mocap_pos=np.stack([r["mocap_pos"] for r in sweep_records], axis=0),  # (n_points, T, 3)
                object_x=np.array([r["object_x"] for r in sweep_records], dtype=np.float64),
                object_y=np.array([r["object_y"] for r in sweep_records], dtype=np.float64),
                dim_names=np.array(sweep_dim_names),
                sweep_values=sweep_values,
                object_x_values=np.array(object_x_values, dtype=np.float64),
                object_y_values=np.array(object_y_values, dtype=np.float64),
                theta_base=theta_base,
            )
            print(f"Saved {len(sweep_records)} trajectories to {out_dir / 'film_sweep_trajectories.npz'}")
            try:
                from visualize_film_sweep import render as render_sweep_viz
                render_sweep_viz(out_dir)
            except Exception as e:
                print(f"Warning: auto-visualization failed ({e}); run manually: python visualize_film_sweep.py --sweep_dir {out_dir}")
    elif args.method == "llm":
        if int(args.parallel) > 1:
            print("Warning: --method llm currently evaluates one candidate per iteration; extra parallel envs stay idle.")
        assert prompt_template_path is not None
        optimum_hint = float(args.llm_optimum_hint) if args.llm_optimum_hint is not None else default_llm_optimum_hint
        best_x, h_best, h_iter, interrupted_exc, elapsed_sec = run_llm(
            fitness,
            opt_x0,
            maxiter=int(args.llm_maxiter),
            seed=args.seed,
            log_path=out_dir / "llm_trace.jsonl",
            llm_model=args.llm_model,
            llm_temperature=float(args.llm_temperature),
            llm_max_retries=int(args.llm_max_retries),
            llm_max_tokens=int(args.llm_max_tokens) or None,
            llm_retry_temperature_bump=float(args.llm_retry_temperature_bump),
            llm_history_window=int(args.llm_history_window),
            llm_step_size_hint=float(args.llm_step_size_hint),
            llm_optimum_hint=optimum_hint,
            prompt_template_path=prompt_template_path,
            vlm_fitness_fn=fitness_and_frames if use_vlm else None,
            vlm_num_frames=int(args.llm_vlm_num_frames),
            rp_fitness_fn=fitness_and_video if use_reward_predictor else None,
            rp_instruction=args.llm_rp_instruction,
            rp_camera_labels=rp_camera_labels,
            rp_socket_path=args.llm_rp_socket_path,
            rp_timeout=float(args.llm_rp_timeout),
            rp_output_dir=(out_dir / "rp_feedback") if use_reward_predictor else None,
        )
        np.savez(out_dir / "llm_curves.npz", best_so_far=h_best, iter_reward=h_iter)
        _save_curve_png(
            out_dir / "reward_curve.png",
            h_best,
            "best_so_far",
            h_iter,
            "iter_reward",
            elapsed_sec=elapsed_sec,
            ylabel="progress_reward" if use_reward_predictor else "episode_return",
        )

    run_finished_dt = datetime.now()
    meta["elapsed_seconds"] = round(float(elapsed_sec), 3)
    meta["elapsed_human"] = _format_duration(elapsed_sec)
    meta["run_started_at"] = (run_finished_dt - timedelta(seconds=elapsed_sec)).isoformat()
    meta["run_finished_at"] = run_finished_dt.isoformat()
    meta["interrupted"] = interrupted_exc is not None
    with open(out_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    best_theta = np.asarray(best_x, dtype=np.float64)

    _apply_film_theta(policy, best_theta, k, target=args.film_target)
    W_attr, mu_attr, g_attr, b_attr = (_film_pca_attr(args.film_target, n) for n in ("W", "mu", "gamma", "beta"))
    film_ckpt = {
        "film_target": args.film_target,
        # Saved under fixed (unprefixed) keys regardless of film_target, for a stable
        # best_film_only.pt schema across targets — "film_target" above says which
        # detr_vae.py insertion point (load_film_pca(..., target=...)) these belong to.
        "film_pca_W": getattr(policy.model, W_attr).cpu(),
        "film_pca_mu": getattr(policy.model, mu_attr).cpu(),
        "film_pca_gamma": getattr(policy.model, g_attr).cpu(),
        "film_pca_beta": getattr(policy.model, b_attr).cpu(),
        "best_theta": torch.from_numpy(best_theta.astype(np.float32)),
        "film_bottleneck_dim": k,
    }
    torch.save(film_ckpt, out_dir / "best_film_only.pt")
    best_logged = float(np.max(h_best)) if len(h_best) else float("nan")
    best_logged_kind = "progress_reward" if use_reward_predictor else "episode_return"
    print(f"Done. Best {best_logged_kind} in log ≈ {best_logged}, total time {_format_duration(elapsed_sec)}")
    print(f"Saved: {out_dir / 'best_film_only.pt'}, {out_dir / 'reward_curve.png'}")

    if interrupted_exc is not None:
        if isinstance(interrupted_exc, KeyboardInterrupt):
            print("Interrupted by user (Ctrl+C); outputs above reflect the partial run.")
            sys.exit(0)
        else:
            # Re-raise so the traceback and a non-zero exit code still surface (this was a real
            # error, not a user-requested stop) — partial results/plots are already saved above.
            raise interrupted_exc


if __name__ == "__main__":
    main()
