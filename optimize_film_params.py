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
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import pickle
import re
import shlex
import shutil
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
from policy import ACTPolicy
from imitate_episodes import make_policy, set_seed
from sim_env import make_sim_env
from visualize_episodes import save_videos, _video_path_for_cam


def _progress(seq, *, enabled: bool, desc: str, total: int | None = None):
    if not enabled:
        return seq
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        return seq
    return tqdm(seq, desc=desc, total=total, dynamic_ncols=True, leave=False)


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
        raise ValueError(f"latent_z_sample dim {arr.size} != {latent_z_dim}")
    return torch.tensor(arr, dtype=torch.float32).cuda()


_FILM_TARGETS = ("visual", "memory", "hs")


def _film_pca_attr(target: str, name: str) -> str:
    """Maps a FiLM insertion target to its detr_vae.py buffer/policy-kwarg name: "visual" keeps
    the original unprefixed film_pca_* names (backward compat); "memory"/"hs" (candidates 4/5 —
    encoder-decoder boundary / pre-action_head) use film_pca_memory_*/film_pca_hs_*. These are
    also exactly the ACTPolicy.__call__/DETRVAE.forward() kwarg names for "gamma"/"beta"."""
    if target not in _FILM_TARGETS:
        raise ValueError(f"Unknown --film_target {target!r}; expected one of {_FILM_TARGETS}")
    prefix = "film_pca" if target == "visual" else f"film_pca_{target}"
    return f"{prefix}_{name}"


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
    allowed_missing = {
        "model.visual_film_gamma",
        "model.visual_film_beta",
        # PCA-bottleneck FiLM buffers: absent from any ckpt saved before this feature (and from
        # ckpts saved without --film_pca_path); load_film_pca() populates them post-hoc.
        # visual / memory (candidate 4, encoder-decoder boundary) / hs (candidate 5,
        # pre-action_head) — see detr_vae.py's load_film_pca(..., target=...).
        "model.film_pca_W",
        "model.film_pca_mu",
        "model.film_pca_gamma",
        "model.film_pca_beta",
        "model.film_pca_memory_W",
        "model.film_pca_memory_mu",
        "model.film_pca_memory_gamma",
        "model.film_pca_memory_beta",
        "model.film_pca_hs_W",
        "model.film_pca_hs_mu",
        "model.film_pca_hs_gamma",
        "model.film_pca_hs_beta",
    }
    missing = set(getattr(loading_status, "missing_keys", []))
    unexpected = set(getattr(loading_status, "unexpected_keys", []))
    if unexpected or (missing - allowed_missing):
        raise RuntimeError(f"load_state_dict failed: {loading_status}")
    policy.cuda()
    policy.eval()
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    return policy, stats, str(ckpt_path)


def _film_theta_from_policy(policy: ACTPolicy, target: str = "visual") -> np.ndarray:
    """theta = [gamma, beta] (each k-dim) — the only free params under the PCA-bottleneck FiLM
    mode, for whichever insertion target (see detr_vae.py's load_film_pca(..., target=...)):
    "visual" (default, before the Transformer encoder), "memory" (candidate 4, encoder-decoder
    boundary), or "hs" (candidate 5, pre-action_head). Requires load_film_pca() to have been
    called already for that target (film_pca_{target}_k > 0)."""
    k_attr, g_attr, b_attr = (_film_pca_attr(target, n) for n in ("k", "gamma", "beta"))
    if int(getattr(policy.model, k_attr)) <= 0:
        raise RuntimeError(
            f"policy.model has no PCA-bottleneck FiLM basis loaded for target={target!r}; "
            f"call load_film_pca(..., target={target!r}) first"
        )
    g = getattr(policy.model, g_attr).detach().float().cpu().numpy()
    b = getattr(policy.model, b_attr).detach().float().cpu().numpy()
    return np.concatenate([g, b], axis=0)


def _apply_film_theta(policy: ACTPolicy, theta: np.ndarray, k: int, target: str = "visual"):
    g_attr, b_attr = (_film_pca_attr(target, n) for n in ("gamma", "beta"))
    g_buf = getattr(policy.model, g_attr)
    b_buf = getattr(policy.model, b_attr)
    g = torch.from_numpy(theta[:k]).to(device=g_buf.device, dtype=g_buf.dtype)
    b = torch.from_numpy(theta[k : 2 * k]).to(device=b_buf.device, dtype=b_buf.dtype)
    with torch.no_grad():
        g_buf.copy_(g)
        b_buf.copy_(b)


def _split_film_theta_batch(theta_batch: np.ndarray, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    theta_batch: (B, 2*k) float64/float32 numpy.
    returns: (film_pca_gamma, film_pca_beta) as torch tensors on CUDA of shape (B, k)
    """
    tb = np.asarray(theta_batch, dtype=np.float32)
    if tb.ndim != 2 or tb.shape[1] != 2 * k:
        raise ValueError(f"theta_batch shape {tb.shape} expected (B, {2*k})")
    g = torch.from_numpy(tb[:, :k]).cuda()
    b = torch.from_numpy(tb[:, k:]).cuda()
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
    show_rollout_progress: bool = False,
    rollout_desc: str = "rollout",
    video_dir: Path | None = None,
    round_label: str = "round",
    capture_frames: bool = False,
    video_layout: str = "combined",
    film_target: str = "visual",
) -> np.ndarray | tuple[np.ndarray, list[list[dict]]]:
    """
    Step multiple envs in parallel (one theta each), batch observations per timestep on one GPU.
    Returns episode_return per trajectory (shape: (B,)).

    film_target: which PCA-bottleneck FiLM insertion point `thetas` searches — "visual"
    (default), "memory" (candidate 4, encoder-decoder boundary), or "hs" (candidate 5,
    pre-action_head). Must match whatever load_film_pca(..., target=film_target) installed on
    `policy.model` beforehand; see _film_pca_attr().

    If video_dir is not None, records video(s) per env to video_dir / f"{round_label}_env{i}.mp4".
    video_layout="combined" (default) writes one side-by-side mp4; "separate" writes one mp4
    per camera (e.g. round_0001_env0_default_cam.mp4).

    If capture_frames is True, also (or additionally, independent of video_dir) collects each
    env's per-timestep observation images and returns them as a second value:
    (rewards_sum, image_lists) where image_lists[i] is a list of {cam_name: HxWxC uint8} dicts,
    one per timestep, for env i (same format save_videos expects). Used e.g. to build VLM visual
    feedback without needing to encode/decode an mp4.
    """
    B = len(envs)
    if thetas.shape[0] != B:
        raise ValueError(f"thetas batch {thetas.shape[0]} != envs {B}")

    k_attr, g_attr, b_attr = (_film_pca_attr(film_target, n) for n in ("k", "gamma", "beta"))
    k = int(getattr(policy.model, k_attr))
    film_gamma_t, film_beta_t = _split_film_theta_batch(thetas, k)
    film_theta_kwargs = {g_attr: film_gamma_t, b_attr: film_beta_t}

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
                raise ValueError("init_qpos_from_dataset requires dataset_dir/num_episodes")
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
    record_video = video_dir is not None
    collect_images = record_video or capture_frames
    image_lists: list[list[dict]] | None = [[] for _ in range(B)] if collect_images else None

    # Same as imitate_episodes eval: inference_mode so temporal_agg does not keep
    # autograd over the full episode when stitching multi-step policy outputs into all_time_actions
    with torch.inference_mode():
        it = _progress(
            range(max_timesteps),
            enabled=bool(show_rollout_progress),
            desc=str(rollout_desc),
            total=max_timesteps,
        )
        for t in it:
            # build batch obs
            qpos_batch = []
            img_batch = []
            for i, ts in enumerate(ts_list):
                obs = ts.observation
                qpos_np = np.asarray(obs["qpos"], dtype=np.float32)[:state_dim]
                qpos_batch.append(pre_process(qpos_np))

                if collect_images:
                    image_lists[i].append(obs["images"])

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
                    **film_theta_kwargs,
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

    if record_video:
        assert video_dir is not None and image_lists is not None
        video_dir.mkdir(parents=True, exist_ok=True)
        for i in range(B):
            save_videos(
                image_lists[i],
                DT,
                video_path=str(video_dir / f"{round_label}_env{i}.mp4"),
                layout=video_layout,
            )

    if capture_frames:
        assert image_lists is not None
        return rewards_sum, image_lists
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


def _handle_optimizer_interrupt(tag: str, e: BaseException, it: int, n_completed: int) -> None:
    """Print a clear message when a run_* optimizer loop is interrupted (Ctrl+C or an
    unhandled exception mid-rollout). The caller still returns its partial history/best-so-far
    normally; main() decides whether to exit cleanly (KeyboardInterrupt) or re-raise (a real
    bug) after saving plots/checkpoints for whatever completed so far.
    """
    kind = "KeyboardInterrupt (Ctrl+C)" if isinstance(e, KeyboardInterrupt) else f"{type(e).__name__}: {e}"
    print(f"[{tag}] interrupted at iter {it} ({kind}); saving partial results ({n_completed} rounds completed)")


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
    out_dir = log_path.parent
    history_best = []
    history_iter_max = []
    best_so_far = -np.inf
    best_theta = theta.copy()

    t_start = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as flog:
        flog.write(
            f"# ARS dim={dim} sigma={sigma} alpha={alpha} n_pairs={n_pairs} "
            f"start_time={datetime.now().isoformat()}\n"
        )

    interrupted_exc: BaseException | None = None
    it = -1
    try:
        for it in range(n_iters):
            t0 = time.perf_counter()
            print(f"[ARS] iter {it}/{n_iters-1} starting; will eval ≈ {2*n_pairs + 1} rollouts")
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
            elapsed_now = time.perf_counter() - t_start
            with open(log_path, "a", encoding="utf-8") as flog:
                flog.write(
                    json.dumps(
                        {
                            "iter": it,
                            "best_so_far": best_so_far,
                            "iter_best": iter_best,
                            "elapsed_sec": round(elapsed_now, 3),
                        }
                    )
                    + "\n"
                )
            _save_progress_checkpoint(
                out_dir,
                "ars_curves.npz",
                {"best_so_far": np.array(history_best), "iter_max": np.array(history_iter_max)},
                np.array(history_best),
                "best_so_far",
                np.array(history_iter_max),
                "iter_max",
                elapsed_now,
            )
            dt = time.perf_counter() - t0
            print(f"[ARS] iter {it}: iter_best={iter_best:.4f} best_so_far={best_so_far:.4f} wall={dt:.2f}s")
    except (KeyboardInterrupt, Exception) as e:
        interrupted_exc = e
        _handle_optimizer_interrupt("ARS", e, it, len(history_best))

    elapsed_sec = time.perf_counter() - t_start
    return best_theta, np.array(history_best), np.array(history_iter_max), interrupted_exc, elapsed_sec


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
    ARS with batched fitness over multiple candidates (single-GPU batched forward).
    fitness_batch_fn: (N,dim)->(N,) episode_return
    """
    rng = np.random.default_rng(seed)
    theta = theta0.astype(np.float64).copy()
    dim = theta.size
    out_dir = log_path.parent
    best_so_far = -np.inf
    best_theta = theta.copy()
    history_best = []
    history_iter_best = []

    t_start = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as flog:
        flog.write(
            f"# ARS(batched) dim={dim} sigma={sigma} alpha={alpha} n_pairs={n_pairs} batch_size={batch_size} "
            f"start_time={datetime.now().isoformat()}\n"
        )

    interrupted_exc: BaseException | None = None
    it = -1
    try:
        for it in range(n_iters):
            t0 = time.perf_counter()
            eps_list = [rng.standard_normal(dim) for _ in range(n_pairs)]
            cand = []
            for eps in eps_list:
                cand.append(theta + sigma * eps)
                cand.append(theta - sigma * eps)
            # evaluate all perturbations
            print(f"[ARS(batched)] iter {it}/{n_iters-1} evaluating {len(cand)} candidates (batch_size={batch_size})")
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
            elapsed_now = time.perf_counter() - t_start
            with open(log_path, "a", encoding="utf-8") as flog:
                flog.write(
                    json.dumps(
                        {
                            "iter": it,
                            "best_so_far": best_so_far,
                            "iter_best": iter_best,
                            "elapsed_sec": round(elapsed_now, 3),
                        }
                    )
                    + "\n"
                )
            _save_progress_checkpoint(
                out_dir,
                "ars_curves.npz",
                {"best_so_far": np.array(history_best), "iter_max": np.array(history_iter_best)},
                np.array(history_best),
                "best_so_far",
                np.array(history_iter_best),
                "iter_max",
                elapsed_now,
            )
            dt = time.perf_counter() - t0
            r_mean = float(np.mean(rewards)) if rewards.size else float("nan")
            r_std = float(np.std(rewards)) if rewards.size else float("nan")
            print(
                f"[ARS(batched)] iter {it}: iter_best={iter_best:.4f} best_so_far={best_so_far:.4f} "
                f"mean±std={r_mean:.4f}±{r_std:.4f} wall={dt:.2f}s"
            )
    except (KeyboardInterrupt, Exception) as e:
        interrupted_exc = e
        _handle_optimizer_interrupt("ARS(batched)", e, it, len(history_best))

    elapsed_sec = time.perf_counter() - t_start
    return best_theta, np.asarray(history_best), np.asarray(history_iter_best), interrupted_exc, elapsed_sec


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
    out_dir = log_path.parent
    history_best = []
    history_gen_max = []
    best_so_far = -np.inf
    best_x = x0.copy()

    t_start = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as flog:
        flog.write(
            f"# CMA-ES dim={x0.size} sigma0={sigma0} maxiter={maxiter} popsize={getattr(es, 'popsize', None)} "
            f"start_time={datetime.now().isoformat()}\n"
        )

    interrupted_exc: BaseException | None = None
    gen = -1
    try:
        while not es.stop():
            t0 = time.perf_counter()
            xs = es.ask()
            gen = int(es.countiter)
            print(f"[CMA] gen {gen}/{maxiter-1} evaluating pop={len(xs)} (waiting for rollouts)")
            rewards = [fitness_fn(np.asarray(x, dtype=np.float64)) for x in xs]
            r_arr = np.asarray(list(map(float, rewards)), dtype=np.float64)
            es.tell(xs, [-float(r) for r in rewards])
            gen_max = float(np.max(r_arr)) if r_arr.size else float("nan")
            ib = int(np.argmax(r_arr)) if r_arr.size else 0
            if rewards[ib] > best_so_far:
                best_so_far = float(rewards[ib])
                best_x = np.asarray(xs[ib], dtype=np.float64).copy()
            history_best.append(best_so_far)
            history_gen_max.append(gen_max)
            elapsed_now = time.perf_counter() - t_start
            with open(log_path, "a", encoding="utf-8") as flog:
                flog.write(
                    json.dumps(
                        {
                            "generation": gen,
                            "params": np.asarray(xs[ib], dtype=np.float64).tolist(),
                            "gen_max": gen_max,
                            "best_so_far": best_so_far,
                            "elapsed_sec": round(elapsed_now, 3),
                        }
                    )
                    + "\n"
                )
            _save_progress_checkpoint(
                out_dir,
                "cma_curves.npz",
                {"best_so_far": np.array(history_best), "gen_max": np.array(history_gen_max)},
                np.array(history_best),
                "best_so_far",
                np.array(history_gen_max),
                "gen_max",
                elapsed_now,
            )
            dt = time.perf_counter() - t0
            r_mean = float(np.mean(r_arr)) if r_arr.size else float("nan")
            r_std = float(np.std(r_arr)) if r_arr.size else float("nan")
            print(
                f"[CMA] gen {gen}: gen_max={gen_max:.4f} best_so_far={best_so_far:.4f} "
                f"mean±std={r_mean:.4f}±{r_std:.4f} wall={dt:.2f}s"
            )
    except (KeyboardInterrupt, Exception) as e:
        interrupted_exc = e
        _handle_optimizer_interrupt("CMA", e, gen, len(history_best))

    elapsed_sec = time.perf_counter() - t_start
    return best_x, np.array(history_best), np.array(history_gen_max), interrupted_exc, elapsed_sec


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
    out_dir = log_path.parent
    best_so_far = -np.inf
    best_x = x0.copy()
    history_best = []
    history_gen_max = []

    t_start = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as flog:
        flog.write(
            f"# CMA-ES(batched) dim={x0.size} sigma0={sigma0} maxiter={maxiter} popsize={getattr(es,'popsize',None)} "
            f"batch_size={batch_size} start_time={datetime.now().isoformat()}\n"
        )

    interrupted_exc: BaseException | None = None
    gen = -1
    try:
        while not es.stop():
            t0 = time.perf_counter()
            xs = es.ask()
            rewards = []
            gen = int(es.countiter)
            print(f"[CMA(batched)] gen {gen}/{maxiter-1} evaluating pop={len(xs)} (batch_size={batch_size})")
            for i in range(0, len(xs), batch_size):
                tb = np.stack([np.asarray(x, dtype=np.float64) for x in xs[i : i + batch_size]], axis=0)
                rewards.extend(list(map(float, fitness_batch_fn(tb))))
            rewards = np.asarray(rewards, dtype=np.float64)
            es.tell(xs, list((-rewards).astype(float)))

            gen_max = float(np.max(rewards))
            ib = int(np.argmax(rewards))
            if rewards[ib] > best_so_far:
                best_so_far = float(rewards[ib])
                best_x = np.asarray(xs[ib], dtype=np.float64).copy()
            history_best.append(best_so_far)
            history_gen_max.append(gen_max)
            elapsed_now = time.perf_counter() - t_start
            with open(log_path, "a", encoding="utf-8") as flog:
                flog.write(
                    json.dumps(
                        {
                            "generation": gen,
                            "params": np.asarray(xs[ib], dtype=np.float64).tolist(),
                            "gen_max": gen_max,
                            "best_so_far": best_so_far,
                            "elapsed_sec": round(elapsed_now, 3),
                        }
                    )
                    + "\n"
                )
            _save_progress_checkpoint(
                out_dir,
                "cma_curves.npz",
                {"best_so_far": np.array(history_best), "gen_max": np.array(history_gen_max)},
                np.array(history_best),
                "best_so_far",
                np.array(history_gen_max),
                "gen_max",
                elapsed_now,
            )
            dt = time.perf_counter() - t0
            r_mean = float(np.mean(rewards)) if rewards.size else float("nan")
            r_std = float(np.std(rewards)) if rewards.size else float("nan")
            print(
                f"[CMA(batched)] gen {gen}: gen_max={gen_max:.4f} best_so_far={best_so_far:.4f} "
                f"mean±std={r_mean:.4f}±{r_std:.4f} wall={dt:.2f}s"
            )
    except (KeyboardInterrupt, Exception) as e:
        interrupted_exc = e
        _handle_optimizer_interrupt("CMA(batched)", e, gen, len(history_best))

    elapsed_sec = time.perf_counter() - t_start
    return best_x, np.asarray(history_best), np.asarray(history_gen_max), interrupted_exc, elapsed_sec


def _format_duration(seconds: float) -> str:
    total = int(round(max(0.0, seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _save_curve_png(
    path: Path,
    y1: np.ndarray,
    y1_label: str,
    y2: np.ndarray | None,
    y2_label: str | None,
    elapsed_sec: float | None = None,
    ylabel: str = "episode_return",
):
    if len(y1) == 0:
        print(f"[plot] skipping {path.name}: no completed rounds to plot")
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(len(y1)), y1, label=y1_label)
    if y2 is not None:
        ax.plot(np.arange(len(y2)), y2, alpha=0.6, label=y2_label or "iter/gen max")
    ax.set_xlabel("iteration / generation")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if elapsed_sec is not None:
        ax.text(
            0.99,
            0.02,
            f"total time: {_format_duration(elapsed_sec)}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            alpha=0.7,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_progress_checkpoint(
    out_dir: Path,
    npz_name: str,
    npz_arrays: dict,
    y1: np.ndarray,
    y1_label: str,
    y2: np.ndarray | None,
    y2_label: str | None,
    elapsed_sec: float,
    ylabel: str = "episode_return",
) -> None:
    """Persist the optimizer's progress so far (curves .npz + reward_curve.png), overwriting the
    previous snapshot in place. Called after every completed iteration/generation (not just at the
    end or on interrupt) so a run killed mid-flight still leaves usable, up-to-date results on
    disk, and so the plot is viewable while the run is still going."""
    np.savez(out_dir / npz_name, **npz_arrays)
    _save_curve_png(out_dir / "reward_curve.png", y1, y1_label, y2, y2_label, elapsed_sec=elapsed_sec, ylabel=ylabel)


def _load_num_optim_prompt_template(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    m = re.search(r'system_prompt\s*=\s*"""(.*?)"""', text, flags=re.S)
    if m is None:
        # Allow "raw prompt" files (plain text) too.
        # This enables alternative prompt templates that are not wrapped in a python variable.
        return text.strip()
    return m.group(1).strip()


def _prompt_template_is_vlm(path: Path) -> bool:
    """True if the prompt template file declares `IS_VLM = True` (see
    prompts/num_optim_Pratyush_vlm.py), i.e. it expects a visual-feedback image attached
    to every LLM call from iteration 1 onward, in addition to the text history."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return bool(re.search(r"^\s*IS_VLM\s*=\s*True\s*$", text, flags=re.M))


def _prompt_template_uses_reward_predictor(path: Path) -> bool:
    """True if the prompt template file declares `IS_REWARD_PREDICTOR = True` (see
    prompts/num_optim_Pratyush_reward_predictor.py), i.e. it expects the local
    reward_predictor's text feedback (over its IPC bridge) on every LLM call from
    iteration 1 onward, in addition to the text history."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return bool(re.search(r"^\s*IS_REWARD_PREDICTOR\s*=\s*True\s*$", text, flags=re.M))


def _format_history_text_for_prompt(history: list[dict], rank: int) -> str:
    """Format past evaluations for prompts/num_optim_Pratyush.py (maximize episode return)."""
    if not history:
        return "No previous samples yet."
    lines = []
    for rec in history:
        params = rec["params"]
        r = float(rec["reward"])
        params_str = ", ".join([f"params[{i}]: {params[i]:.2f}" for i in range(rank)])
        lines.append(f"iter {rec['iter']}: {params_str}, R(params): {r:.6f}")
    return "\n".join(lines)


def _render_num_optim_prompt(
    template: str,
    *,
    rank: int,
    optimum_reward: float,
    step_size: float,
    episode_num: int,
    total_episodes: int,
    history_text: str,
    reward_predictor_text: str = "",
) -> str:
    """Fill placeholders for prompts/num_optim_Pratyush.py-style templates (no Jinja).
    `reward_predictor_text` fills {{ reward_predictor_text }}, used only by
    prompts/num_optim_Pratyush_reward_predictor.py-style templates (no-op otherwise)."""
    rendered = template
    rendered = rendered.replace("{{ rank - 1 }}", str(rank - 1))
    rendered = rendered.replace("{{ rank }}", str(rank))

    rendered = re.sub(
        r"\{\{\s*optimum_reward:\.1f\s*\}\}",
        f"{float(optimum_reward):.1f}",
        rendered,
    )
    rendered = re.sub(r"\{\{\s*optimum_reward\s*\}\}", f"{float(optimum_reward)}", rendered)

    rendered = re.sub(r"\{\{\s*step_size\s*\}\}", str(float(step_size)), rendered)
    rendered = re.sub(r"\{\{\s*episode_num\s*\}\}", str(int(episode_num)), rendered)
    rendered = re.sub(r"\{\{\s*total_episodes\s*\}\}", str(int(total_episodes)), rendered)
    rendered = re.sub(r"\{\{\s*history_text\s*\}\}", history_text, rendered)
    rendered = re.sub(r"\{\{\s*reward_predictor_text\s*\}\}", reward_predictor_text, rendered)

    rendered = re.sub(r"<start_of_turn>.*$", "", rendered, flags=re.M).strip()
    return rendered


def _parse_llm_params_response(text: str, rank: int) -> np.ndarray | None:
    if not text:
        return None
    m = re.search(r"<param\b[^>]*>(.*?)</param\s*>", text, flags=re.S | re.I)
    scope = m.group(1) if m else text
    matches = re.findall(r"params\[(\d+)\]\s*:\s*([-+]?\d*\.?\d+)", scope)
    if matches:
        vals = np.full(rank, np.nan, dtype=np.float64)
        for idx_str, val_str in matches:
            idx = int(idx_str)
            if 0 <= idx < rank and np.isnan(vals[idx]):
                vals[idx] = float(val_str)
        if np.all(np.isfinite(vals)):
            return vals
    raw_nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", scope)
    if len(raw_nums) >= rank:
        return np.asarray([float(x) for x in raw_nums[:rank]], dtype=np.float64)
    return None


def _clip_quantize_params(x: np.ndarray, low: float = -6.0, high: float = 6.0, decimals: int = 1) -> np.ndarray:
    return np.round(np.clip(np.asarray(x, dtype=np.float64), low, high), decimals)


def _sample_unseen_params(rng: np.random.Generator, rank: int, seen: set[tuple[float, ...]]) -> np.ndarray:
    for _ in range(4096):
        cand = rng.integers(-600, 601, size=rank).astype(np.float64) / 100.0
        key = tuple(float(v) for v in cand.tolist())
        if key not in seen:
            return cand
    # Extremely unlikely fallback.
    cand = rng.uniform(-6.0, 6.0, size=rank)
    return _clip_quantize_params(cand)


def _init_openai_compatible_client(base_url: str, api_key: str):
    try:
        from openai import OpenAI
        import httpx
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Please install openai package for --method llm: `pip install openai`") from e
    # Library default read timeout is 600s; a stuck-but-connected SOL request can silently
    # block that long per attempt. 300s still gives "thinking" models room to generate while
    # failing a truly hung request in reasonable time. connect stays short (5s) so a fully-down
    # endpoint fails fast.
    timeout = httpx.Timeout(300.0, connect=5.0)
    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)


def _maybe_load_dotenv():
    # Optional: allow local `.env` without forcing a dependency.
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    load_dotenv(override=False)


def _get_env(name: str) -> str | None:
    v = os.getenv(name)
    if v is None:
        return None
    v = str(v).strip()
    return v if v else None


def _build_vlm_feedback_image(image_dicts: list[dict], num_frames: int = 5) -> np.ndarray:
    """Sample `num_frames` evenly-spaced timesteps from one rollout's captured per-step camera
    images (same list produced by rollout_batch_episode_returns(..., capture_frames=True)),
    label each with its frame index, and stack them top-to-bottom into a single RGB uint8
    composite image. Multiple cameras appear side-by-side within each row, matching
    visualize_episodes.save_videos' frame layout.
    """
    import cv2  # local import: only needed for the VLM prompt path

    total = len(image_dicts)
    if total == 0:
        raise ValueError("No frames captured for VLM feedback")
    n = max(1, min(num_frames, total))
    indices = [total - 1] if n == 1 else sorted({round(i * (total - 1) / (n - 1)) for i in range(n)})

    cam_names = list(image_dicts[0].keys())
    rows = []
    for idx in indices:
        cams = [np.asarray(image_dicts[idx][cam]) for cam in cam_names]
        row = np.concatenate(cams, axis=1).copy()  # side-by-side cameras, HxWxC uint8 RGB
        label = f"frame {idx}"
        cv2.putText(row, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(row, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        rows.append(row)

    max_w = max(r.shape[1] for r in rows)
    if any(r.shape[1] != max_w for r in rows):
        rows = [
            r if r.shape[1] == max_w else np.pad(r, ((0, 0), (0, max_w - r.shape[1]), (0, 0)))
            for r in rows
        ]
    return np.concatenate(rows, axis=0)


def _encode_image_array_to_data_uri(img: np.ndarray, quality: int = 90) -> str:
    """RGB uint8 HxWxC array -> `data:image/jpeg;base64,...` string usable as an image_url."""
    import cv2  # local import: only needed for the VLM prompt path

    bgr = np.ascontiguousarray(img[:, :, ::-1])  # RGB -> BGR for cv2.imencode
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to JPEG-encode VLM feedback image")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _score_video_with_reward_predictor(
    video_paths: list[str],
    *,
    instruction: str,
    camera_labels: list[str],
    socket_path: str,
    timeout: float,
    output_dir: Path,
) -> dict:
    """Score one rollout's synchronized per-camera videos via the local reward_predictor's
    IPC bridge: whole-video scoring (runner="main_single", not the process-reward-curve
    runner), subgoals auto-generated from `instruction` on every call (no `subgoals=`
    passed). Requires the reward_predictor repo to already be on sys.path (see main())
    and its `python -m ipc.server` running with OPENAI_API_KEY sourced (for subgoal
    generation). Returns {"progress_reward", "success_score", "components"}."""
    from ipc.client import score_trajectory  # local import: only needed for this path

    if len(video_paths) != len(camera_labels):
        raise ValueError(
            f"got {len(video_paths)} videos but {len(camera_labels)} camera_labels"
        )
    if not 1 <= len(video_paths) <= 5:
        raise ValueError(f"reward_predictor supports 1-5 synchronized videos, got {len(video_paths)}")
    video_kwargs = {f"video{i}": p for i, p in enumerate(video_paths, start=1)}

    return score_trajectory(
        instruction,
        **video_kwargs,
        runner="main_single",
        camera_labels=camera_labels,
        socket_path=socket_path,
        timeout=timeout,
        output_dir=str(output_dir),
    )


def _format_reward_predictor_feedback_text(result: dict) -> str:
    progress = float(result.get("progress_reward", float("nan")))
    success = float(result.get("success_score", float("nan")))
    lines = [f"progress_reward: {progress:.3f}, success_score: {success:.3f}"]
    for name, prob in (result.get("components") or {}).items():
        lines.append(f"  {name}: {float(prob):.3f}")
    return "\n".join(lines)


def _call_llm_next_params(
    client,
    *,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int | None = None,
    image_data_uri: str | None = None,
) -> tuple[str, str]:
    """Returns (content, reasoning) — reasoning is the model's separate thinking-channel
    text when the backend exposes one (see below), else ""."""
    if image_data_uri:
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data_uri}},
        ]
    else:
        content = prompt
    kwargs = {}
    if max_tokens is not None:
        # "Thinking" models spend generation budget on a <think> block before the final
        # answer; too small a budget (or an unset one, on some backends) truncates mid-thought
        # and message.content comes back empty (finish_reason="length") even though the API
        # call itself succeeded. An explicit, generous budget makes that failure mode rare and,
        # when it still happens, distinguishable via finish_reason rather than silently empty.
        kwargs["max_tokens"] = int(max_tokens)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=float(temperature),
        **kwargs,
    )
    message = resp.choices[0].message
    content_resp = message.content
    # "Thinking" models served over OpenAI-compatible backends (vLLM/SGLang reasoning
    # parsers, etc.) typically split their chain-of-thought out of `message.content` and
    # into a separate field — most commonly `reasoning_content`, occasionally `reasoning`.
    # Without capturing it, the logged response is just the bare <param> block with no way
    # to audit whether the model actually engaged with the required <think> analysis (state/
    # visual/trend/sensitivity/mode) or looked at the attached image.
    reasoning_resp = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
    return (content_resp if content_resp is not None else ""), (reasoning_resp or "")


def run_llm(
    fitness_fn,
    x0: np.ndarray,
    *,
    maxiter: int,
    seed: int,
    log_path: Path,
    llm_model: str,
    llm_temperature: float,
    llm_max_retries: int,
    llm_max_tokens: int | None,
    llm_retry_temperature_bump: float,
    llm_history_window: int,
    llm_step_size_hint: float,
    llm_optimum_hint: float,
    prompt_template_path: Path,
    vlm_fitness_fn=None,
    vlm_num_frames: int = 5,
    rp_fitness_fn=None,
    rp_instruction: str | None = None,
    rp_camera_labels: list[str] | None = None,
    rp_socket_path: str = "/tmp/reward_predictor_ipc.sock",
    rp_timeout: float = 3600.0,
    rp_output_dir: Path | None = None,
):
    """
    vlm_fitness_fn: optional Callable[[np.ndarray], tuple[float, list[dict]]] — same role as
    fitness_fn but also returns the evaluated candidate's per-timestep captured images (see
    rollout_batch_episode_returns(..., capture_frames=True)). When given (i.e. the prompt
    template sets IS_VLM = True, see prompts/num_optim_Pratyush_vlm.py), it replaces fitness_fn
    for every evaluation, and a composite image built from vlm_num_frames evenly-spaced frames
    of the most recently evaluated rollout is attached to every LLM call from iteration 1 onward
    (the text history passed in the prompt still spans all iterations; only the image is
    last-round-only).

    rp_fitness_fn: optional Callable[[np.ndarray], tuple[float, list[str]]] — mutually exclusive
    with vlm_fitness_fn. When given (i.e. the prompt template sets IS_REWARD_PREDICTOR = True, see
    prompts/num_optim_Pratyush_reward_predictor.py), it replaces fitness_fn for every rollout, but
    NOT the reward: its own return value (episode_return) is only kept as diagnostic history
    metadata, never used as R(params). Instead, the local reward_predictor (over its IPC bridge,
    rp_socket_path) scores the returned per-camera videos (one per rp_camera_labels entry, kept
    separate — never merged into one side-by-side frame) against auto-generated subgoals for
    rp_instruction, and R(params) becomes that call's progress_reward. A text breakdown of the
    score (success_score + per-subgoal probabilities) is attached to every LLM call from
    iteration 1 onward (last-round-only, same cadence as vlm_fitness_fn's image).
    """
    rank = int(np.asarray(x0).size)
    out_dir = log_path.parent
    rng = np.random.default_rng(seed)
    _maybe_load_dotenv()
    base_url = _get_env("OPENAI_BASE_URL") or "https://openai.rc.asu.edu/v1"
    api_key = _get_env("OPENAI_API_KEY")
    if api_key is None:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Set it in your shell (recommended) or in a local `.env` file.\n"
            "Example:\n"
            "  export OPENAI_API_KEY='...'\n"
            "Optional:\n"
            "  export OPENAI_BASE_URL='https://openai.rc.asu.edu/v1'"
        )
    client = _init_openai_compatible_client(
        base_url=base_url,
        api_key=api_key,
    )
    template = _load_num_optim_prompt_template(prompt_template_path)
    prompt_log_dir = log_path.parent / "llm_prompt_logs"
    prompt_log_dir.mkdir(parents=True, exist_ok=True)
    if prompt_template_path.is_file():
        shutil.copy2(prompt_template_path, prompt_log_dir / prompt_template_path.name)

    seen: set[tuple[float, ...]] = set()
    history: list[dict] = []
    history_best = []
    history_iter_reward = []
    best_so_far = -np.inf
    best_x = _clip_quantize_params(x0)
    use_vlm = vlm_fitness_fn is not None
    last_vlm_image_uri: str | None = None  # visual feedback from the previous iteration's rollout only
    use_reward_predictor = rp_fitness_fn is not None
    last_rp_feedback_text = "No reward-predictor feedback yet (first iteration)."

    t_start = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as flog:
        flog.write(
            f"# LLM dim={rank} maxiter={maxiter} model={llm_model} temp={llm_temperature} "
            f"max_retries={llm_max_retries} max_tokens={llm_max_tokens} "
            f"retry_temperature_bump={llm_retry_temperature_bump} start_time={datetime.now().isoformat()}\n"
        )

    interrupted_exc: BaseException | None = None
    it = -1
    try:
        for it in range(maxiter):
            t0 = time.perf_counter()
            raw_response = ""
            source = "seed"
            extra_history_fields: dict = {}
            if it == 0:
                cand = _clip_quantize_params(x0)
            else:
                prompt_hist = history[-llm_history_window:] if llm_history_window > 0 else history
                history_text = _format_history_text_for_prompt(prompt_hist, rank)
                prompt = _render_num_optim_prompt(
                    template,
                    rank=rank,
                    optimum_reward=float(llm_optimum_hint),
                    step_size=llm_step_size_hint,
                    episode_num=it + 1,
                    total_episodes=maxiter,
                    history_text=history_text,
                    reward_predictor_text=last_rp_feedback_text,
                )
                (prompt_log_dir / f"rendered_prompt_iter_{it:04d}.txt").write_text(prompt, encoding="utf-8")
                cand = None
                n_tries = max(1, int(llm_max_retries))
                every_call_raised = True
                saw_parseable_vector = False
                attempt_responses: list[str] = []
                for retry in range(n_tries):
                    if retry > 0:
                        # Backoff before re-hitting SOL after any failed attempt (error, parse
                        # failure, or duplicate) — the openai SDK already backs off *within* one
                        # call's internal retries, but nothing previously paced *these* app-level
                        # retries, so a down/rate-limited endpoint got hit back-to-back.
                        # 10s, 20s, 40s, 80s, 160s, then plateaus (cap = 5th doubling).
                        backoff_s = min(10.0 * 2.0 ** (retry - 1), 160.0)
                        print(f"[LLM] iter {it}: previous attempt failed, backing off {backoff_s:.0f}s before retry {retry}")
                        time.sleep(backoff_s)
                    # Retries reuse the exact same prompt, so at low temperature the model tends
                    # to just repeat its previous (already-seen) answer, or the same truncated-
                    # empty response, verbatim — bumping temperature per retry gives it a real
                    # chance to land somewhere different instead of burning the retry budget on
                    # a near-deterministic repeat.
                    retry_temperature = min(2.0, llm_temperature + retry * llm_retry_temperature_bump)
                    try:
                        print(
                            f"[LLM] iter {it}/{maxiter-1} requesting next params from model={llm_model} "
                            f"(retry {retry}/{n_tries - 1}, temp={retry_temperature:.2f}) — waiting for optimizer response"
                        )
                        raw_response, reasoning_response = _call_llm_next_params(
                            client,
                            model=llm_model,
                            prompt=prompt,
                            temperature=retry_temperature,
                            max_tokens=llm_max_tokens,
                            image_data_uri=last_vlm_image_uri if use_vlm else None,
                        )
                        attempt_log = f"--- retry {retry} (temp={retry_temperature:.2f}) ---\n"
                        if reasoning_response:
                            # Logged for audit only — never fed back into history_text or parsed
                            # for <param>, so it can't change downstream behavior.
                            attempt_log += f"--- reasoning ---\n{reasoning_response}\n--- content ---\n"
                        attempt_log += raw_response
                        attempt_responses.append(attempt_log)
                        every_call_raised = False
                        parsed = _parse_llm_params_response(raw_response, rank)
                        if parsed is None:
                            continue
                        saw_parseable_vector = True
                        parsed = _clip_quantize_params(parsed)
                        key = tuple(float(v) for v in parsed.tolist())
                        if key in seen:
                            continue
                        cand = parsed
                        source = f"llm_success_after_retry_{retry}" if retry > 0 else "llm_success"
                        break
                    except Exception as e:
                        raw_response = f"__llm_error__: {e}"
                        attempt_responses.append(f"--- retry {retry} (temp={retry_temperature:.2f}) ---\n{raw_response}")
                (prompt_log_dir / f"rendered_response_iter_{it:04d}.txt").write_text(
                    "\n\n".join(attempt_responses), encoding="utf-8"
                )
                if cand is None:
                    cand = _sample_unseen_params(rng, rank, seen)
                    # Distinguish why the LLM path failed (logged in source / optional fallback_reason).
                    if saw_parseable_vector:
                        source = "duplicate_after_retries"
                    elif every_call_raised:
                        source = "api_error"
                    else:
                        source = "parse_failure"

            cand = _clip_quantize_params(cand)
            key = tuple(float(v) for v in cand.tolist())
            if key in seen:
                cand = _sample_unseen_params(rng, rank, seen)
                key = tuple(float(v) for v in cand.tolist())
                source = f"{source}_dedup"

            if use_vlm:
                reward, frames = vlm_fitness_fn(cand)
                reward = float(reward)
                try:
                    composite = _build_vlm_feedback_image(frames, num_frames=vlm_num_frames)
                    last_vlm_image_uri = _encode_image_array_to_data_uri(composite)
                    # Save every round's composite unconditionally (not just when it ends up
                    # attached to a later prompt) so round 0 and the final round are covered too.
                    (prompt_log_dir / f"vlm_feedback_round_{it:04d}.jpg").write_bytes(
                        base64.b64decode(last_vlm_image_uri.split(",", 1)[1])
                    )
                except Exception as e:
                    print(f"[LLM] iter {it}: warning: failed to build VLM feedback image: {e}")
                    last_vlm_image_uri = None
            elif use_reward_predictor:
                # R(params) is now the reward_predictor's progress_reward, not episode_return — a
                # rollout still happens (to produce the video) but its own env reward is only kept
                # as diagnostic history metadata below, never as the optimization objective. A
                # scoring failure here is fatal to this round (there is no fallback reward to fall
                # back on) and is intentionally left to propagate to the run's outer except-block,
                # same as any other fitness-evaluation failure.
                rollout_return, video_paths = rp_fitness_fn(cand)
                rp_result = _score_video_with_reward_predictor(
                    video_paths,
                    instruction=rp_instruction,
                    camera_labels=rp_camera_labels,
                    socket_path=rp_socket_path,
                    timeout=rp_timeout,
                    output_dir=rp_output_dir / f"round_{it:04d}",
                )
                reward = float(rp_result["progress_reward"])
                extra_history_fields["rollout_episode_return"] = float(rollout_return)
                extra_history_fields["rp_success_score"] = float(rp_result.get("success_score", float("nan")))
                last_rp_feedback_text = _format_reward_predictor_feedback_text(rp_result)
                (prompt_log_dir / f"rp_feedback_round_{it:04d}.json").write_text(
                    json.dumps(rp_result, indent=2), encoding="utf-8"
                )
            else:
                reward = float(fitness_fn(cand))
            seen.add(key)

            if reward > best_so_far:
                best_so_far = reward
                best_x = cand.copy()

            history.append(
                {
                    "iter": it,
                    "params": [float(v) for v in cand.tolist()],
                    "reward": reward,
                    "best_so_far": float(best_so_far),
                    "source": source,
                    "raw_response": raw_response,
                    "elapsed_sec": round(time.perf_counter() - t_start, 3),
                    **extra_history_fields,
                }
            )
            history_best.append(float(best_so_far))
            history_iter_reward.append(float(reward))

            with open(log_path, "a", encoding="utf-8") as flog:
                flog.write(json.dumps(history[-1], ensure_ascii=False) + "\n")
            elapsed_now = time.perf_counter() - t_start
            _save_progress_checkpoint(
                out_dir,
                "llm_curves.npz",
                {"best_so_far": np.array(history_best), "iter_reward": np.array(history_iter_reward)},
                np.array(history_best),
                "best_so_far",
                np.array(history_iter_reward),
                "iter_reward",
                elapsed_now,
                ylabel="progress_reward" if use_reward_predictor else "episode_return",
            )
            dt = time.perf_counter() - t0
            reward_kind = "progress_reward" if use_reward_predictor else "reward"
            print(f"[LLM] iter {it}: {reward_kind}={reward:.4f} best_so_far={best_so_far:.4f} source={source} wall={dt:.2f}s")
    except (KeyboardInterrupt, Exception) as e:
        interrupted_exc = e
        _handle_optimizer_interrupt("LLM", e, it, len(history_best))

    elapsed_sec = time.perf_counter() - t_start
    return best_x, np.asarray(history_best), np.asarray(history_iter_reward), interrupted_exc, elapsed_sec


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
    p.add_argument("--method", type=str, choices=("cma", "ars", "llm"), required=True)
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
    parent_dir = Path(args.output_dir) if args.output_dir is not None else Path(args.ckpt).resolve().parent
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
        "env_max_reward": env_max_reward,
    }
    if args.save_videos:
        meta["videos_dir"] = "videos"
        meta["video_layout"] = args.video_layout
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
    else:
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
