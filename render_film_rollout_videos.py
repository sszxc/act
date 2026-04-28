#!/usr/bin/env python3
"""
Read run_meta.json and best_film_only.pt from a FiLM search dir and record two rollout videos:
  - Optimized FiLM (best_film_only.pt)
  - Architectural default FiLM (gamma=1, beta=0)

Writes output under run_dir subdirectory (default rollout_videos/).

Example:
  python render_film_rollout_videos.py --run_dir tmp/film_search_cma --seed 0

chunk_size / hidden_dim / temporal_agg etc. must match training/search (defaults or CLI override if missing in run_meta).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

from constants import DT, SIM_TASK_CONFIGS, DEFAULT_STATE_DIM
from imitate_episodes import rollout_single_episode_return, set_seed
from optimize_film_params import (
    _build_eval_config,
    _load_policy_and_stats,
    _parse_float_list,
    _parse_latent_z,
)
from sim_env import make_sim_env


def _load_best_film_theta(pt_path: Path) -> np.ndarray:
    d = torch.load(pt_path, map_location="cpu")
    if "best_theta" in d:
        t = d["best_theta"]
        if isinstance(t, torch.Tensor):
            return t.detach().float().cpu().numpy().reshape(-1)
        return np.asarray(t, dtype=np.float64).reshape(-1)
    g = d["visual_film_gamma"].detach().float().cpu().numpy().reshape(-1)
    b = d["visual_film_beta"].detach().float().cpu().numpy().reshape(-1)
    return np.concatenate([g, b], axis=0)


def _identity_film_theta(hidden_dim: int) -> np.ndarray:
    return np.concatenate(
        [np.ones(hidden_dim, dtype=np.float64), np.zeros(hidden_dim, dtype=np.float64)]
    )


def _safe_rename(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(f"missing expected file: {src}")
    if dst.is_file():
        dst.unlink()
    src.rename(dst)


def main():
    p = argparse.ArgumentParser(description="FiLM best vs identity rollout videos")
    p.add_argument("--run_dir", type=str, required=True, help="Directory with run_meta.json and best_film_only.pt")
    p.add_argument(
        "--videos_subdir",
        type=str,
        default="rollout_videos",
        help="Output subdirectory under run_dir",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stats_path", type=str, default=None)
    p.add_argument(
        "--ckpt_override",
        type=str,
        default=None,
        help="Override ckpt path from run_meta",
    )
    p.add_argument("--temporal_agg", action="store_true")
    p.add_argument("--latent_z_sample", type=str, default=None)
    p.add_argument(
        "--fixed_init_qpos",
        type=str,
        default=None,
        help="Comma or JSON list; same as during search",
    )
    p.add_argument(
        "--init_qpos_from_dataset",
        action="store_true",
    )
    p.add_argument("--policy_class", type=str, default="ACT")
    p.add_argument("--chunk_size", type=int, default=100)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--dim_feedforward", type=int, default=3200)
    p.add_argument("--latent_z_dim", type=int, default=32)
    p.add_argument("--kl_weight", type=float, default=10.0)
    p.add_argument("--quiet_rollout", action="store_true", help="Silence rollout step tqdm")
    args = p.parse_args()

    if args.policy_class != "ACT":
        print("FiLM applies only to ACT", file=sys.stderr)
        sys.exit(1)

    run_dir = Path(args.run_dir).resolve()
    meta_path = run_dir / "run_meta.json"
    film_pt = run_dir / "best_film_only.pt"
    if not meta_path.is_file():
        print(f"Missing {meta_path}", file=sys.stderr)
        sys.exit(1)
    if not film_pt.is_file():
        print(f"Missing {film_pt}", file=sys.stderr)
        sys.exit(1)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    task_name = meta["task_name"]
    if task_name not in SIM_TASK_CONFIGS:
        print(f"Unknown task_name: {task_name}", file=sys.stderr)
        sys.exit(1)
    task_cfg = SIM_TASK_CONFIGS[task_name]

    ckpt_path = Path(args.ckpt_override or meta["ckpt"]).resolve()
    meta_ckpt_resolved = Path(meta["ckpt"]).resolve()
    if args.ckpt_override and ckpt_path != meta_ckpt_resolved:
        warnings.warn(
            f"--ckpt_override ({ckpt_path}) differs from run_meta.ckpt ({meta_ckpt_resolved}).",
            UserWarning,
            stacklevel=1,
        )

    fixed_object_pose = np.asarray(meta["fixed_object_pose"], dtype=np.float64)
    fixed_init_qpos = (
        _parse_float_list(args.fixed_init_qpos) if args.fixed_init_qpos else None
    )

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
        "action_dim": task_cfg.get(
            "action_dim", task_cfg.get("state_dim", DEFAULT_STATE_DIM)
        ),
    }

    stats_path = Path(args.stats_path) if args.stats_path else None
    policy, stats, ckpt_loaded = _load_policy_and_stats(
        ckpt_path,
        stats_path,
        args.policy_class,
        policy_config,
    )

    hidden_dim = int(policy.model.visual_film_gamma.numel())
    best_theta = _load_best_film_theta(film_pt)
    if best_theta.size != 2 * hidden_dim:
        raise ValueError(
            f"best_film theta dim {best_theta.size} != 2*hidden_dim={2 * hidden_dim}"
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
    out_dir = run_dir / args.videos_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    identity_theta = _identity_film_theta(hidden_dim)

    results: list[dict] = []

    def do_rollout(label: str, film_theta: np.ndarray | None, rollout_id: int) -> tuple[float, float]:
        set_seed(args.seed)
        env = make_sim_env(task_name, time_limit=max_timesteps * DT)
        try:
            ep_ret, ep_hi = rollout_single_episode_return(
                policy,
                env,
                eval_cfg,
                pre,
                post,
                pca=None,
                use_pca_action=False,
                rollout_latent_z=latent_z,
                film_theta=film_theta,
                fixed_object_pose=fixed_object_pose,
                fixed_init_qpos=fixed_init_qpos,
                init_qpos_from_dataset=args.init_qpos_from_dataset,
                dataset_dir=task_cfg.get("dataset_dir"),
                num_episodes=task_cfg.get("num_episodes"),
                direct_replay=False,
                save_episode=True,
                output_dir=str(out_dir),
                rollout_id=rollout_id,
                quiet=args.quiet_rollout,
            )
        finally:
            del env
        results.append(
            {
                "label": label,
                "rollout_id": rollout_id,
                "episode_return": float(ep_ret),
                "episode_highest_reward": float(ep_hi),
            }
        )
        return ep_ret, ep_hi

    print("Recording best FiLM rollout...")
    do_rollout("best_film", best_theta, rollout_id=0)
    print("Recording identity FiLM baseline rollout...")
    do_rollout("identity_film_baseline", identity_theta, rollout_id=1)

    _safe_rename(out_dir / "video0.mp4", out_dir / "rollout_best_film.mp4")
    _safe_rename(out_dir / "video0_qpos.png", out_dir / "rollout_best_film_qpos.png")
    _safe_rename(out_dir / "video1.mp4", out_dir / "rollout_baseline_identity_film.mp4")
    _safe_rename(
        out_dir / "video1_qpos.png",
        out_dir / "rollout_baseline_identity_film_qpos.png",
    )

    summary = {
        "run_dir": str(run_dir),
        "videos_subdir": args.videos_subdir,
        "ckpt_loaded": ckpt_loaded,
        "ckpt_from_meta": meta.get("ckpt"),
        "task_name": task_name,
        "seed": args.seed,
        "fixed_object_pose": fixed_object_pose.tolist(),
        "baseline_description": "identity FiLM: gamma=1, beta=0 (DETRVAE buffer default)",
        "policy_overrides_note": (
            "chunk_size, hidden_dim, temporal_agg, latent_z, etc. must match search/train; "
            "this summary records CLI values used in this run."
        ),
        "cli": {
            "chunk_size": args.chunk_size,
            "hidden_dim": args.hidden_dim,
            "dim_feedforward": args.dim_feedforward,
            "latent_z_dim": args.latent_z_dim,
            "temporal_agg": args.temporal_agg,
            "init_qpos_from_dataset": args.init_qpos_from_dataset,
            "fixed_init_qpos_provided": bool(args.fixed_init_qpos),
        },
        "outputs": {
            "best_video": str(out_dir / "rollout_best_film.mp4"),
            "baseline_video": str(out_dir / "rollout_baseline_identity_film.mp4"),
        },
        "rollouts": results,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Done. Videos under: {out_dir}")
    print(f"Wrote summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
