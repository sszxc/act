#!/usr/bin/env python3
"""
Offline PCA fit for the PCA-bottleneck FiLM mode (see detr_vae.py's film_pca_* buffers and
optimize_film_params.py --film_pca_path / --film_bottleneck_dim).

Hooks policy.model.input_proj to record the pre-FiLM visual feature map (backbone + input_proj
output, before the elementwise gamma/beta applied at detr_vae.py's `src = src * gamma + beta`),
sampled pixel-by-pixel (NOT spatially pooled — FiLM is applied at every spatial location, so the
PCA basis should be fit on that same per-location distribution) from demonstration images (HDF5
episodes already used for training), pooled across all cameras (FiLM is shared across cams in
this architecture, so a single basis is fit over their union rather than one basis per camera).

Fits PCA up to --max_k components and saves W (hidden_dim, max_k), mu (hidden_dim,), and metadata
to a single .npz artifact. optimize_film_params.py --film_pca_path/--film_bottleneck_dim slices
W[:, :k] from this file at search time — no need to re-run this script just to change k, only
when the ckpt/task (i.e. the activation distribution) changes.

No sim rollout needed — this only runs the vision backbone forward on recorded demo frames.

Example:
  python fit_film_pca.py --ckpt results/sim_transfer_cube_scripted/policy_best.ckpt \\
    --task_name sim_transfer_cube_scripted --num_episodes 20 --frames_per_episode 30 \\
    --points_per_frame 16 --max_k 64 --output tmp/film_pca/sim_transfer_cube_scripted.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.decomposition import PCA

from constants import SIM_TASK_CONFIGS, DEFAULT_STATE_DIM
from optimize_film_params import _load_policy_and_stats


def _collect_activations(
    policy,
    dataset_dir: str,
    camera_names: list[str],
    episode_ids: list[int],
    frames_per_episode: int,
    points_per_frame: int,
    rng: np.random.Generator,
    *,
    show_progress: bool = False,
) -> np.ndarray:
    """Per-pixel, per-camera samples of the pre-FiLM visual feature (hidden_dim,), pooled across
    all cameras/episodes/timesteps into one (N, hidden_dim) array. Mirrors detr_vae.py's
    forward(): features, pos = self.backbones[0](image[:, cam_id]); self.input_proj(features[0]);
    called directly here (skipping the transformer/action head) since only input_proj's output is
    needed.
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    captured: dict[str, torch.Tensor] = {}

    def _hook(module, inp, out):
        captured["feat"] = out.detach()

    handle = policy.model.input_proj.register_forward_hook(_hook)
    chunks: list[np.ndarray] = []
    try:
        with torch.inference_mode():
            it = episode_ids
            if show_progress:
                try:
                    from tqdm import tqdm  # type: ignore

                    it = tqdm(episode_ids, desc="episodes")
                except Exception:
                    pass
            for ep_id in it:
                path = Path(dataset_dir) / f"episode_{ep_id}.hdf5"
                with h5py.File(path, "r") as root:
                    episode_len = int(root["/action"].shape[0])
                    n_frames = min(frames_per_episode, episode_len)
                    ts_choices = rng.choice(episode_len, size=n_frames, replace=False)
                    for t in ts_choices:
                        cams = []
                        for cam in camera_names:
                            im = root[f"/observations/images/{cam}"][int(t)]  # H,W,C uint8
                            im = np.transpose(im, (2, 0, 1)).astype(np.float32) / 255.0
                            cams.append(im)
                        img = torch.from_numpy(np.stack(cams, axis=0)).float().cuda().unsqueeze(0)
                        img = normalize(img)  # (1, num_cam, 3, H, W) — same call shape as policy.py

                        for cam_id in range(len(camera_names)):
                            features, _ = policy.model.backbones[0](img[:, cam_id])
                            policy.model.input_proj(features[0])  # populates captured["feat"]
                            feat = captured["feat"][0]  # (hidden_dim, h, w)
                            C, h, w = feat.shape
                            n_pts = min(points_per_frame, h * w)
                            idx = rng.choice(h * w, size=n_pts, replace=False)
                            flat = feat.reshape(C, h * w)[:, idx].transpose(0, 1).float().cpu().numpy()
                            chunks.append(flat)
    finally:
        handle.remove()
    return np.concatenate(chunks, axis=0)


def main():
    p = argparse.ArgumentParser(description="Offline PCA fit for the PCA-bottleneck FiLM mode")
    p.add_argument("--ckpt", type=str, required=True, help="path to policy .ckpt")
    p.add_argument("--stats_path", type=str, default=None, help="dataset_stats.pkl; defaults next to ckpt")
    p.add_argument("--task_name", type=str, required=True, help="task name in SIM_TASK_CONFIGS")
    p.add_argument("--output", type=str, required=True, help="output .npz path")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_episodes", type=int, default=20, help="how many demo episodes to sample from")
    p.add_argument("--frames_per_episode", type=int, default=30, help="random timesteps sampled per episode")
    p.add_argument(
        "--points_per_frame",
        type=int,
        default=16,
        help="random spatial locations sampled per frame per camera (avoids oversampling the "
        "spatially-smooth redundancy within a single feature map)",
    )
    p.add_argument(
        "--max_k",
        type=int,
        default=64,
        help="number of PCA components to fit/store; optimize_film_params.py slices W[:, :k] "
        "from this at search time, so pick something comfortably >= any k you plan to try",
    )
    # policy architecture (must match training — same flags/defaults as optimize_film_params.py)
    p.add_argument("--policy_class", type=str, default="ACT")
    p.add_argument("--chunk_size", type=int, default=100)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--dim_feedforward", type=int, default=3200)
    p.add_argument("--latent_z_dim", type=int, default=32)
    p.add_argument("--kl_weight", type=float, default=10.0)
    p.add_argument("--show_progress", action="store_true")
    args = p.parse_args()

    if args.policy_class != "ACT":
        print("FiLM PCA fit only applies to ACT (DETRVAE); use --policy_class ACT", file=sys.stderr)
        sys.exit(1)

    task_name = args.task_name
    if task_name not in SIM_TASK_CONFIGS:
        print(f"Unknown task_name: {task_name}. Options: {list(SIM_TASK_CONFIGS.keys())}", file=sys.stderr)
        sys.exit(1)
    task_cfg = SIM_TASK_CONFIGS[task_name]
    dataset_dir = task_cfg["dataset_dir"]
    camera_names = task_cfg["camera_names"]

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
        "camera_names": camera_names,
        "state_dim": task_cfg.get("state_dim", DEFAULT_STATE_DIM),
        "action_dim": task_cfg.get("action_dim", task_cfg.get("state_dim", DEFAULT_STATE_DIM)),
    }
    policy, _stats, ckpt_loaded = _load_policy_and_stats(
        Path(args.ckpt),
        Path(args.stats_path) if args.stats_path else None,
        args.policy_class,
        policy_config,
    )

    hidden_dim = int(policy.model.visual_film_gamma.numel())

    n_available = task_cfg.get("num_episodes")
    if n_available is None:
        n_available = len(list(Path(dataset_dir).glob("episode_*.hdf5")))
    n_episodes = min(args.num_episodes, n_available)
    rng = np.random.default_rng(args.seed)
    episode_ids = sorted(int(i) for i in rng.choice(n_available, size=n_episodes, replace=False))

    print(
        f"Sampling {n_episodes}/{n_available} episodes x up to {args.frames_per_episode} frames x "
        f"{args.points_per_frame} points x {len(camera_names)} camera(s) from {dataset_dir}"
    )
    t0 = time.perf_counter()
    X = _collect_activations(
        policy,
        dataset_dir,
        camera_names,
        episode_ids,
        args.frames_per_episode,
        args.points_per_frame,
        rng,
        show_progress=args.show_progress,
    )
    dt = time.perf_counter() - t0
    print(f"Collected {X.shape[0]} samples of dim {X.shape[1]} in {dt:.1f}s")
    if not np.all(np.isfinite(X)):
        print("ERROR: collected activations contain NaN/Inf", file=sys.stderr)
        sys.exit(1)

    max_k = int(args.max_k)
    cap = min(max_k, X.shape[0], hidden_dim)
    if cap < max_k:
        print(
            f"Warning: --max_k={max_k} > min(n_samples={X.shape[0]}, hidden_dim={hidden_dim}); "
            f"using max_k={cap} instead"
        )
        max_k = cap
    if max_k <= 0:
        print("Not enough samples to fit PCA (need at least 1)", file=sys.stderr)
        sys.exit(1)

    pca = PCA(n_components=max_k, svd_solver="auto", random_state=args.seed)
    pca.fit(X)
    W = pca.components_.T.astype(np.float32)  # (hidden_dim, max_k)
    mu = pca.mean_.astype(np.float32)  # (hidden_dim,)
    evr = pca.explained_variance_ratio_.astype(np.float32)

    top_n = min(8, max_k)
    print(f"Top-{top_n} EVR: {np.round(evr[:top_n], 4).tolist()}")
    print(f"Cumulative EVR @ k={top_n}: {float(np.sum(evr[:top_n])):.4f}")
    print(f"Cumulative EVR @ k={max_k}: {float(np.sum(evr)):.4f}")

    meta = {
        "ckpt": ckpt_loaded,
        "task_name": task_name,
        "dataset_dir": dataset_dir,
        "camera_names": camera_names,
        "hidden_dim": hidden_dim,
        "max_k": max_k,
        "num_episodes_available": int(n_available),
        "num_episodes_sampled": n_episodes,
        "episode_ids": episode_ids,
        "frames_per_episode": args.frames_per_episode,
        "points_per_frame": args.points_per_frame,
        "n_samples": int(X.shape[0]),
        "seed": args.seed,
        "fit_time": datetime.now().isoformat(),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        W=W,
        mu=mu,
        explained_variance_ratio=evr,
        meta=np.array(json.dumps(meta)),
    )
    print(f"Saved {out_path} (W {W.shape}, mu {mu.shape})")


if __name__ == "__main__":
    main()
