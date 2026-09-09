#!/usr/bin/env python3
"""
Offline autoencoder fit for the AE-bottleneck FiLM mode (see detr_vae.py's
film_ae_*/film_ae_memory_*/film_ae_hs_* buffers+submodules and optimize_film_params.py
--film_bottleneck_method ae / --film_ae_path / --film_bottleneck_dim / --film_target).

Nonlinear analog of fit_film_pca.py: same activation collection (--target selects the same
three insertion points, reusing fit_film_pca.py's _collect_activations()/
_collect_activations_transformer() verbatim), but instead of fitting a PCA basis, trains a
small encoder (hidden_dim -> k) / decoder (k -> hidden_dim) pair by MSE reconstruction. The
decoder is trained bias-free (see src/film_utils.py's make_film_ae_decoder()) so that
decode(0) == 0 exactly — required for optimize_film_params.py's search to start from a true
identity at gamma=1, beta=0.

Unlike PCA (fit once at --max_k, sliced to any k <= max_k at search time), an autoencoder's
latent width is fixed at fit time: --k here must equal the --film_bottleneck_dim you intend to
search with optimize_film_params.py --film_bottleneck_method ae.

Example:
  python fit_film_ae.py --ckpt results/sim_hmf_proto5_grasp_red_box/policy_best.ckpt \\
    --task_name sim_hmf_proto5_grasp_red_box --target visual --k 1 --ae_hidden 0 \\
    --output tmp/film_ae/sim_hmf_proto5_grasp_red_box_visual_k1_h0.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from constants import SIM_TASK_CONFIGS, DEFAULT_STATE_DIM
from optimize_film_params import _load_policy_and_stats, _FILM_TARGETS
from fit_film_pca import _collect_activations, _collect_activations_transformer
from src.film_utils import make_film_ae_encoder, make_film_ae_decoder


def _train_ae(
    X: np.ndarray,
    k: int,
    hidden: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    val_frac: float,
    patience: int,
    seed: int,
    device: str = "cuda",
) -> tuple[nn.Module, nn.Module, np.ndarray, dict]:
    """Trains encoder/decoder to minimize MSE(mu + decoder(encoder(x - mu)), x), with early
    stopping on a held-out val split. Returns (encoder, decoder, mu, train_meta) — encoder/
    decoder left on `device` in eval() mode; caller moves/copies as needed."""
    hidden_dim = X.shape[1]
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * val_frac)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    mu = X.mean(axis=0)
    X_t = torch.from_numpy(X).float().to(device)
    mu_t = torch.from_numpy(mu).float().to(device)
    train_t = X_t[train_idx]
    val_t = X_t[val_idx]

    torch.manual_seed(seed)
    encoder = make_film_ae_encoder(hidden_dim, k, hidden).to(device)
    decoder = make_film_ae_decoder(hidden_dim, k, hidden).to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr, weight_decay=weight_decay)

    def _recon_mse(batch: torch.Tensor) -> torch.Tensor:
        z = encoder(batch - mu_t)
        x_hat = mu_t + decoder(z)
        return torch.mean((x_hat - batch) ** 2)

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0
    n_train = train_t.shape[0]
    history = []
    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        order = torch.randperm(n_train, device=device)
        train_loss_sum = 0.0
        for start in range(0, n_train, batch_size):
            idx = order[start : start + batch_size]
            batch = train_t[idx]
            opt.zero_grad()
            loss = _recon_mse(batch)
            loss.backward()
            opt.step()
            train_loss_sum += float(loss.item()) * batch.shape[0]
        train_mse = train_loss_sum / n_train

        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            val_mse = float(_recon_mse(val_t).item())
        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse})

        if val_mse < best_val - 1e-9:
            best_val = val_mse
            best_state = (
                {k_: v.detach().clone() for k_, v in encoder.state_dict().items()},
                {k_: v.detach().clone() for k_, v in decoder.state_dict().items()},
            )
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  early stop at epoch {epoch} (no val improvement for {patience} epochs)")
                break

    assert best_state is not None
    encoder.load_state_dict(best_state[0])
    decoder.load_state_dict(best_state[1])
    encoder.eval()
    decoder.eval()
    return encoder, decoder, mu, {"best_val_mse": best_val, "n_epochs_ran": len(history), "history_tail": history[-5:]}


def main():
    p = argparse.ArgumentParser(description="Offline autoencoder fit for the AE-bottleneck FiLM mode")
    p.add_argument("--ckpt", type=str, required=True, help="path to policy .ckpt")
    p.add_argument("--stats_path", type=str, default=None, help="dataset_stats.pkl; defaults next to ckpt")
    p.add_argument("--task_name", type=str, required=True, help="task name in SIM_TASK_CONFIGS")
    p.add_argument("--output", type=str, required=True, help="output .pt path")
    p.add_argument(
        "--target",
        type=str,
        choices=_FILM_TARGETS,
        default="visual",
        help="Which encoder-FiLM-decoder insertion point to fit an AE for: 'visual' (default, "
        "pre-Transformer-encoder), 'memory' (candidate 4: encoder-decoder boundary), or 'hs' "
        "(candidate 5: pre-action_head). Must match --film_target when this .pt is later used "
        "with optimize_film_params.py --film_bottleneck_method ae.",
    )
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
        "--k",
        type=int,
        required=True,
        help="AE bottleneck width (latent dim). Unlike fit_film_pca.py's --max_k, this is fixed "
        "at fit time and must equal --film_bottleneck_dim at search time exactly.",
    )
    p.add_argument(
        "--ae_hidden",
        type=int,
        default=0,
        help="0 (default): no hidden layer, linear bottleneck (hidden_dim -> k -> hidden_dim) — "
        "closest structural analog of PCA, minus orthogonality/tied weights. >0: one hidden "
        "layer each side (hidden_dim -> ae_hidden -> k -> ae_hidden -> hidden_dim) with GELU, "
        "i.e. a genuinely nonlinear encoder/decoder.",
    )
    p.add_argument("--ae_epochs", type=int, default=300)
    p.add_argument("--ae_lr", type=float, default=1e-3)
    p.add_argument("--ae_weight_decay", type=float, default=1e-5)
    p.add_argument("--ae_batch_size", type=int, default=256)
    p.add_argument("--ae_val_frac", type=float, default=0.1, help="fraction of samples held out for early stopping")
    p.add_argument("--ae_patience", type=int, default=30, help="early-stop patience, in epochs without val improvement")
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
        print("FiLM AE fit only applies to ACT (DETRVAE); use --policy_class ACT", file=sys.stderr)
        sys.exit(1)
    if args.k <= 0:
        print(f"--k={args.k} must be >= 1", file=sys.stderr)
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
    policy, stats, ckpt_loaded = _load_policy_and_stats(
        Path(args.ckpt),
        Path(args.stats_path) if args.stats_path else None,
        args.policy_class,
        policy_config,
    )

    hidden_dim = int(policy.model.visual_film_gamma.numel())
    state_dim = policy_config["state_dim"]

    n_available = task_cfg.get("num_episodes")
    if n_available is None:
        n_available = len(list(Path(dataset_dir).glob("episode_*.hdf5")))
    n_episodes = min(args.num_episodes, n_available)
    rng = np.random.default_rng(args.seed)
    episode_ids = sorted(int(i) for i in rng.choice(n_available, size=n_episodes, replace=False))

    print(
        f"Sampling {n_episodes}/{n_available} episodes x up to {args.frames_per_episode} frames x "
        f"{args.points_per_frame} points x {len(camera_names)} camera(s) from {dataset_dir} "
        f"(target={args.target})"
    )
    t0 = time.perf_counter()
    if args.target == "visual":
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
    else:
        X = _collect_activations_transformer(
            policy,
            dataset_dir,
            camera_names,
            state_dim,
            np.asarray(stats["qpos_mean"], dtype=np.float32),
            np.asarray(stats["qpos_std"], dtype=np.float32),
            episode_ids,
            args.frames_per_episode,
            args.points_per_frame,
            rng,
            args.target,
            show_progress=args.show_progress,
        )
    dt = time.perf_counter() - t0
    print(f"Collected {X.shape[0]} samples of dim {X.shape[1]} in {dt:.1f}s")
    if not np.all(np.isfinite(X)):
        print("ERROR: collected activations contain NaN/Inf", file=sys.stderr)
        sys.exit(1)

    k = int(args.k)
    if k > X.shape[0]:
        print(f"--k={k} > n_samples={X.shape[0]}; collect more (--num_episodes/--frames_per_episode/--points_per_frame)", file=sys.stderr)
        sys.exit(1)

    print(f"Training AE: hidden_dim={hidden_dim} -> k={k} (ae_hidden={args.ae_hidden}), {args.ae_epochs} epochs max")
    t0 = time.perf_counter()
    encoder, decoder, mu, train_meta = _train_ae(
        X,
        k=k,
        hidden=int(args.ae_hidden),
        epochs=int(args.ae_epochs),
        lr=float(args.ae_lr),
        weight_decay=float(args.ae_weight_decay),
        batch_size=int(args.ae_batch_size),
        val_frac=float(args.ae_val_frac),
        patience=int(args.ae_patience),
        seed=args.seed,
    )
    dt = time.perf_counter() - t0
    print(f"Trained in {dt:.1f}s: best val MSE={train_meta['best_val_mse']:.6f} ({train_meta['n_epochs_ran']} epochs ran)")

    # Sanity check mirroring detr_vae.py's load_film_ae() zero-probe: decode(0) must be ~0.
    decoder_device = next(decoder.parameters()).device
    with torch.no_grad():
        zero_out = decoder(torch.zeros(1, k, device=decoder_device))
    max_abs = float(zero_out.abs().max().item())
    if max_abs > 1e-5:
        print(f"ERROR: decoder(0) max abs = {max_abs:.2e}, expected ~0 (bias-free decoder invariant broken)", file=sys.stderr)
        sys.exit(1)
    print(f"OK: decoder(0) max abs = {max_abs:.2e} (identity-at-baseline invariant holds)")

    meta = {
        "ckpt": ckpt_loaded,
        "target": args.target,
        "task_name": task_name,
        "dataset_dir": dataset_dir,
        "camera_names": camera_names,
        "hidden_dim": hidden_dim,
        "k": k,
        "ae_hidden": int(args.ae_hidden),
        "ae_epochs": int(args.ae_epochs),
        "ae_lr": float(args.ae_lr),
        "ae_weight_decay": float(args.ae_weight_decay),
        "ae_batch_size": int(args.ae_batch_size),
        "ae_val_frac": float(args.ae_val_frac),
        "ae_patience": int(args.ae_patience),
        "best_val_mse": train_meta["best_val_mse"],
        "n_epochs_ran": train_meta["n_epochs_ran"],
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
    torch.save(
        {
            "hidden_dim": hidden_dim,
            "k": k,
            "ae_hidden": int(args.ae_hidden),
            "encoder_state_dict": {k_: v.cpu() for k_, v in encoder.state_dict().items()},
            "decoder_state_dict": {k_: v.cpu() for k_, v in decoder.state_dict().items()},
            "mu": torch.from_numpy(mu).float(),
            "meta": meta,
        },
        out_path,
    )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
