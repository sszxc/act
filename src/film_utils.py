"""PCA-bottleneck FiLM helpers: loading the policy/stats, and getting/setting theta = [gamma,
beta] on whichever insertion target (--film_target) is active. See optimize_film_params.py's
module docstring for what theta/film_target mean.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch

from policy import ACTPolicy
from imitate_episodes import make_policy

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
