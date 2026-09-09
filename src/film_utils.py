"""PCA-/AE-bottleneck FiLM helpers: loading the policy/stats, building the AE encoder/decoder
architecture, and getting/setting theta = [gamma, beta] on whichever (method, insertion target)
is active. See optimize_film_params.py's module docstring for what theta/film_target mean, and
its --film_bottleneck_method for pca vs ae.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn

from policy import ACTPolicy
from imitate_episodes import make_policy

_FILM_TARGETS = ("visual", "memory", "hs")
_FILM_METHODS = ("pca", "ae")


def _film_bottleneck_attr(method: str, target: str, name: str) -> str:
    """Maps a (bottleneck method, FiLM insertion target) pair to its detr_vae.py
    buffer/submodule/policy-kwarg name. method="pca": "visual" keeps the original unprefixed
    film_pca_* names (backward compat); "memory"/"hs" use film_pca_memory_*/film_pca_hs_*.
    method="ae": same shape, film_ae_* / film_ae_memory_* / film_ae_hs_* (see load_film_ae()).
    These are also exactly the ACTPolicy.__call__/DETRVAE.forward() kwarg names for
    "gamma"/"beta"."""
    if method not in _FILM_METHODS:
        raise ValueError(f"Unknown film_bottleneck_method {method!r}; expected one of {_FILM_METHODS}")
    if target not in _FILM_TARGETS:
        raise ValueError(f"Unknown --film_target {target!r}; expected one of {_FILM_TARGETS}")
    base = "film_pca" if method == "pca" else "film_ae"
    prefix = base if target == "visual" else f"{base}_{target}"
    return f"{prefix}_{name}"


def _film_pca_attr(target: str, name: str) -> str:
    """Backward-compat alias for _film_bottleneck_attr("pca", target, name)."""
    return _film_bottleneck_attr("pca", target, name)


def make_film_ae_encoder(hidden_dim: int, k: int, hidden: int) -> nn.Module:
    """hidden_dim -> k. hidden<=0: single Linear (linear bottleneck, closest analog of PCA
    minus orthogonality/tied-weights). hidden>0: hidden_dim -> hidden -> k with GELU (nonlinear).
    Bias is fine here (unlike the decoder — see make_film_ae_decoder)."""
    if hidden <= 0:
        return nn.Linear(hidden_dim, k)
    return nn.Sequential(nn.Linear(hidden_dim, hidden), nn.GELU(), nn.Linear(hidden, k))


def make_film_ae_decoder(hidden_dim: int, k: int, hidden: int) -> nn.Module:
    """k -> hidden_dim, mirroring make_film_ae_encoder. Every Linear here MUST be bias=False:
    the FiLM residual formula is x + decode(delta), and needs decode(0) == 0 exactly so that
    gamma=1, beta=0 (the search's starting point) is a true identity — see
    DETRVAE.load_film_ae(), which asserts this with a zero-probe. Bias-free composes fine with
    mu (added back outside the decoder, same convention as the PCA path's `mu + W @ z`)."""
    if hidden <= 0:
        return nn.Linear(k, hidden_dim, bias=False)
    return nn.Sequential(
        nn.Linear(k, hidden, bias=False), nn.GELU(), nn.Linear(hidden, hidden_dim, bias=False)
    )


def _load_film_ae_checkpoint(path: Path, hidden_dim: int, k: int) -> tuple[nn.Module, nn.Module, torch.Tensor, dict]:
    """Loads a fit_film_ae.py .pt checkpoint, rebuilds the exact encoder/decoder architecture
    it was fit with (make_film_ae_encoder/decoder, using the checkpoint's own stored `ae_hidden`
    — NOT a CLI flag, since the architecture is fixed at fit time), and loads its weights.
    Validates hidden_dim (must match the policy) and k (must match --film_bottleneck_dim exactly
    — unlike PCA, an AE fit at k=8 cannot be sliced down to k=1; re-fit fit_film_ae.py at the k
    you want to search). Returns (encoder, decoder, mu, meta).
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ckpt_hidden_dim = int(ckpt["hidden_dim"])
    ckpt_k = int(ckpt["k"])
    ae_hidden = int(ckpt["ae_hidden"])
    if ckpt_hidden_dim != hidden_dim:
        raise ValueError(
            f"--film_ae_path hidden_dim={ckpt_hidden_dim} != model hidden_dim={hidden_dim}; "
            "was it fit against a different --hidden_dim / architecture?"
        )
    if ckpt_k != k:
        raise ValueError(
            f"--film_ae_path was fit with k={ckpt_k}, but --film_bottleneck_dim={k}. Unlike PCA, "
            "an AE's latent width is fixed at fit time (fit_film_ae.py --k) and can't be sliced "
            "down/up at search time — re-run fit_film_ae.py with --k matching --film_bottleneck_dim."
        )
    encoder = make_film_ae_encoder(hidden_dim, k, ae_hidden)
    decoder = make_film_ae_decoder(hidden_dim, k, ae_hidden)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    decoder.load_state_dict(ckpt["decoder_state_dict"])
    mu = ckpt["mu"]
    if not torch.is_tensor(mu):
        mu = torch.as_tensor(mu, dtype=torch.float32)
    return encoder, decoder, mu, ckpt.get("meta", {})


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
        # AE-bottleneck FiLM counterparts (no "_W": film_ae_encoder/decoder are added
        # dynamically by load_film_ae(), so they're simply absent from state_dict — on either
        # side — until then, and never show up as a missing/unexpected key).
        "model.film_ae_mu",
        "model.film_ae_gamma",
        "model.film_ae_beta",
        "model.film_ae_memory_mu",
        "model.film_ae_memory_gamma",
        "model.film_ae_memory_beta",
        "model.film_ae_hs_mu",
        "model.film_ae_hs_gamma",
        "model.film_ae_hs_beta",
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


def _film_theta_from_policy(policy: ACTPolicy, target: str = "visual", method: str = "pca") -> np.ndarray:
    """theta = [gamma, beta] (each k-dim) — the only free params under the PCA-/AE-bottleneck
    FiLM mode, for whichever (method, insertion target) (see detr_vae.py's
    load_film_pca()/load_film_ae()): target "visual" (default, before the Transformer
    encoder), "memory" (candidate 4, encoder-decoder boundary), or "hs" (candidate 5,
    pre-action_head). Requires load_film_pca()/load_film_ae() to have been called already for
    that (method, target) (film_{pca,ae}_{target}_k > 0)."""
    k_attr, g_attr, b_attr = (_film_bottleneck_attr(method, target, n) for n in ("k", "gamma", "beta"))
    if int(getattr(policy.model, k_attr)) <= 0:
        raise RuntimeError(
            f"policy.model has no {method}-bottleneck FiLM basis loaded for target={target!r}; "
            f"call load_film_{method}(..., target={target!r}) first"
        )
    g = getattr(policy.model, g_attr).detach().float().cpu().numpy()
    b = getattr(policy.model, b_attr).detach().float().cpu().numpy()
    return np.concatenate([g, b], axis=0)


def _apply_film_theta(policy: ACTPolicy, theta: np.ndarray, k: int, target: str = "visual", method: str = "pca"):
    g_attr, b_attr = (_film_bottleneck_attr(method, target, n) for n in ("gamma", "beta"))
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
