"""Small parsing/formatting helpers shared by optimize_film_params.py's CLI and rollout code."""
from __future__ import annotations

import json

import numpy as np
import torch


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
