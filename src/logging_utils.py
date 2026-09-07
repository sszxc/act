"""Duration formatting + reward-curve plotting/checkpoint helpers shared by every run_* optimizer
in optimizers.py and llm_optimizer.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np


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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    if len(y1) == 0:
        # Still write a PNG on failed / 0-round runs so every experiment dir has a
        # reward_curve.png (empty axes + annotation), not a missing file.
        ax.text(
            0.5,
            0.5,
            "no completed rounds to plot",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            alpha=0.7,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        ax.plot(np.arange(len(y1)), y1, label=y1_label)
        if y2 is not None:
            ax.plot(np.arange(len(y2)), y2, alpha=0.6, label=y2_label or "iter/gen max")
        ax.legend()
    ax.set_xlabel("iteration / generation")
    ax.set_ylabel(ylabel)
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
