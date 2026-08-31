#!/usr/bin/env python3
"""
Hard-coded comparison plot: overlay several groups of repeated optimize_film_params.py runs
(e.g. target x optimizer combos) on one figure, each as a mean +/- band curve.

No CLI — edit GROUPS/BASE_DIR/BAND/METRIC below and rerun. Reuses the loading/aggregation
helpers from plot_reward_stats.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from plot_reward_stats import _band, _load_run, _pad

BASE_DIR = Path("results/sim_hmf_proto5_pick_allgood")

GROUPS = {
    "target_visual - vlm": [
        "icl_20260830-194953_qwen3-vl-32b-thinking",
        "icl_20260830-210012_qwen3-vl-32b-thinking",
        "icl_20260830-215602_qwen3-vl-32b-thinking",
        "icl_20260830-225932_qwen3-vl-32b-thinking",
    ],
    "target_visual - cma": [
        "icl_20260830-200725_cma",
        "icl_20260830-204850_cma",
        "icl_20260830-220423_cma",
        "icl_20260830-223911_cma",
    ],
    "target_hs - vlm": [
        "icl_20260830-234348_qwen3-vl-32b-thinking",
        "icl_20260831-004350_qwen3-vl-32b-thinking",
        "icl_20260831-013857_qwen3-vl-32b-thinking",
        "icl_20260831-022940_qwen3-vl-32b-thinking",
        "icl_20260831-032052_qwen3-vl-32b-thinking",
    ],
    "target_hs - cma": [
        "icl_20260830-234238_cma",
        "icl_20260831-002116_cma",
        "icl_20260831-010426_cma",
        "icl_20260831-014319_cma",
        "icl_20260831-022115_cma",
    ],
}

COLORS = {
    "target_visual - vlm": "tab:blue",
    "target_visual - cma": "tab:orange",
    "target_hs - vlm": "tab:green",
    "target_hs - cma": "tab:red",
}

BAND = "std"  # std | var | sem | minmax
METRIC = "both"  # best | raw | both
SHOW_INDIVIDUAL = False  # overlay faint per-run curves on top of each group's mean+band
OUTPUT = BASE_DIR / "reward_stats_compare_target_optimizer.png"


def _plot_group(ax, curves: list[np.ndarray], pad_mode: str, name: str, color):
    stack = _pad(curves, pad_mode)
    x, mean, lo, hi, _n_per_x = _band(stack, BAND)
    if SHOW_INDIVIDUAL:
        for c in curves:
            ax.plot(np.arange(len(c)), c, color=color, alpha=0.12, linewidth=1, zorder=1)
    ax.plot(x, mean, color=color, label=f"{name} (n={len(curves)})", linewidth=2, zorder=3)
    ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0, zorder=2)


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_panels = 2 if METRIC == "both" else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(9, 4.5 * n_panels), squeeze=False)
    axes = axes[:, 0]

    for name, run_names in GROUPS.items():
        color = COLORS.get(name)
        bests, raws = [], []
        for run_name in run_names:
            loaded = _load_run(BASE_DIR / run_name)
            if loaded is None:
                continue
            best, raw, _raw_key, _method = loaded
            bests.append(best)
            if raw is not None:
                raws.append(raw)
        print(f"{name}: {len(bests)}/{len(run_names)} runs loaded")

        ai = 0
        if METRIC in ("best", "both") and bests:
            _plot_group(axes[ai], bests, "ffill", name, color)
            ai += 1
        if METRIC in ("raw", "both") and raws:
            _plot_group(axes[ai], raws, "nan", name, color)

    ai = 0
    if METRIC in ("best", "both"):
        axes[ai].set_ylabel("episode_return (best_so_far)")
        axes[ai].set_title("cumulative best")
        ai += 1
    if METRIC in ("raw", "both"):
        axes[ai].set_ylabel("episode_return (per-iter value)")
        axes[ai].set_title("per iteration/generation")

    for ax in axes:
        ax.set_xlabel("iteration / generation")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(f"target x optimizer comparison (band={BAND})")
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=130)
    plt.close(fig)
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
