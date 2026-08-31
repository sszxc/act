#!/usr/bin/env python3
"""
Aggregate reward-vs-iteration curves from several optimize_film_params.py runs (repeats of the
same search config) into one statistical plot: mean curve with a shaded std/var/sem band.

Each positional argument is an --output_dir produced by optimize_film_params.py, i.e. a directory
containing one of ars_curves.npz / cma_curves.npz / llm_curves.npz (each has a "best_so_far" array
plus a method-specific per-iteration array: iter_max for ars, gen_max for cma, iter_reward for llm)
and a run_meta.json. Runs may have different lengths (e.g. one was interrupted early); shorter runs
are padded to the longest length before averaging.

Examples:
  # explicit list of run dirs
  python plot_reward_stats.py results/sim_hmf_proto5_pick_allgood/icl_20260830-*_cma

  # glob pattern (quote it so the script expands it, not the shell) + only the latest N by name
  python plot_reward_stats.py "results/sim_hmf_proto5_pick_allgood/icl_*qwen3-vl*" --latest 5

  # custom output path / shading
  python plot_reward_stats.py results/.../icl_*_cma --latest 5 --band var -o tmp/cma_stats.png
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import warnings
from pathlib import Path

import numpy as np


def _resolve_dirs(patterns: list[str], latest: int | None) -> list[Path]:
    """Expand each arg: kept as-is if it's an existing directory, else glob-expanded. Concatenated
    in order, deduplicated, then sorted by name (run dirs are timestamp-prefixed, so name order is
    chronological) and trimmed to the last `latest` if given."""
    dirs: list[Path] = []
    for pat in patterns:
        if os.path.isdir(pat):
            dirs.append(Path(pat))
            continue
        matches = sorted(p for p in glob.glob(pat) if os.path.isdir(p))
        if not matches:
            print(f"[warn] no directory matched: {pat}")
        dirs.extend(Path(m) for m in matches)

    seen = set()
    uniq = []
    for d in dirs:
        rd = d.resolve()
        if rd not in seen:
            seen.add(rd)
            uniq.append(d)
    uniq.sort(key=lambda p: p.name)
    if latest is not None:
        uniq = uniq[-latest:]
    return uniq


def _load_run(run_dir: Path):
    """Return (best_so_far, raw_values, raw_key, method) for one run dir, or None if it has no
    curves file yet (e.g. an interrupted run that crashed before its first checkpoint)."""
    npz_paths = sorted(run_dir.glob("*_curves.npz"))
    if not npz_paths:
        print(f"[warn] skipping {run_dir}: no *_curves.npz found")
        return None
    if len(npz_paths) > 1:
        print(f"[warn] {run_dir}: multiple *_curves.npz found, using {npz_paths[0].name}")
    z = np.load(npz_paths[0])
    if "best_so_far" not in z.files:
        print(f"[warn] skipping {run_dir}: {npz_paths[0].name} has no 'best_so_far'")
        return None
    best = np.asarray(z["best_so_far"], dtype=np.float64)
    raw_keys = [k for k in z.files if k != "best_so_far"]
    raw_key = raw_keys[0] if raw_keys else None
    raw = np.asarray(z[raw_key], dtype=np.float64) if raw_key else None

    method = npz_paths[0].name.split("_curves.npz")[0]  # ars / cma / llm
    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get("method") == "llm" and meta.get("llm_model"):
                method = f"llm:{meta['llm_model']}"
        except Exception:
            pass
    return best, raw, raw_key, method


def _pad(curves: list[np.ndarray], mode: str) -> np.ndarray:
    """Stack ragged 1D curves into (n_runs, max_len), padding short ones.
    mode="ffill": repeat each curve's last value (right choice for a cumulative best-so-far curve:
    a run that stopped early would plausibly have kept that best). mode="nan": pad with NaN, later
    aggregated with nanmean/nanstd (right choice for a per-iteration raw value, which shouldn't be
    fabricated)."""
    max_len = max(len(c) for c in curves)
    out = np.full((len(curves), max_len), np.nan, dtype=np.float64)
    for i, c in enumerate(curves):
        out[i, : len(c)] = c
        if mode == "ffill" and len(c) < max_len and len(c) > 0:
            out[i, len(c):] = c[-1]
    return out


def _band(stack: np.ndarray, kind: str):
    """stack: (n_runs, max_len), possibly with NaN padding. Returns (x, mean, lo, hi, n_per_x)."""
    n_per_x = np.sum(~np.isnan(stack), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN slice, if any
        mean = np.nanmean(stack, axis=0)
        std = np.nanstd(stack, axis=0)
    if kind == "std":
        spread = std
    elif kind == "var":
        spread = std**2
    elif kind == "sem":
        spread = std / np.sqrt(np.maximum(n_per_x, 1))
    elif kind == "minmax":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            lo = np.nanmin(stack, axis=0)
            hi = np.nanmax(stack, axis=0)
        return np.arange(stack.shape[1]), mean, lo, hi, n_per_x
    else:
        raise ValueError(kind)
    return np.arange(stack.shape[1]), mean, mean - spread, mean + spread, n_per_x


def _plot_panel(ax, curves: list[np.ndarray], pad_mode: str, band: str, label: str, color, show_individual: bool):
    stack = _pad(curves, pad_mode)
    x, mean, lo, hi, n_per_x = _band(stack, band)
    if show_individual:
        for c in curves:
            ax.plot(np.arange(len(c)), c, color=color, alpha=0.15, linewidth=1, zorder=1)
    ax.plot(x, mean, color=color, label=f"{label} (n={len(curves)})", linewidth=2, zorder=3)
    ax.fill_between(x, lo, hi, color=color, alpha=0.2, linewidth=0, zorder=2)
    return int(n_per_x.min()), int(n_per_x.max())


def main():
    p = argparse.ArgumentParser(
        description="Plot mean +/- std/var/sem across repeated optimize_film_params.py runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("exp_dirs", nargs="+", help="run dirs (each an optimize_film_params.py --output_dir), or glob patterns")
    p.add_argument("--latest", type=int, default=None, help="after resolving/sorting by name, keep only the last N runs")
    p.add_argument("--band", choices=["std", "var", "sem", "minmax"], default="std", help="shaded-region kind (default: std)")
    p.add_argument("--metric", choices=["best", "raw", "both"], default="both", help="which curve(s) to plot (default: both)")
    p.add_argument("--label", type=str, default=None, help="legend label override (default: inferred method, or 'mixed' if runs differ)")
    p.add_argument("--title", type=str, default=None, help="plot title override")
    p.add_argument("-o", "--output", type=str, default=None, help="output .png path (default: <common parent>/reward_stats_<label>.png)")
    p.add_argument("--no_individual", action="store_true", help="don't overlay the faint per-run curves")
    p.add_argument("--dpi", type=int, default=120)
    args = p.parse_args()

    run_dirs = _resolve_dirs(args.exp_dirs, args.latest)
    if not run_dirs:
        raise SystemExit("no run directories resolved from the given paths/patterns")

    bests, raws, raw_keys, methods = [], [], [], []
    for d in run_dirs:
        loaded = _load_run(d)
        if loaded is None:
            continue
        best, raw, raw_key, method = loaded
        bests.append(best)
        methods.append(method)
        if raw is not None:
            raws.append(raw)
            raw_keys.append(raw_key)
    if not bests:
        raise SystemExit("no usable runs (none had a readable *_curves.npz)")

    print(f"Loaded {len(bests)} run(s):")
    for d, m, b in zip(run_dirs, methods, bests):
        print(f"  {d}  [{m}]  len={len(b)}  final best={b[-1]:.4f}")

    methods_uniq = sorted(set(methods))
    if len(methods_uniq) > 1:
        print(f"[warn] mixing runs from different methods/models: {methods_uniq}")
    label = args.label or (methods_uniq[0] if len(methods_uniq) == 1 else "mixed")

    raw_keys_uniq = sorted(set(raw_keys))
    raw_label = raw_keys_uniq[0] if len(raw_keys_uniq) == 1 else "per-iter value"

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_panels = 2 if args.metric == "both" else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 4 * n_panels), squeeze=False)
    axes = axes[:, 0]
    color = "tab:blue"
    ai = 0
    if args.metric in ("best", "both"):
        lo_n, hi_n = _plot_panel(axes[ai], bests, "ffill", args.band, f"{label}: best_so_far", color, not args.no_individual)
        axes[ai].set_ylabel("episode_return (best_so_far)")
        axes[ai].set_title("cumulative best" + ("" if lo_n == hi_n else f"  (n per x: {lo_n}-{hi_n})"))
        ai += 1
    if args.metric in ("raw", "both") and raws:
        lo_n, hi_n = _plot_panel(axes[ai], raws, "nan", args.band, f"{label}: {raw_label}", "tab:orange", not args.no_individual)
        axes[ai].set_ylabel(f"episode_return ({raw_label})")
        axes[ai].set_title("per iteration/generation" + ("" if lo_n == hi_n else f"  (n per x: {lo_n}-{hi_n})"))
        ai += 1
    elif args.metric in ("raw", "both"):
        print("[warn] no per-iteration raw arrays found; skipping that panel")

    for ax in axes[:ai]:
        ax.set_xlabel("iteration / generation")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(args.title or f"{label} — {len(bests)} run(s), band={args.band}")
    fig.tight_layout()

    if args.output:
        out_path = Path(args.output)
    else:
        parents = {d.resolve().parent for d in run_dirs}
        common_parent = parents.pop() if len(parents) == 1 else Path(os.path.commonpath([str(d.resolve()) for d in run_dirs]))
        safe_label = label.replace("/", "_").replace(":", "_")
        out_path = common_parent / f"reward_stats_{safe_label}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
