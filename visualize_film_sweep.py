#!/usr/bin/env python3
"""
Turn a film_sweep_trajectories.npz (from `optimize_film_params.py --method sweep`) into a
standalone, self-contained HTML page: a Plotly 3D end-effector path explorer, one tab per swept
FiLM dimension, linked to a height-over-time and a reward-vs-value chart. If the sweep also
crossed an object x/y grid (--object_sweep_x_values/--object_sweep_y_values), two sliders beside
and below the 3D plot pick which object position's dim/value sweep is shown.

Just fills in film_sweep_viz_template.html's embedded `#sweep-data` JSON blob — open the output
file directly in a browser, no server needed. Run again any time the sweep dir changes.

Example:
  python visualize_film_sweep.py --sweep_dir results/sim_hmf_proto5_pick_allgood/icl_20260903-154549_sweep
  # -> .../icl_20260903-154549_sweep/film_sweep_viz.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _subsample_indices(n: int, max_points: int) -> np.ndarray:
    """Evenly spaced indices into [0, n), always including the first and last point."""
    if n <= max_points:
        return np.arange(n)
    return np.unique(np.round(np.linspace(0, n - 1, max_points)).astype(int))


def build_payload(npz_path: Path, meta_path: Path | None, max_points: int) -> dict:
    d = np.load(npz_path, allow_pickle=False)
    dim_name = d["dim_name"]
    value = d["value"]
    episode_return = d["episode_return"]
    mocap_pos = d["mocap_pos"]  # (n_points, T, 3)

    meta = {
        "dim_names": [str(x) for x in d["dim_names"].tolist()],
        "sweep_values": [float(x) for x in d["sweep_values"].tolist()],
        "theta_base": [float(x) for x in d["theta_base"].tolist()],
        "source_file": npz_path.name,
    }
    if meta_path is not None and meta_path.is_file():
        run_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for key in ("task_name", "film_target", "ckpt", "fixed_object_pose"):
            if key in run_meta:
                meta[key] = run_meta[key]
    meta.setdefault("fixed_object_pose", [0.0, 0.0, 0.0])

    # Object x/y grid (optimize_film_params.py --object_sweep_x_values/--object_sweep_y_values),
    # crossed with the dim/value sweep. Absent in npz files from before that feature -> every
    # row falls back to the single fixed_object_pose x/y (same as when the grid has 1 point).
    has_object_grid = "object_x" in d.files and "object_y" in d.files
    object_x = d["object_x"] if has_object_grid else None
    object_y = d["object_y"] if has_object_grid else None
    if "object_x_values" in d.files:
        meta["object_x_values"] = [round(float(x), 4) for x in d["object_x_values"].tolist()]
    if "object_y_values" in d.files:
        meta["object_y_values"] = [round(float(x), 4) for x in d["object_y_values"].tolist()]

    idx = _subsample_indices(mocap_pos.shape[1], max_points)
    rows = [
        {
            "dim_name": str(dim_name[i]),
            "value": float(value[i]),
            "episode_return": float(episode_return[i]),
            "mocap_pos": mocap_pos[i, idx, :].round(4).tolist(),
            "object_x": round(float(object_x[i]), 4) if has_object_grid else meta["fixed_object_pose"][0],
            "object_y": round(float(object_y[i]), 4) if has_object_grid else meta["fixed_object_pose"][1],
        }
        for i in range(mocap_pos.shape[0])
    ]

    return {"meta": meta, "rows": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep_dir", type=str, required=True, help="Output dir from --method sweep (contains film_sweep_trajectories.npz)")
    p.add_argument("--npz_path", type=str, default=None, help="Override: explicit path to the .npz (defaults to <sweep_dir>/film_sweep_trajectories.npz)")
    p.add_argument("--output", type=str, default=None, help="Output .html path (defaults to <sweep_dir>/film_sweep_viz.html)")
    p.add_argument("--max_points", type=int, default=200, help="Subsample each trajectory to at most this many points (keeps the file light)")
    p.add_argument(
        "--template",
        type=str,
        default=None,
        help="Override template path (defaults to film_sweep_viz_template.html next to this script)",
    )
    args = p.parse_args()

    sweep_dir = Path(args.sweep_dir).resolve()
    npz_path = Path(args.npz_path).resolve() if args.npz_path else sweep_dir / "film_sweep_trajectories.npz"
    if not npz_path.is_file():
        raise SystemExit(f"Not found: {npz_path} (did the sweep finish? see sweep_history.jsonl in {sweep_dir})")
    meta_path = sweep_dir / "run_meta.json"
    template_path = Path(args.template).resolve() if args.template else Path(__file__).resolve().parent / "film_sweep_viz_template.html"
    output_path = Path(args.output).resolve() if args.output else sweep_dir / "film_sweep_viz.html"

    payload = build_payload(npz_path, meta_path, args.max_points)
    # Escape "</" so a stray "</script"-like substring in e.g. a ckpt path can't break out of
    # the <script> block the JSON is embedded in.
    payload_json = json.dumps(payload).replace("</", "<\\/")

    template = template_path.read_text(encoding="utf-8")
    marker = '<script type="application/json" id="sweep-data">null</script>'
    if marker not in template:
        raise SystemExit(f"Template {template_path} is missing the expected `{marker}` placeholder")
    html = template.replace(marker, f'<script type="application/json" id="sweep-data">{payload_json}</script>')

    output_path.write_text(html, encoding="utf-8")
    print(f"Wrote {output_path} ({len(payload['rows'])} trajectories, {args.max_points} pts/traj max)")
    print("Open it directly in a browser, or ask Claude to publish it as an Artifact.")


if __name__ == "__main__":
    main()
