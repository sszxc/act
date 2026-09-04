#!/usr/bin/env python3
"""
Turn one or more fit_film_pca.py .npz outputs into a standalone HTML report: a cumulative
explained-variance-ratio (EVR) chart comparing all fits, plus a per-fit tab with the individual
per-component EVR bars, fit metadata, and a retained/lost-variance table at common k checkpoints.

Answers "how much information is lost by truncating to k dims / what does each dim carry":
cumulative[k-1] is the fraction of variance retained by the first k PCA dims (so 1 - cumulative[k-1]
is what --film_bottleneck_dim k would discard); evr[i] is dim i's own share.

Just fills in film_pca_viz_template.html's embedded #pca-data JSON blob — open the output file
directly in a browser, no server needed. Re-run any time a PCA fit changes or a new one is added.

Example:
  python visualize_film_pca.py tmp/film_pca/*.npz
  # -> tmp/film_pca/film_pca_viz.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build_payload(npz_paths: list[Path]) -> dict:
    files = []
    for path in npz_paths:
        d = np.load(path, allow_pickle=False)
        evr = d["explained_variance_ratio"].astype(np.float64)
        cumulative = np.cumsum(evr)
        meta = json.loads(str(d["meta"]))
        label = f"{meta.get('task_name', path.stem)} ({meta.get('target', '?')})"
        files.append(
            {
                "label": label,
                "source_file": path.name,
                "evr": evr.round(6).tolist(),
                "cumulative": np.minimum(cumulative, 1.0).round(6).tolist(),
                "meta": meta,
            }
        )
    return {"files": files}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("npz_paths", type=str, nargs="+", help="fit_film_pca.py output .npz file(s)")
    p.add_argument("--output", type=str, default=None, help="Output .html path (defaults next to the first input)")
    p.add_argument(
        "--template",
        type=str,
        default=None,
        help="Override template path (defaults to film_pca_viz_template.html next to this script)",
    )
    args = p.parse_args()

    npz_paths = sorted(Path(s).resolve() for s in args.npz_paths)
    missing = [str(pth) for pth in npz_paths if not pth.is_file()]
    if missing:
        raise SystemExit(f"Not found: {missing}")

    template_path = (
        Path(args.template).resolve() if args.template else Path(__file__).resolve().parent / "film_pca_viz_template.html"
    )
    output_path = Path(args.output).resolve() if args.output else npz_paths[0].parent / "film_pca_viz.html"

    payload = build_payload(npz_paths)
    # Escape "</" so a stray "</script"-like substring in e.g. a ckpt path can't break out of
    # the <script> block the JSON is embedded in.
    payload_json = json.dumps(payload).replace("</", "<\\/")

    template = template_path.read_text(encoding="utf-8")
    marker = '<script type="application/json" id="pca-data">null</script>'
    if marker not in template:
        raise SystemExit(f"Template {template_path} is missing the expected `{marker}` placeholder")
    html = template.replace(marker, f'<script type="application/json" id="pca-data">{payload_json}</script>')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Wrote {output_path} ({len(payload['files'])} fit(s))")
    print("Open it directly in a browser, or ask Claude to publish it as an Artifact.")


if __name__ == "__main__":
    main()
