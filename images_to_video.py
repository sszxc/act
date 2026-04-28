#!/usr/bin/env python3
"""
Stitch images in a folder (sorted by filename) into MP4 using visualize_episodes._save_video.

Run from repo root act/ or call with absolute paths:
  python images_to_video.py tmp/film_features_after_film -o tmp/film_mean.mp4 --fps 10
  python images_to_video.py tmp/film_features_after_film -o out.mp4 --resize-1280x480
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from visualize_episodes import _save_video  # noqa: E402

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# Optional nearest-neighbor resize to (width, height)
_RESIZE_1280x480_WH = (1280, 480)


def _collect_image_paths(input_dir: str):
    names = []
    for n in os.listdir(input_dir):
        if os.path.splitext(n)[1].lower() in _IMAGE_EXTS and os.path.isfile(
            os.path.join(input_dir, n)
        ):
            names.append(n)
    names.sort()
    return [os.path.join(input_dir, n) for n in names]


def _load_frames_rgb_uint8(paths, resize_wh=None):
    frames = []
    for p in paths:
        with Image.open(p) as im:
            rgb = im.convert("RGB")
            if resize_wh is not None:
                rgb = rgb.resize(resize_wh, resample=Image.NEAREST)
            frames.append(np.asarray(rgb, dtype=np.uint8))
    return frames


def main():
    parser = argparse.ArgumentParser(description="Folder of images → MP4 (via visualize_episodes._save_video)")
    parser.add_argument("input_dir", help="Directory containing images")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output video path; default <input_dir>/stitched.mp4",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="FPS")
    parser.add_argument(
        "--resize-1280x480",
        action="store_true",
        help="Nearest-neighbor resize each frame to 1280×480 (width×height)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        print(f"Error: not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    paths = _collect_image_paths(input_dir)
    if not paths:
        print(f"Error: no supported images ({', '.join(sorted(_IMAGE_EXTS))}) in: {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output
    if not output_path:
        output_path = os.path.join(input_dir, "stitched.mp4")
    output_path = os.path.abspath(output_path)

    resize_wh = _RESIZE_1280x480_WH if args.resize_1280x480 else None
    if args.verbose:
        print(f"Loaded {len(paths)} images: first {paths[0]} … last {paths[-1]}")
        if resize_wh is not None:
            print(f"Resize (nearest-neighbor): {resize_wh[0]}×{resize_wh[1]} (W×H)")

    frames = _load_frames_rgb_uint8(paths, resize_wh=resize_wh)
    _save_video(frames, output_path, fps=args.fps, verbose=args.verbose)


if __name__ == "__main__":
    main()
