#!/usr/bin/env python3
"""
将指定文件夹下的图片按文件名排序后合成 MP4，复用 visualize_episodes._save_video。

用法（建议在工程根 act/ 下执行，或任意目录用绝对路径调用本脚本）:
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

# 可选：最近邻缩放到此尺寸 (width, height)
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
    parser = argparse.ArgumentParser(description="文件夹内图片 → MP4（使用 visualize_episodes._save_video）")
    parser.add_argument("input_dir", help="包含图片的目录")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="输出视频路径，默认 <input_dir>/stitched.mp4",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="帧率")
    parser.add_argument(
        "--resize-1280x480",
        action="store_true",
        help="每帧用最近邻插值缩放到 1280×480（宽×高）",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="打印详情")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        print(f"错误: 不是目录: {input_dir}", file=sys.stderr)
        sys.exit(1)

    paths = _collect_image_paths(input_dir)
    if not paths:
        print(f"错误: 目录内没有支持的图片 ({', '.join(sorted(_IMAGE_EXTS))}): {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output
    if not output_path:
        output_path = os.path.join(input_dir, "stitched.mp4")
    output_path = os.path.abspath(output_path)

    resize_wh = _RESIZE_1280x480_WH if args.resize_1280x480 else None
    if args.verbose:
        print(f"读取 {len(paths)} 张: 首 {paths[0]} … 尾 {paths[-1]}")
        if resize_wh is not None:
            print(f"缩放: 最近邻 → {resize_wh[0]}×{resize_wh[1]} (宽×高)")

    frames = _load_frames_rgb_uint8(paths, resize_wh=resize_wh)
    _save_video(frames, output_path, fps=args.fps, verbose=args.verbose)


if __name__ == "__main__":
    main()
