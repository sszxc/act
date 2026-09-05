"""Strip teleop pause segments out of already-converted episode_*.hdf5 files (the ACT-format
output of convert_teleop_dataset.py: observations/{qpos,qvel,images/<cam>} + action).

A pause = a run of >= --min_pause_s seconds where ||qvel|| < --vel_thresh (norm over all joints).
Matching frames (qpos/qvel/action/images, all cameras) are deleted outright and the remaining
frames are re-concatenated; action is recomputed as action[t] = qpos[t+1] (qpos held at the end)
since deletion shifts what "next frame" means. episode_i.hdf5 numbering is preserved 1:1 with the
input (no episodes are dropped, only shortened) - only sample_count and duration change.

Usage:
    python clean_pauses.py --data_dir data/real_pick_yellow_bottle/good_41 \\
        --out_dir data/real_pick_yellow_bottle/good_41_clean
"""
import argparse
import glob
import os

import h5py
import numpy as np

FPS = 30.0


def find_pause_mask(qvel, vel_thresh, min_pause_frames):
    """Returns a boolean keep-mask (True = keep) marking frames inside qualifying pause runs False."""
    speed = np.linalg.norm(qvel, axis=1)
    below = speed < vel_thresh
    keep = np.ones(len(speed), dtype=bool)
    i = 0
    n = len(below)
    runs = []
    while i < n:
        if below[i]:
            j = i
            while j < n and below[j]:
                j += 1
            if j - i >= min_pause_frames:
                keep[i:j] = False
                runs.append((i, j))
            i = j
        else:
            i += 1
    return keep, runs


def clean_episode(in_path, out_path, vel_thresh, min_pause_frames, log):
    with h5py.File(in_path) as f:
        qpos = f["observations/qpos"][:]
        qvel = f["observations/qvel"][:]
        cams = {cam: f[f"observations/images/{cam}"][:] for cam in f["observations/images"]}

    keep, runs = find_pause_mask(qvel, vel_thresh, min_pause_frames)
    n_before, n_after = len(qpos), int(keep.sum())

    qpos = qpos[keep]
    qvel = qvel[keep]
    cams = {cam: imgs[keep] for cam, imgs in cams.items()}
    action = np.concatenate([qpos[1:], qpos[-1:]], axis=0)  # action[t] = qpos[t+1]; hold pose at the end

    with h5py.File(out_path, "w", rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs["sim"] = False
        with h5py.File(in_path) as fin:
            root.attrs["source_episode_id"] = fin.attrs.get("source_episode_id", "")
        obs = root.create_group("observations")
        image_grp = obs.create_group("images")
        for cam, imgs in cams.items():
            h, w = imgs.shape[1:3]
            dset = image_grp.create_dataset(cam, (n_after, h, w, 3), dtype="uint8", chunks=(1, h, w, 3))
            dset.attrs["CLASS"] = np.bytes_("IMAGE")
            dset[...] = imgs
        obs.create_dataset("qpos", data=qpos)
        obs.create_dataset("qvel", data=qvel)
        root.create_dataset("action", data=action)

    if runs:
        run_str = ", ".join(f"{(j - i) / FPS:.2f}s" for i, j in runs)
        log(f"{os.path.basename(in_path)}: removed {len(runs)} pause(s) [{run_str}] "
            f"-> {n_before} -> {n_after} frames")
    else:
        log(f"{os.path.basename(in_path)}: no pauses, {n_before} frames unchanged")
    return n_before - n_after


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True, help="dir of converted episode_*.hdf5 files")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--vel_thresh", type=float, default=0.15, help="||qvel|| below this counts as stationary")
    ap.add_argument("--min_pause_s", type=float, default=0.5, help="minimum stationary run duration to strip")
    args = ap.parse_args()

    min_pause_frames = int(round(args.min_pause_s * FPS))
    in_files = sorted(glob.glob(os.path.join(args.data_dir, "episode_*.hdf5")),
                       key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))
    print(f"found {len(in_files)} episodes in {args.data_dir}\n")

    os.makedirs(args.out_dir, exist_ok=True)
    total_removed = 0
    for in_path in in_files:
        out_path = os.path.join(args.out_dir, os.path.basename(in_path))
        if os.path.exists(out_path):
            try:
                with h5py.File(out_path, "r") as f:
                    f["observations/qpos"]  # touch to force validity check
                print(f"{os.path.basename(in_path)}: already done, skipping")
                continue
            except OSError:
                print(f"{os.path.basename(in_path)}: existing output is corrupt, redoing")
        total_removed += clean_episode(in_path, out_path, args.vel_thresh, min_pause_frames, log=print)

    print(f"\nwrote {len(in_files)} episodes to {args.out_dir} "
          f"(removed {total_removed} frames total, {total_removed / FPS:.1f}s)")


if __name__ == "__main__":
    main()
