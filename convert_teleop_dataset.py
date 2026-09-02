"""Convert raw teleop episode dirs (trajectory.h5 + videos/) into ACT-format episode_*.hdf5.

Each source episode dir (e.g. data/20260901_good_data_90hz/20260901_HHMMSS_xxxx) must have:
  manifest.json   (schema_version, task_label, time_sync.video_origin_ns/video_end_ns, video_paths, sample_counts)
  trajectory.h5   trajectories/combined/{position,velocity,source_time_ns}   (schema_version 1.3+ only)
  videos/<cam>.mp4

Only episodes with schema_version == "1.3" and task_label == "success" are converted; everything else
is skipped and logged (schema 1.1 episodes have no `combined` stream and are dropped rather than merged
by hand, since it's a single episode in the current batches).

Alignment: manifest sample_counts are not reliable (checked against actual decoded frame count instead).
The `combined` joint stream is already a uniform 90Hz nearest-sample grid derived from the arm+hand
streams; video is natively 30fps and doesn't carry true per-frame capture timestamps (mp4 is written at
a constant 1/30s spacing from video_origin_ns). Joint recording and video also don't start/stop together
(joint recording usually starts a bit before video, and sometimes stops several seconds before video
does). So: clip to the intersection of [video_origin_ns, video_end_ns] and the combined stream's own time
range, then resample everything (each camera + joint state) onto a nominal 30Hz grid over that window by
nearest-timestamp match.

Output: --out_dir/episode_{0..N-1}.hdf5, one per kept episode, ordered by source dir name (chronological).
Feed --out_dir as a `good` (or `good2`, `good3`, ...) source dir to merge_teleop_dataset.py to combine
with other batches later.

Usage:
    python convert_teleop_dataset.py --data_root ~/Documents/data/20260901_good_data_90hz \\
        --out_dir data/real_pick_yellow_bottle/good --cameras left top
"""
import argparse
import glob
import json
import os

import cv2
import h5py
import numpy as np

FPS = 30.0
PERIOD_NS = 1e9 / FPS


def nearest_indices(sorted_times, query_times):
    """For each query time, index into sorted_times of the closest value."""
    idx = np.searchsorted(sorted_times, query_times)
    idx = np.clip(idx, 1, len(sorted_times) - 1)
    left_closer = (query_times - sorted_times[idx - 1]) <= (sorted_times[idx] - query_times)
    return idx - left_closer.astype(np.int64)


def read_video_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame[:, :, ::-1].copy())  # BGR -> RGB
    cap.release()
    return frames


def convert_episode(ep_dir, cameras, log):
    manifest = json.load(open(os.path.join(ep_dir, "manifest.json")))
    ep_id = manifest["episode_id"]
    if manifest.get("schema_version") != "1.3":
        log(f"{ep_id}: SKIP schema_version={manifest.get('schema_version')} (no combined stream)")
        return None
    if manifest.get("task_label") != "success":
        log(f"{ep_id}: SKIP task_label={manifest.get('task_label')}")
        return None

    ts = manifest["time_sync"]
    video_origin_ns, video_end_ns = float(ts["video_origin_ns"]), float(ts["video_end_ns"])

    with h5py.File(os.path.join(ep_dir, "trajectory.h5")) as f:
        pos = f["trajectories/combined/position"][:]
        vel = f["trajectories/combined/velocity"][:]
        src_t = f["trajectories/combined/source_time_ns"][:].astype(np.float64)

    lo = max(video_origin_ns, src_t[0])
    hi = min(video_end_ns, src_t[-1])
    if hi <= lo:
        log(f"{ep_id}: SKIP empty video/joint intersection window")
        return None
    n_steps = int((hi - lo) // PERIOD_NS) + 1
    out_times = lo + np.arange(n_steps) * PERIOD_NS

    cam_images = {}
    for cam in cameras:
        vp = os.path.join(ep_dir, manifest["video_paths"][cam])
        frames = read_video_frames(vp)
        manifest_n = manifest["sample_counts"].get(cam)
        if len(frames) != manifest_n:
            log(f"{ep_id}: NOTE {cam} manifest sample_counts={manifest_n} but decoded {len(frames)} frames")
        frame_times = video_origin_ns + np.arange(len(frames)) * PERIOD_NS
        fi = nearest_indices(frame_times, out_times)
        cam_images[cam] = np.stack([frames[i] for i in fi], axis=0)

    joint_idx = nearest_indices(src_t, out_times)
    qpos = pos[joint_idx]
    qvel = vel[joint_idx]
    action = np.concatenate([qpos[1:], qpos[-1:]], axis=0)  # action[t] = qpos[t+1]; hold pose at the end

    head_clip = (lo - min(video_origin_ns, src_t[0])) / 1e9
    tail_clip_video = max(0.0, video_end_ns - hi) / 1e9
    tail_clip_joint = max(0.0, src_t[-1] - hi) / 1e9
    flags = []
    if head_clip > 0.3:
        flags.append(f"head_clipped={head_clip:.2f}s")
    if tail_clip_video > 0.3:
        flags.append(f"video_tail_dropped={tail_clip_video:.2f}s")
    if tail_clip_joint > 0.3:
        flags.append(f"joint_tail_dropped={tail_clip_joint:.2f}s")
    flag_str = f"  [{', '.join(flags)}]" if flags else ""
    log(f"{ep_id}: OK n_steps={n_steps} ({n_steps / FPS:.1f}s){flag_str}")

    return {"qpos": qpos, "qvel": qvel, "action": action, "images": cam_images, "episode_id": ep_id}


def write_hdf5(out_path, ep):
    n_steps = ep["qpos"].shape[0]
    with h5py.File(out_path, "w", rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs["sim"] = False
        root.attrs["source_episode_id"] = ep["episode_id"]
        obs = root.create_group("observations")
        image_grp = obs.create_group("images")
        for cam, imgs in ep["images"].items():
            h, w = imgs.shape[1:3]
            dset = image_grp.create_dataset(cam, (n_steps, h, w, 3), dtype="uint8", chunks=(1, h, w, 3))
            dset.attrs["CLASS"] = np.bytes_("IMAGE")
            dset[...] = imgs
        obs.create_dataset("qpos", data=ep["qpos"])
        obs.create_dataset("qvel", data=ep["qvel"])
        root.create_dataset("action", data=ep["action"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True,
                     help="dir containing episode subdirs (each with manifest.json/trajectory.h5/videos/)")
    ap.add_argument("--out_dir", required=True, help="output dir for episode_{i}.hdf5")
    ap.add_argument("--cameras", nargs="+", default=["left", "top"])
    args = ap.parse_args()

    data_root = os.path.expanduser(args.data_root)
    ep_dirs = sorted(glob.glob(os.path.join(data_root, "*")))
    ep_dirs = [d for d in ep_dirs if os.path.isdir(d) and os.path.exists(os.path.join(d, "manifest.json"))]
    print(f"found {len(ep_dirs)} episode dirs under {data_root}\n")

    os.makedirs(args.out_dir, exist_ok=True)

    kept = 0
    max_n_steps = 0
    for ep_dir in ep_dirs:
        result = convert_episode(ep_dir, args.cameras, log=print)
        if result is None:
            continue
        out_path = os.path.join(args.out_dir, f"episode_{kept}.hdf5")
        write_hdf5(out_path, result)
        max_n_steps = max(max_n_steps, result["qpos"].shape[0])
        kept += 1

    print(f"\nwrote {kept} episodes to {args.out_dir} (max episode length: {max_n_steps} steps)")


if __name__ == "__main__":
    main()
