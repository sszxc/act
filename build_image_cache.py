"""Pre-resized dataset cache: keeps only the cameras/resolution a sweep actually trains on.

The source hdf5s store 9 cameras at native resolution (~200GB/batch), so random-frame reads
are the training bottleneck -- the GPU sits idle. A 2-camera 240x320 copy is ~25GB, fits in
page cache, and is otherwise a drop-in dataset_dir (train with image_size=null).

  python build_image_cache.py --src data/real_pick_yellow_bottle/good_41 \
      --out data/real_pick_yellow_bottle/good_41_tw240 --cams top wrist --size 240 320
"""
import argparse, glob, os
from multiprocessing import Pool
import cv2, h5py, numpy as np


def convert(job):
    src, dst, cams, H, W = job
    if os.path.exists(dst):
        return f'skip {dst}'
    tmp = dst + '.tmp'
    with h5py.File(src, 'r') as r, h5py.File(tmp, 'w') as w:
        for k, v in r.attrs.items():
            w.attrs[k] = v
        w.create_dataset('action', data=r['/action'][()])
        g = w.create_group('observations')
        for k in ('qpos', 'qvel'):
            g.create_dataset(k, data=r[f'/observations/{k}'][()])
        gi = g.create_group('images')
        for c in cams:
            d = r[f'/observations/images/{c}']
            T = d.shape[0]
            out = gi.create_dataset(c, (T, H, W, 3), dtype='uint8', chunks=(1, H, W, 3))
            for i0 in range(0, T, 64):  # block reads; per-frame reads on the source are slow
                blk = d[i0:i0 + 64]
                out[i0:i0 + len(blk)] = np.stack(
                    [cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA) for f in blk])
    os.replace(tmp, dst)
    return f'done {dst}'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--cams', nargs='+', default=['top', 'wrist'])
    ap.add_argument('--size', nargs=2, type=int, default=[240, 320])
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.src, 'episode_*.hdf5')),
                   key=lambda p: int(p.split('_')[-1].split('.')[0]))
    jobs = [(f, os.path.join(a.out, os.path.basename(f)), a.cams, a.size[0], a.size[1])
            for f in files]
    with Pool(a.workers) as p:
        for i, msg in enumerate(p.imap_unordered(convert, jobs)):
            print(f'[{i+1}/{len(jobs)}] {msg}', flush=True)
