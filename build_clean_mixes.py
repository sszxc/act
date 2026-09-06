"""Rebuild the mixing ladder with pause-stripped scripted episodes.

The first ladder showed that adding raw scripted data collapses commanded motion
(motion_ratio 0.320 -> 0.018 at 19 scripted episodes). The scripted batch moves at half the
human speed and 15.7% of its frames sit in qualifying pauses, against 1.4% for human teleop, so
mixing it in shifts the L1 median toward "do not move". These dirs are identical to the originals
except the scripted entries point at scripted_0904_c19_clean, isolating that one variable.
"""
import json
import os

ROOT = 'data/real_pick_yellow_bottle'
VAL = [0, 3, 6, 9, 19, 21, 23, 24, 39]
TRAIN_H = [i for i in range(41) if i not in VAL]
MIXES = {'cmix_h32_s0': (32, 0), 'cmix_h32_s5': (32, 5), 'cmix_h32_s10': (32, 10),
         'cmix_h32_s19': (32, 19), 'cmix_h16_s19': (16, 19), 'cmix_h0_s19': (0, 19)}


def human_target(i):
    """good_41 entries are themselves symlinks; resolve to the real file, keep the link relative."""
    return os.path.relpath(os.path.realpath(f'{ROOT}/good_41/episode_{i}.hdf5'), ROOT)


def main():
    for name, (nh, ns) in MIXES.items():
        d = os.path.join(ROOT, name)
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))
        manifest, idx = {}, 0
        for i in VAL + TRAIN_H[:nh]:
            os.symlink(os.path.join('..', human_target(i)), f'{d}/episode_{idx}.hdf5')
            manifest[idx] = {'source_dir': 'good_41', 'source_episode': i, 'kind': 'human',
                             'role': 'val' if i in VAL else 'train'}
            idx += 1
        for i in range(ns):
            os.symlink(f'../scripted_0904_c19_clean/episode_{i}.hdf5', f'{d}/episode_{idx}.hdf5')
            manifest[idx] = {'source_dir': 'scripted_0904_c19_clean', 'source_episode': i,
                             'kind': 'scripted_clean', 'role': 'train'}
            idx += 1
        json.dump(manifest, open(f'{d}/manifest.json', 'w'), indent=1)
        bad = [k for k in range(idx) if not os.path.exists(f'{d}/episode_{k}.hdf5')]
        print(f'{name:14s} {idx:3d} episodes  ({nh}H train + {ns}S clean + 9 val)  broken={len(bad)}')


if __name__ == '__main__':
    main()
