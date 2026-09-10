"""Recreate a symlink-merge dataset dir (good_41, scripted_50_clean, h41_s50c_tw240, ...) pointing
at resynced_* source dirs instead of the originals -- same episode_N -> source mapping, just with
the first path component of each symlink target's relative path prefixed 'resynced_'.

Usage:
    python mirror_symlink_dir.py --src data/real_pick_yellow_bottle/good_41 \\
        --out data/real_pick_yellow_bottle/resynced_good_41
"""
import argparse, glob, os

ap = argparse.ArgumentParser()
ap.add_argument('--src', required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
n = 0
for f in sorted(glob.glob(os.path.join(a.src, 'episode_*.hdf5'))):
    target = os.readlink(f)
    assert not os.path.isabs(target), f'{f} -> {target} is not a relative symlink'
    parts = target.split('/')  # e.g. ['..', 'good_0901_c20', 'episode_0.hdf5']
    up, name, rest = parts[0], parts[1], parts[2:]
    new_name = name if name.startswith('resynced_') else f'resynced_{name}'
    new_target = '/'.join([up, new_name] + rest)
    dst = os.path.join(a.out, os.path.basename(f))
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(new_target, dst)
    n += 1
    resolved = os.path.realpath(dst)
    if not os.path.exists(resolved):
        print(f'WARNING: {dst} -> {new_target} does not resolve ({resolved} missing)')

print(f'{a.out}: {n} symlinks written (mirrored from {a.src})')
