"""Merge multiple manually-labeled good/bad session folders into two flat,
contiguously re-indexed datasets, ordered by each episode's original file mtime.

Layout expected under --task_dir (e.g. data/sim_hmf_proto5_teleop/pick):
    good, good2, good3, ...   (any dir matching good\\d*)
    bad, bad2, bad3, ...      (any dir matching bad\\d*)
Each source dir holds `episode_{idx}.hdf5` plus optional companions
(`episode_{idx}_qpos.png`, `episode_{idx}_video.mp4`, ...).

Output: two sibling dirs `good_<N>` and `bad_<N>` (N = merged episode count),
renumbered 0..N-1 in mtime order across all source dirs of that label.

Default mode symlinks to the originals (non-destructive) and writes a
manifest.json recording new_index -> (source_dir, source_episode, mtime) for
provenance. --move instead physically moves the files (no symlinks, no
manifest) and deletes the now-empty source dirs once done.

Usage:
    python merge_teleop_dataset.py --task_dir data/sim_hmf_proto5_teleop/pick
    python merge_teleop_dataset.py --task_dir data/sim_hmf_proto5_teleop/pick --overwrite
    python merge_teleop_dataset.py --task_dir data/sim_hmf_proto5_teleop/pick --dry_run
    python merge_teleop_dataset.py --task_dir data/sim_hmf_proto5_teleop/pick --move --overwrite
"""
import argparse
import json
import os
import re
import shutil
import sys

EPISODE_RE = re.compile(r'^episode_(\d+)(?=[._])')


def find_source_dirs(task_dir, label):
    """Dirs directly under task_dir matching label + optional digits, e.g. good, good2, good3.
    Excludes merged outputs (good_58) since those contain an underscore before the digits."""
    pattern = re.compile(rf'^{label}\d*$')
    dirs = []
    for name in sorted(os.listdir(task_dir)):
        full = os.path.join(task_dir, name)
        if os.path.isdir(full) and pattern.match(name):
            dirs.append(full)
    return dirs


def collect_episodes(src_dir):
    """Group files in src_dir by episode index. Returns {idx: {'hdf5': path, 'files': [(new_suffix, path), ...], 'mtime': float}}."""
    groups = {}
    for name in os.listdir(src_dir):
        m = EPISODE_RE.match(name)
        if not m:
            continue
        idx = int(m.group(1))
        suffix = name[len(m.group(0)):]  # e.g. '.hdf5', '_qpos.png', '_video.mp4'
        groups.setdefault(idx, []).append((suffix, os.path.join(src_dir, name)))

    episodes = {}
    for idx, files in groups.items():
        hdf5_path = next((p for suf, p in files if suf == '.hdf5'), None)
        if hdf5_path is None:
            print(f'  WARNING: {src_dir}/episode_{idx}.* has no .hdf5, skipping', file=sys.stderr)
            continue
        episodes[idx] = {
            'files': files,
            'mtime': os.path.getmtime(hdf5_path),
        }
    return episodes


def merge_label(task_dir, label, overwrite, dry_run, move):
    src_dirs = find_source_dirs(task_dir, label)
    if not src_dirs:
        print(f'[{label}] no source dirs found (looked for {label}\\d* under {task_dir}), skipping')
        return

    print(f'[{label}] source dirs: {[os.path.basename(d) for d in src_dirs]}')
    entries = []  # (mtime, src_dir_name, orig_idx, files)
    for src_dir in src_dirs:
        episodes = collect_episodes(src_dir)
        print(f'  {os.path.basename(src_dir)}: {len(episodes)} episodes')
        for idx, info in episodes.items():
            entries.append((info['mtime'], os.path.basename(src_dir), idx, info['files']))

    entries.sort(key=lambda e: e[0])  # chronological order across all source dirs
    n = len(entries)
    if n == 0:
        print(f'[{label}] 0 episodes found, skipping')
        return

    out_dir = os.path.join(task_dir, f'{label}_{n}')
    print(f'[{label}] -> {n} episodes total -> {out_dir}')

    if dry_run:
        for new_idx, (mtime, src_name, orig_idx, files) in enumerate(entries):
            print(f'    {new_idx:3d} <- {src_name}/episode_{orig_idx}')
        return

    if os.path.exists(out_dir):
        if not overwrite:
            print(f'  ERROR: {out_dir} already exists (use --overwrite to replace it)', file=sys.stderr)
            return
        # safety: only remove if every entry is a symlink or our manifest, never real files
        for name in os.listdir(out_dir):
            p = os.path.join(out_dir, name)
            if name != 'manifest.json' and not os.path.islink(p):
                print(f'  ERROR: {out_dir} contains a non-symlink, non-manifest file '
                      f'({name}); refusing to overwrite', file=sys.stderr)
                return
        shutil.rmtree(out_dir)

    os.makedirs(out_dir)
    manifest = []
    for new_idx, (mtime, src_name, orig_idx, files) in enumerate(entries):
        for suffix, src_path in files:
            dest_path = os.path.join(out_dir, f'episode_{new_idx}{suffix}')
            if move:
                os.rename(src_path, dest_path)
            else:
                os.symlink(os.path.realpath(src_path), dest_path)
        manifest.append({
            'index': new_idx,
            'source_dir': src_name,
            'source_episode': orig_idx,
            'mtime': mtime,
        })

    if move:
        for src_dir in src_dirs:
            shutil.rmtree(src_dir)
        print(f'  wrote {out_dir} ({n} episodes, moved); removed source dirs '
              f'{[os.path.basename(d) for d in src_dirs]}')
    else:
        with open(os.path.join(out_dir, 'manifest.json'), 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f'  wrote {out_dir} ({n} episodes, symlinked) + manifest.json')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--task_dir', required=True, help='e.g. data/sim_hmf_proto5_teleop/pick')
    ap.add_argument('--overwrite', action='store_true', help='replace an existing good_N/bad_N output dir')
    ap.add_argument('--dry_run', action='store_true', help='print the merge plan without writing anything')
    ap.add_argument('--move', action='store_true',
                     help='physically move files instead of symlinking, skip manifest.json, '
                          'and delete the source dirs once moved (destructive, no undo)')
    args = ap.parse_args()

    for label in ('good', 'bad'):
        merge_label(args.task_dir, label, args.overwrite, args.dry_run, args.move)


if __name__ == '__main__':
    main()
