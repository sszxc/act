"""Collect a sweep directory into one table.

Reads each run's config_hydra_resolved.yaml + deploy_metrics.json (+ replay_eval.json if
present) and prints/writes a CSV sorted by the deployment metric. `--pick` writes the winning
run's action-representation settings as shell vars, which is how run_sweep.sh chains stages.

    python summarize_sweep.py results/sweep_20260904 --csv results/sweep_20260904/summary.csv
    python summarize_sweep.py results/sweep_20260904 --prefix s1_ --pick winner.env
"""
import argparse
import csv
import json
import os
import sys

import yaml

FIELDS = ['run', 'action_repr', 'action_offset', 'qpos_dropout', 'camera_names', 'dataset',
          'seed', 'l1_rad', 'skill', 'motion_ratio', 'best_deploy_epoch',
          'val_loss', 'val_loss_epoch', 'replay_skill', 'replay_motion_ratio', 'status']


def collect(sweep_dir, prefix=''):
    rows = []
    for name in sorted(os.listdir(sweep_dir)):
        run = os.path.join(sweep_dir, name)
        cfg_path = os.path.join(run, 'config_hydra_resolved.yaml')
        if not (name.startswith(prefix) and os.path.isfile(cfg_path)):
            continue
        cfg = yaml.safe_load(open(cfg_path))
        row = {f: '' for f in FIELDS}
        row.update(
            run=name,
            action_repr=cfg.get('action_repr', 'absolute'),
            action_offset=cfg.get('action_offset', -1),
            qpos_dropout=cfg.get('qpos_dropout', 0.0),
            camera_names='+'.join(cfg.get('camera_names') or ['<task default>']),
            dataset=os.path.basename(cfg.get('dataset_dir') or '<task default>'),
            seed=cfg.get('seed', 0),
            status='running',
        )
        dm_path = os.path.join(run, 'deploy_metrics.json')
        if os.path.isfile(dm_path):
            dm = json.load(open(dm_path))
            row.update(status='done', l1_rad=round(dm['l1_rad'], 5), skill=round(dm['skill'], 4),
                       motion_ratio=round(dm['motion_ratio'], 4),
                       best_deploy_epoch=dm['best_deploy_epoch'],
                       val_loss=round(dm['best_val_loss'], 5),
                       val_loss_epoch=dm['best_val_loss_epoch'])
        re_path = os.path.join(run, 'replay_eval.json')
        if os.path.isfile(re_path):
            r = json.load(open(re_path)).get('temporal_agg', {})
            row.update(replay_skill=round(r.get('skill', float('nan')), 4),
                       replay_motion_ratio=round(r.get('motion_ratio', float('nan')), 4))
        rows.append(row)
    done = [r for r in rows if r['status'] == 'done']
    rest = [r for r in rows if r['status'] != 'done']
    return sorted(done, key=lambda r: r['l1_rad']) + rest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('sweep_dir')
    p.add_argument('--prefix', default='')
    p.add_argument('--csv', default=None)
    p.add_argument('--pick', default=None, help='write the best run\'s settings as shell vars here')
    args = p.parse_args()

    rows = collect(args.sweep_dir, args.prefix)
    if not rows:
        print(f'no runs matching {args.prefix!r} in {args.sweep_dir}', file=sys.stderr)
        return 1

    show = ['run', 'action_repr', 'action_offset', 'qpos_dropout', 'camera_names', 'dataset',
            'seed', 'l1_rad', 'skill', 'motion_ratio', 'replay_skill', 'status']
    w = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in show}
    print('  '.join(c.ljust(w[c]) for c in show))
    for r in rows:
        print('  '.join(str(r[c]).ljust(w[c]) for c in show))

    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=FIELDS)
            wr.writeheader()
            wr.writerows(rows)
        print(f'\nwrote {args.csv}')

    if args.pick:
        best = next((r for r in rows if r['status'] == 'done'), None)
        if best is None:
            print('nothing finished; cannot pick', file=sys.stderr)
            return 1
        with open(args.pick, 'w') as f:
            f.write(f"WINNER_RUN={best['run']}\n")
            f.write(f"ACTION_REPR={best['action_repr']}\n")
            f.write(f"ACTION_OFFSET={best['action_offset']}\n")
            f.write(f"QPOS_DROPOUT={best['qpos_dropout']}\n")
        print(f"\npicked {best['run']} (l1_rad={best['l1_rad']}) -> {args.pick}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
