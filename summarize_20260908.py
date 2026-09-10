"""Collect the 2026-09-08 sweep into one table.

Adds what summarize_sweep.py lacks for this sweep: track_corr, the arm-only (joints 0-7)
restriction that makes 8-joint and 24-joint runs comparable, the track-selected checkpoint,
and a plateau column answering "did the extra epochs buy anything".

    python summarize_20260908.py results/sweep_20260908 --csv results/sweep_20260908/summary.csv
"""
import argparse, csv, json, os, sys
import numpy as np
import yaml

FIELDS = ['run', 'joints', 'chunk', 'stride', 'cvae', 'bs', 'lr', 'dataset', 'seed', 'epochs',
          'track_plateau', 'track_plateau_sd', 'track_smooth', 'track_final', 'arm_track_corr', 'arm_skill', 'arm_motion_ratio',
          'arm_l1_rad', 'best_track_epoch', 'plateau_epoch', 'late_gain_pct',
          'l1_rad_best', 'best_deploy_epoch', 'val_loss', 'status']


def plateau_level(history, key, from_epoch_frac=0.4):
    """Mean +- sd of `key` over every evaluation in the last 60% of training.

    This is the headline statistic. The max over ~50 evaluations (what policy_best_track.ckpt
    is selected by) is optimistically biased -- r0's 0.234 at epoch 400 is one spike on a curve
    whose post-plateau mean is 0.192 -- and the within-run epoch-to-epoch sd measured here
    (~0.010) is the smallest difference worth discussing at all.
    """
    if len(history) < 10:
        return '', ''
    e = np.array([h['epoch'] for h in history])
    v = np.array([h.get(key, np.nan) for h in history], dtype=float)
    sel = v[e >= from_epoch_frac * e.max()]
    return round(float(sel.mean()), 4), round(float(sel.std(ddof=1)), 4)


def smoothed(history, key, w=5):
    """(max, final) of a `w`-point running mean. The raw max over ~50 evaluations is
    optimistically biased -- r0's headline 0.234 at epoch 400 is a single-point spike on a
    curve that otherwise sits at 0.19 -- so the smoothed peak is the fair point estimate and
    the smoothed tail says where a long run actually ends up."""
    v = np.array([h.get(key, np.nan) for h in history], dtype=float)
    if len(v) < w:
        return '', ''
    k = np.convolve(v, np.ones(w) / w, mode='valid')
    return round(float(k.max()), 4), round(float(k[-1]), 4)


def plateau(history, key):
    """Epoch at which a 5-point running mean of `key` first came within 2% of its best, and
    how much of the total gain arrived after the halfway point of training (late_gain_pct)."""
    if len(history) < 10:
        return '', ''
    e = np.array([h['epoch'] for h in history])
    v = np.array([h.get(key, np.nan) for h in history], dtype=float)
    k = np.convolve(v, np.ones(5) / 5, mode='valid')
    ek = e[4:]
    best = k.max()
    first = ek[np.argmax(k >= best - 0.02 * abs(best))]
    half = k[ek >= e.max() / 2]
    early_best = k[ek < e.max() / 2].max() if (ek < e.max() / 2).any() else k[0]
    late = (best - early_best) / abs(best) * 100 if best else 0.0
    return int(first), round(float(late), 1)


def collect(sweep_dir):
    rows = []
    for name in sorted(os.listdir(sweep_dir)):
        run = os.path.join(sweep_dir, name)
        cfg_path = os.path.join(run, 'config_hydra_resolved.yaml')
        if not os.path.isfile(cfg_path):
            continue
        cfg = yaml.safe_load(open(cfg_path))
        jid = cfg.get('joint_ids')
        row = {f: '' for f in FIELDS}
        row.update(run=name, joints=len(jid) if jid else 24, chunk=cfg.get('chunk_size'),
                   cvae='none' if cfg.get('no_encoder') else f"kl{cfg.get('kl_weight')}",
                   bs=cfg.get('batch_size'), lr=cfg.get('lr'),
                   dataset=os.path.basename(cfg.get('dataset_dir') or ''),
                   seed=cfg.get('seed'), epochs=cfg.get('num_epochs'),
                   stride=cfg.get('action_stride', 1), status='running')
        dm_path = os.path.join(run, 'deploy_metrics.json')
        if os.path.isfile(dm_path):
            dm = json.load(open(dm_path))
            t = dm.get('best_track_metrics', dm)
            r4 = lambda k: round(t.get(k, t.get(k.replace('arm_', ''), float('nan'))), 4)
            pe, lg = plateau(dm.get('history', []), 'arm_track_corr')
            ts, tf = smoothed(dm.get('history', []), 'arm_track_corr')
            pl, plsd = plateau_level(dm.get('history', []), 'arm_track_corr')
            row.update(status='done', track_plateau=pl, track_plateau_sd=plsd,
                       track_smooth=ts, track_final=tf,
                       arm_track_corr=r4('arm_track_corr'), arm_skill=r4('arm_skill'),
                       arm_motion_ratio=r4('arm_motion_ratio'),
                       arm_l1_rad=round(t.get('arm_l1_rad', t.get('l1_rad', float('nan'))), 5),
                       best_track_epoch=dm.get('best_track_epoch', ''),
                       plateau_epoch=pe, late_gain_pct=lg,
                       l1_rad_best=round(dm['l1_rad'], 5),
                       best_deploy_epoch=dm['best_deploy_epoch'],
                       val_loss=round(dm['best_val_loss'], 5))
        rows.append(row)
    done = [r for r in rows if r['status'] == 'done']
    rest = [r for r in rows if r['status'] != 'done']
    return sorted(done, key=lambda r: -(r['track_plateau'] or 0)) + rest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('sweep_dir')
    p.add_argument('--csv', default=None)
    a = p.parse_args()
    rows = collect(a.sweep_dir)
    if not rows:
        print(f'no runs in {a.sweep_dir}', file=sys.stderr)
        return 1
    show = [f for f in FIELDS if f not in ('best_deploy_epoch', 'l1_rad_best')]
    w = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in show}
    print('  '.join(c.ljust(w[c]) for c in show))
    for r in rows:
        print('  '.join(str(r[c]).ljust(w[c]) for c in show))
    if a.csv:
        with open(a.csv, 'w', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=FIELDS); wr.writeheader(); wr.writerows(rows)
        print(f'\nwrote {a.csv}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
