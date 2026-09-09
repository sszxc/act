"""Re-score every run of a sweep on ONE common prediction horizon.

`deploy/track_corr` is computed over a run's whole chunk, so it is not comparable across
chunk_size: a longer chunk covers more demonstrated motion, which sits further above the
command-error floor and inflates the correlation for free. Same for l1_rad, which simply grows
with the horizon. This re-scores each checkpoint on its first `--seconds` of prediction only
(chunk steps 0..round(30*seconds/stride)), which is the same physical horizon for every run
regardless of chunk_size or action_stride.

    python eval_common_horizon.py results/sweep_20260908 --seconds 1.0 \
        --csv results/sweep_20260908/horizon_1s.csv
"""
import argparse, csv, os, pickle, sys
import numpy as np
import torch
import yaml

from imitate_episodes import build_deploy_set, make_policy, _chunk_metrics
from utils import EpisodicDataset


def score(run_dir, seconds, ckpt_name):
    cfg = yaml.safe_load(open(os.path.join(run_dir, 'config_hydra_resolved.yaml')))
    stats = pickle.load(open(os.path.join(run_dir, 'dataset_stats.pkl'), 'rb'))
    ckpt = os.path.join(run_dir, ckpt_name)
    if not os.path.isfile(ckpt):
        return None
    jid = stats.get('joint_ids')
    stride = int(stats.get('action_stride', 1))
    nq = int(cfg['chunk_size'])
    n_steps = max(1, min(nq, int(round(30 * seconds / stride))))

    ds = EpisodicDataset(np.array(sorted(cfg['val_episode_ids'])), cfg['dataset_dir'],
                         list(cfg['camera_names']), stats, num_queries=nq,
                         image_size=cfg.get('image_size'), action_repr=cfg['action_repr'],
                         action_offset=cfg['action_offset'], joint_ids=jid,
                         action_stride=stride)
    (image, qpos, action, is_pad), ref = build_deploy_set(ds, stats)

    dim = len(jid) if jid else 24
    policy = make_policy('ACT', {
        'lr': 1e-4, 'num_queries': nq, 'kl_weight': 1, 'hidden_dim': cfg['hidden_dim'],
        'dim_feedforward': cfg['dim_feedforward'], 'latent_z_dim': 32, 'lr_backbone': 1e-5,
        'backbone': 'resnet18', 'enc_layers': 4, 'dec_layers': 7, 'nheads': 8,
        'camera_names': list(cfg['camera_names']), 'state_dim': dim, 'action_dim': dim,
        'qpos_dropout': cfg.get('qpos_dropout', 0.0), 'no_encoder': cfg.get('no_encoder', False)})
    policy.load_state_dict(torch.load(ckpt, map_location='cuda'), strict=False)
    policy.cuda().eval()

    preds = []
    with torch.inference_mode():
        for i in range(0, len(qpos), 64):
            preds.append(policy(qpos[i:i + 64], image[i:i + 64]))
    a_hat = torch.cat(preds)[:, :n_steps]
    action, is_pad = action[:, :n_steps], is_pad[:, :n_steps]
    m = (~is_pad).unsqueeze(-1).float()
    qpos_raw = (qpos * ref['qpos_std'] + ref['qpos_mean']).unsqueeze(1)
    denorm = (lambda a: a * ref['delta_std'] + ref['delta_mean'] + qpos_raw) \
        if cfg['action_repr'] == 'delta' else (lambda a: a * ref['action_std'] + ref['action_mean'])
    pred_abs, gt_abs = denorm(a_hat), denorm(action)
    n_arm = min(8, dim)
    out = _chunk_metrics(pred_abs[..., :n_arm], gt_abs[..., :n_arm], qpos_raw[..., :n_arm], m)
    out.update(run=os.path.basename(run_dir), chunk=nq, stride=stride, n_steps=n_steps,
               joints=dim, seconds_covered=round(n_steps * stride / 30, 3))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('sweep_dir')
    p.add_argument('--seconds', type=float, default=1.0)
    p.add_argument('--ckpt_name', default='policy_best_track.ckpt')
    p.add_argument('--csv', default=None)
    a = p.parse_args()
    rows = []
    for n in sorted(os.listdir(a.sweep_dir)):
        d = os.path.join(a.sweep_dir, n)
        if not os.path.isfile(os.path.join(d, 'config_hydra_resolved.yaml')):
            continue
        try:
            r = score(d, a.seconds, a.ckpt_name)
        except Exception as e:
            print(f'{n}: FAILED {e}', file=sys.stderr); continue
        if r:
            rows.append(r)
    if not rows:
        return 1
    rows.sort(key=lambda r: -r['track_corr'])
    cols = ['run', 'joints', 'chunk', 'stride', 'n_steps', 'seconds_covered',
            'track_corr', 'skill', 'motion_ratio', 'l1_rad']
    fmt = lambda v: f'{v:.4f}' if isinstance(v, float) else str(v)
    w = {c: max(len(c), *(len(fmt(r[c])) for r in rows)) for c in cols}
    print(f'\n=== arm joints 0-7, first {a.seconds}s of every prediction, {a.ckpt_name} ===')
    print('  '.join(c.ljust(w[c]) for c in cols))
    for r in rows:
        print('  '.join(fmt(r[c]).ljust(w[c]) for c in cols))
    if a.csv:
        with open(a.csv, 'w', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=cols); wr.writeheader()
            wr.writerows([{c: r[c] for c in cols} for r in rows])
        print(f'\nwrote {a.csv}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
