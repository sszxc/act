"""Open-loop replay evaluation: no robot, no sim.

Feeds a checkpoint the *recorded* observations of held-out episodes, runs the exact action
selection eval_bc() uses (chunked, or temporal ensembling), and compares the commands it would
have sent against the recorded ones. Observations stay teacher-forced, so this does not capture
compounding error -- but it does directly measure the failure reported from the robot ("outputs
are very small, the robot barely moves"), which the training loss cannot:

  cmd_l1        mean |command - recorded action|, radians
  freeze_l1     same for the trivial "hold current qpos" policy
  skill         cmd_l1 / freeze_l1;  < 1 means better than freezing, > 1 means worse
  motion        mean |command - qpos[t]|, radians -- how far the robot is asked to move
  gt_motion     same for the recorded action
  motion_ratio  motion / gt_motion;  << 1 IS the "barely moves" failure, ~1 is healthy
  track_corr    per-joint Pearson r between commanded motion (cmd - qpos) and recorded motion
                (action - qpos), averaged over joints. Scale-free, so unlike cmd_l1 it cannot be
                improved by simply commanding less motion -- a frozen policy scores ~0, not ~1.
                Read it WITH motion_ratio: useful means r high AND motion_ratio near 1.

    python replay_eval.py --ckpt_dir results/... --chunk_size 50 --camera_names left top
"""
import argparse
import json
import os
import pickle

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from imitate_episodes import make_policy
from utils import load_cam_images, set_seed

TEMPORAL_AGG_K = 0.01  # matches rollout_single_episode_return()


def run_defaults(ckpt_dir):
    """Read a run's own config_hydra_resolved.yaml so eval can't silently disagree with training.
    qpos_dropout in particular: >=1 means the policy was trained with no proprio at all, and
    feeding it real qpos at eval would be off-distribution."""
    path = os.path.join(ckpt_dir, 'config_hydra_resolved.yaml')
    if not os.path.isfile(path):
        return {}
    import yaml
    cfg = yaml.safe_load(open(path))
    keys = ('camera_names', 'chunk_size', 'qpos_dropout', 'action_offset', 'dataset_dir',
            'num_episodes', 'val_episode_ids', 'hidden_dim', 'dim_feedforward', 'seed')
    return {k: cfg[k] for k in keys if cfg.get(k) is not None}


def apply_run_defaults(args, parser):
    """CLI values the user actually typed win; everything else falls back to the run's config."""
    defaults = run_defaults(args.ckpt_dir)
    for k, v in defaults.items():
        if hasattr(args, k) and getattr(args, k) == parser.get_default(k):
            setattr(args, k, v)
    return args


def predict_chunks(policy, root, camera_names, image_size, qpos_n, batch=32):
    """all_actions[t] = the policy's chunk prediction at timestep t, normalized. (T, nq, D)"""
    T = len(qpos_n)
    out = []
    with torch.inference_mode():
        for i in range(0, T, batch):
            idx = range(i, min(T, i + batch))
            imgs = np.stack([load_cam_images(root, camera_names, t, image_size) for t in idx])
            img = torch.from_numpy(imgs).cuda().permute(0, 1, 4, 2, 3).float() / 255.0
            out.append(policy(qpos_n[i:i + len(imgs)], img).cpu())
    return torch.cat(out).numpy()


def temporal_ensemble(all_actions, newest_first=False, k=TEMPORAL_AGG_K):
    """Average, for time t, the predictions of every chunk that covers it.

    Upstream ACT weights exp(-k*j) with j=0 the *oldest* chunk still in range, i.e. it weights
    stale predictions most. At 30Hz with nq=50 that is up to 1.7s of history at near-uniform
    weight (k=0.01 only decays to 0.61 across the window) -- a low-pass filter with ~0.8s of
    lag, which is what the replay plots show. newest_first=True reverses the weighting so the
    freshest prediction dominates; `chunked` is the no-ensembling reference.
    """
    T, nq, D = all_actions.shape
    out = np.zeros((T, D), dtype=np.float64)
    for t in range(T):
        lo = max(0, t - nq + 1)
        preds = np.stack([all_actions[i, t - i] for i in range(lo, t + 1)])
        j = np.arange(len(preds))
        w = np.exp(-k * (j[::-1] if newest_first else j))
        out[t] = (preds * (w / w.sum())[:, None]).sum(0)
    return out


def chunked(all_actions):
    """No temporal ensembling: re-query every nq steps, execute that chunk open-loop."""
    T, nq, D = all_actions.shape
    return np.stack([all_actions[t - t % nq, t % nq] for t in range(T)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_dir', required=True)
    p.add_argument('--ckpt_name', default='policy_best.ckpt')
    p.add_argument('--dataset_dir', default='data/real_pick_yellow_bottle/good_41')
    p.add_argument('--num_episodes', type=int, default=41)
    p.add_argument('--camera_names', nargs='+', default=['left', 'top'])
    p.add_argument('--image_size', nargs=2, type=int, default=[240, 320])
    p.add_argument('--chunk_size', type=int, default=50)
    p.add_argument('--hidden_dim', type=int, default=512)
    p.add_argument('--dim_feedforward', type=int, default=3200)
    p.add_argument('--state_dim', type=int, default=24)
    p.add_argument('--qpos_dropout', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--val_episode_ids', nargs='+', type=int, default=None,
                   help='explicit held-out episodes; must match training '
                        '(default: reproduce the seeded 80/20 split)')
    p.add_argument('--max_episodes', type=int, default=None, help='cap val episodes, for speed')
    p.add_argument('--plot', action='store_true', help='save per-episode command-vs-recorded plots')
    p.add_argument('--out', default=None, help='json output path (default <ckpt_dir>/replay_eval.json)')
    args = apply_run_defaults(p.parse_args(), p)

    with open(os.path.join(args.ckpt_dir, 'dataset_stats.pkl'), 'rb') as f:
        stats = pickle.load(f)
    action_repr = stats.get('action_repr', 'absolute')

    if args.val_episode_ids:
        val_indices = np.array(sorted(args.val_episode_ids))
    else:
        set_seed(args.seed)          # same split the run trained with
        idx = np.random.permutation(args.num_episodes)
        val_indices = idx[int(0.8 * args.num_episodes):]
    if args.max_episodes:
        val_indices = val_indices[:args.max_episodes]

    policy = make_policy('ACT', {
        'lr': 1e-4, 'num_queries': args.chunk_size, 'kl_weight': 1,
        'hidden_dim': args.hidden_dim, 'dim_feedforward': args.dim_feedforward,
        'latent_z_dim': 32, 'lr_backbone': 1e-5, 'backbone': 'resnet18',
        'enc_layers': 4, 'dec_layers': 7, 'nheads': 8,
        'camera_names': args.camera_names, 'state_dim': args.state_dim,
        'action_dim': args.state_dim, 'qpos_dropout': args.qpos_dropout,
    })
    policy.load_state_dict(torch.load(os.path.join(args.ckpt_dir, args.ckpt_name),
                                      map_location='cuda'), strict=False)
    policy.cuda().eval()

    # Action selection is a free sweep here: the chunk predictions are computed once and every
    # mode is a different weighted average of them. 'temporal_agg' is what eval_bc does today.
    modes = {'chunked': chunked, 'temporal_agg': temporal_ensemble}
    for k in (0.01, 0.1, 0.5):
        modes[f'agg_newest_k{k}'] = (lambda kk: lambda a: temporal_ensemble(a, True, kk))(k)
    acc = {m: {k: [] for k in ('cmd_l1', 'freeze_l1', 'motion', 'gt_motion', 'track_corr')}
           for m in modes}

    plots = {}
    for ep in sorted(val_indices.tolist()):
        with h5py.File(os.path.join(args.dataset_dir, f'episode_{ep}.hdf5'), 'r') as root:
            qpos = root['/observations/qpos'][()]
            gt_action = root['/action'][()]
            qpos_n = torch.from_numpy(
                (qpos - stats['qpos_mean']) / stats['qpos_std']).float().cuda()
            all_actions = predict_chunks(policy, root, args.camera_names, args.image_size, qpos_n)

        for name, fn in modes.items():
            raw = fn(all_actions)
            if action_repr == 'delta':
                cmd = raw * stats['delta_std'] + stats['delta_mean'] + qpos
            else:
                cmd = raw * stats['action_std'] + stats['action_mean']
            acc[name]['cmd_l1'].append(np.abs(cmd - gt_action).mean())
            acc[name]['freeze_l1'].append(np.abs(qpos - gt_action).mean())
            acc[name]['motion'].append(np.abs(cmd - qpos).mean())
            acc[name]['gt_motion'].append(np.abs(gt_action - qpos).mean())
            cd, gd = cmd - qpos, gt_action - qpos
            cd = cd - cd.mean(0); gd = gd - gd.mean(0)
            den = np.sqrt((cd ** 2).sum(0) * (gd ** 2).sum(0))
            acc[name]['track_corr'].append(
                np.nanmean(np.where(den > 0, (cd * gd).sum(0) / np.where(den > 0, den, 1), np.nan)))
            if args.plot:
                plots.setdefault(ep, {})[name] = cmd
        if ep in plots:
            save_plot(args.ckpt_dir, ep, qpos, gt_action, plots.pop(ep))
        print(f'  episode {ep}: T={len(qpos)}')

    result = {'ckpt': os.path.join(args.ckpt_dir, args.ckpt_name), 'action_repr': action_repr,
              'val_episodes': sorted(val_indices.tolist())}
    for name in modes:
        m = {k: float(np.mean(v)) for k, v in acc[name].items()}
        m['skill'] = m['cmd_l1'] / m['freeze_l1']
        m['motion_ratio'] = m['motion'] / m['gt_motion']
        result[name] = m
        print(f"{name:19s} cmd_l1={m['cmd_l1']:.5f} skill={m['skill']:.3f} "
              f"motion_ratio={m['motion_ratio']:.3f} track_corr={m['track_corr']:.3f}")

    out = args.out or os.path.join(args.ckpt_dir, 'replay_eval.json')
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'wrote {out}')


COLORS = {'chunked': 'tab:green', 'temporal_agg': 'tab:red',
          'agg_newest_k0.01': 'tab:blue', 'agg_newest_k0.1': 'tab:purple',
          'agg_newest_k0.5': 'tab:orange'}


def save_plot(ckpt_dir, ep, qpos, gt_action, cmds):
    dims = [0, 1, 2, 6, 10, 14]  # 3 arm joints + 3 hand joints
    fig, axes = plt.subplots(len(dims), 1, figsize=(12, 2 * len(dims)), sharex=True)
    for ax, d in zip(axes, dims):
        ax.plot(gt_action[:, d], 'k--', lw=1.2, label='recorded action')
        for name, cmd in cmds.items():
            ax.plot(cmd[:, d], color=COLORS.get(name), lw=1, alpha=0.85, label=name)
        ax.set_ylabel(f'dim {d}')
    axes[0].legend(fontsize=8, ncol=4)
    axes[0].set_title(f'episode {ep} — open-loop replay, action selection compared')
    fig.tight_layout()
    fig.savefig(os.path.join(ckpt_dir, f'replay_ep{ep}.png'), dpi=90)
    plt.close(fig)


if __name__ == '__main__':
    main()
