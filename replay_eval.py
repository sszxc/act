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

Observations stay teacher-forced by default (--qpos_source dataset): the qpos fed to the policy
at every step is the recorded one, so compounding error in the proprio channel is invisible.
--qpos_source policy closes that loop -- the policy is fed its OWN previous command instead
(perfect-tracking assumption, since there's no sim/robot here to actually move) -- for one
action-selection mode at a time (--replay_mode). Images are still teacher-forced either way.
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


def denormalize(raw, qpos_ref, stats, action_repr):
    if action_repr == 'delta':
        return raw * stats['delta_std'] + stats['delta_mean'] + qpos_ref
    return raw * stats['action_std'] + stats['action_mean']


def rollout_closed_loop(policy, root, camera_names, image_size, qpos0, T, stats, action_repr,
                         chunk_size, temporal_agg, k=TEMPORAL_AGG_K, newest_first=False):
    """Closed-loop replay: the qpos fed to the policy is its OWN previous command, not the
    recording (perfect-tracking assumption -- there's no sim/robot here to give real dynamics).
    Images still come from the recording. temporal_agg=False re-queries every chunk_size steps
    (mirrors chunked()); temporal_agg=True queries every step and aggregates overlapping chunks,
    same weighting as temporal_ensemble() (mirrors what eval_bc does in a real rollout).
    """
    query_every = 1 if temporal_agg else chunk_size
    D = qpos0.shape[0]
    all_time_actions = np.zeros((T, T + chunk_size, D)) if temporal_agg else None
    qpos_sim = qpos0.astype(np.float64).copy()
    cmd = np.zeros((T, D))
    chunk = None
    with torch.inference_mode():
        for t in range(T):
            if t % query_every == 0:
                img = load_cam_images(root, camera_names, t, image_size)[None]
                img = torch.from_numpy(img).cuda().permute(0, 1, 4, 2, 3).float() / 255.0
                qpos_n = torch.from_numpy(
                    (qpos_sim - stats['qpos_mean']) / stats['qpos_std']).float().cuda()[None]
                chunk = policy(qpos_n, img)[0].cpu().numpy()  # (chunk_size, D)
            if temporal_agg:
                all_time_actions[t, t:t + chunk_size] = chunk
                lo = max(0, t - chunk_size + 1)
                preds = all_time_actions[lo:t + 1, t]
                j = np.arange(len(preds))
                w = np.exp(-k * (j[::-1] if newest_first else j))
                raw = (preds * (w / w.sum())[:, None]).sum(0)
            else:
                raw = chunk[t % chunk_size]
            cmd[t] = denormalize(raw, qpos_sim, stats, action_repr)
            qpos_sim = cmd[t]
    return cmd


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
    p.add_argument('--qpos_source', choices=['dataset', 'policy'], default='dataset',
                   help="'dataset' (default): fully teacher-forced, as above. 'policy': feed the "
                        "policy its own previous command as qpos instead (perfect-tracking "
                        "assumption -- no sim/robot here), showing compounding error; runs a "
                        "single mode (--replay_mode) instead of the full sweep, since each mode "
                        "would otherwise drift onto its own qpos trajectory.")
    p.add_argument('--replay_mode', choices=['chunked', 'temporal_agg'], default='temporal_agg',
                   help='action-selection mode driving the closed loop; only used with '
                        '--qpos_source policy')
    p.add_argument('--temporal_agg_k', type=float, default=TEMPORAL_AGG_K)
    p.add_argument('--temporal_agg_newest', action='store_true')
    args = apply_run_defaults(p.parse_args(), p)

    with open(os.path.join(args.ckpt_dir, 'dataset_stats.pkl'), 'rb') as f:
        stats = pickle.load(f)
    action_repr = stats.get('action_repr', 'absolute')
    # joint_ids: the run trained on a subset of the 24 joints, so every recorded array has to be
    # sliced the same way before it can be compared with what the policy emits.
    joint_ids = stats.get('joint_ids', None)
    joint_ids = np.asarray(joint_ids, dtype=int) if joint_ids is not None else None
    if joint_ids is not None:
        args.state_dim = len(joint_ids)
        print(f'joint subset from dataset_stats.pkl: {joint_ids.tolist()}')

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
    # (--qpos_source policy instead runs a single closed-loop mode; see rollout_closed_loop.)
    if args.qpos_source == 'dataset':
        modes = {'chunked': chunked, 'temporal_agg': temporal_ensemble}
        for k in (0.01, 0.1, 0.5):
            modes[f'agg_newest_k{k}'] = (lambda kk: lambda a: temporal_ensemble(a, True, kk))(k)
    else:
        modes = {args.replay_mode: None}
    # Also accumulate the same five over arm+wrist dims 0-7 alone (`arm_*`): a 24-joint run's
    # plain numbers are three-quarters finger error, so they cannot be compared with an
    # 8-joint run's without this restriction.
    KEYS = ('cmd_l1', 'freeze_l1', 'motion', 'gt_motion', 'track_corr')
    acc = {m: {k: [] for k in KEYS + tuple('arm_' + k for k in KEYS)} for m in modes}

    plots = {}
    for ep in sorted(val_indices.tolist()):
        with h5py.File(os.path.join(args.dataset_dir, f'episode_{ep}.hdf5'), 'r') as root:
            qpos = root['/observations/qpos'][()]
            gt_action = root['/action'][()]
            if joint_ids is not None:
                qpos, gt_action = qpos[:, joint_ids], gt_action[:, joint_ids]

            if args.qpos_source == 'dataset':
                qpos_n = torch.from_numpy(
                    (qpos - stats['qpos_mean']) / stats['qpos_std']).float().cuda()
                all_actions = predict_chunks(policy, root, args.camera_names, args.image_size, qpos_n)
                cmds = {name: denormalize(fn(all_actions), qpos, stats, action_repr)
                        for name, fn in modes.items()}
            else:
                cmds = {args.replay_mode: rollout_closed_loop(
                    policy, root, args.camera_names, args.image_size, qpos[0], len(qpos),
                    stats, action_repr, args.chunk_size,
                    temporal_agg=(args.replay_mode == 'temporal_agg'),
                    k=args.temporal_agg_k, newest_first=args.temporal_agg_newest)}

        for name, cmd in cmds.items():
            for pre, sl in (('', slice(None)), ('arm_', slice(0, min(8, cmd.shape[1])))):
                c, g, q = cmd[:, sl], gt_action[:, sl], qpos[:, sl]
                acc[name][pre + 'cmd_l1'].append(np.abs(c - g).mean())
                acc[name][pre + 'freeze_l1'].append(np.abs(q - g).mean())
                acc[name][pre + 'motion'].append(np.abs(c - q).mean())
                acc[name][pre + 'gt_motion'].append(np.abs(g - q).mean())
                cd, gd = c - q, g - q
                cd = cd - cd.mean(0); gd = gd - gd.mean(0)
                den = np.sqrt((cd ** 2).sum(0) * (gd ** 2).sum(0))
                acc[name][pre + 'track_corr'].append(np.nanmean(
                    np.where(den > 0, (cd * gd).sum(0) / np.where(den > 0, den, 1), np.nan)))
            if args.plot:
                plots.setdefault(ep, {})[name] = cmd
        if ep in plots:
            save_plot(args.ckpt_dir, ep, qpos, gt_action, plots.pop(ep))
        print(f'  episode {ep}: T={len(qpos)}')

    result = {'ckpt': os.path.join(args.ckpt_dir, args.ckpt_name), 'action_repr': action_repr,
              'val_episodes': sorted(val_indices.tolist())}
    for name in modes:
        m = {k: float(np.mean(v)) for k, v in acc[name].items()}
        for pre in ('', 'arm_'):
            m[pre + 'skill'] = m[pre + 'cmd_l1'] / m[pre + 'freeze_l1']
            m[pre + 'motion_ratio'] = m[pre + 'motion'] / m[pre + 'gt_motion']
        result[name] = m
        print(f"{name:19s} arm: cmd_l1={m['arm_cmd_l1']:.5f} skill={m['arm_skill']:.3f} "
              f"motion_ratio={m['arm_motion_ratio']:.3f} track_corr={m['arm_track_corr']:.3f}"
              f"   (all dims: skill={m['skill']:.3f} track_corr={m['track_corr']:.3f})")

    out = args.out or os.path.join(args.ckpt_dir, 'replay_eval.json')
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'wrote {out}')


COLORS = {'chunked': 'tab:green', 'temporal_agg': 'tab:red',
          'agg_newest_k0.01': 'tab:blue', 'agg_newest_k0.1': 'tab:purple',
          'agg_newest_k0.5': 'tab:orange'}


def save_plot(ckpt_dir, ep, qpos, gt_action, cmds):
    dims = [d for d in (0, 1, 2, 6, 10, 14) if d < gt_action.shape[1]]  # arm + hand joints
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
