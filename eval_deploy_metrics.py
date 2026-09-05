"""Score an existing checkpoint with the deployment metrics (deploy/{l1_rad,skill,motion_ratio}),
without a robot. Reproduces the run's val split from its seed, so it also works on checkpoints
trained before those metrics existed.

    python eval_deploy_metrics.py --ckpt_dir results/.../combo_aggressive --chunk_size 50 \
        --camera_names left top --image_size 240 320
"""
import argparse
import json
import os
import pickle

import numpy as np
import torch

from imitate_episodes import build_deploy_set, deploy_metrics, make_policy
from utils import EpisodicDataset, get_norm_stats, set_seed


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
    p.add_argument('--action_repr', default=None, help='default: read from dataset_stats.pkl')
    p.add_argument('--action_offset', type=int, default=-1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n_per_episode', type=int, default=8)
    args = p.parse_args()

    with open(os.path.join(args.ckpt_dir, 'dataset_stats.pkl'), 'rb') as f:
        stats = pickle.load(f)
    action_repr = args.action_repr or stats.get('action_repr', 'absolute')
    if action_repr == 'delta' and 'delta_mean' not in stats:
        stats.update(get_norm_stats(args.dataset_dir, args.num_episodes,
                                    num_queries=args.chunk_size, action_offset=args.action_offset))

    # Same split as training: set_seed(seed) then permutation(num_episodes), 80/20.
    set_seed(args.seed)
    idx = np.random.permutation(args.num_episodes)
    val_indices = idx[int(0.8 * args.num_episodes):]
    print(f'val episodes: {sorted(val_indices.tolist())}')

    val_dataset = EpisodicDataset(val_indices, args.dataset_dir, args.camera_names, stats,
                                  num_queries=args.chunk_size, image_size=args.image_size,
                                  action_repr=action_repr, action_offset=args.action_offset)
    deploy_set = build_deploy_set(val_dataset, stats, n_per_episode=args.n_per_episode)

    policy = make_policy('ACT', {
        'lr': 1e-4, 'num_queries': args.chunk_size, 'kl_weight': 1,
        'hidden_dim': args.hidden_dim, 'dim_feedforward': args.dim_feedforward,
        'latent_z_dim': 32, 'lr_backbone': 1e-5, 'backbone': 'resnet18',
        'enc_layers': 4, 'dec_layers': 7, 'nheads': 8,
        'camera_names': args.camera_names, 'state_dim': args.state_dim,
        'action_dim': args.state_dim,
    })
    sd = torch.load(os.path.join(args.ckpt_dir, args.ckpt_name), map_location='cuda')
    print(policy.load_state_dict(sd, strict=False))
    policy.cuda().eval()

    m = deploy_metrics(policy, deploy_set, action_repr)
    print(json.dumps({'ckpt': os.path.join(args.ckpt_dir, args.ckpt_name),
                      'action_repr': action_repr, **m}, indent=2))


if __name__ == '__main__':
    main()
