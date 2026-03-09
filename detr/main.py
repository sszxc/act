# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from types import SimpleNamespace

import numpy as np
import torch
from .models import build_ACT_model, build_CNNMLP_model

import IPython
e = IPython.embed

# Defaults for DETR/backbone/transformer; merged with policy_config from imitate_episodes (Hydra).
DETR_DEFAULTS = {
    'lr': 1e-4,
    'lr_backbone': 1e-5,
    'weight_decay': 1e-4,
    'backbone': 'resnet18',
    'dilation': False,
    'position_embedding': 'sine',
    'masks': False,
    'enc_layers': 4,
    'dec_layers': 7,
    'dim_feedforward': 2048,
    'hidden_dim': 256,
    'dropout': 0.1,
    'nheads': 8,
    'num_queries': 400,
    'pre_norm': False,
    'camera_names': [],
    'state_dim': 14,
    'action_dim': None,
}


def _config_to_args(config_dict):
    merged = {**DETR_DEFAULTS, **config_dict}
    return SimpleNamespace(**merged)


def build_ACT_model_and_optimizer(args_override):
    args = _config_to_args(args_override)
    model = build_ACT_model(args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return model, optimizer


def build_CNNMLP_model_and_optimizer(args_override):
    args = _config_to_args(args_override)
    model = build_CNNMLP_model(args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return model, optimizer
