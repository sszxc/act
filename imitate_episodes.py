import os
import time
# Headless/offscreen rendering when no display; avoids "OpenGL platform library has not been loaded".
if 'MUJOCO_GL' not in os.environ:
    os.environ['MUJOCO_GL'] = 'egl'  # Use egl with GPU; use 'osmesa' if no GPU.

import sys
import shutil
import json
from datetime import datetime
import torch
import numpy as np
import cv2
import pickle
import random
import argparse
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
import h5py

from constants import DT, DEFAULT_STATE_DIM, ROOT_DIM, STATE_DIM_ALLEGRO, SIM_TASK_CONFIGS
from constants import ENV_FAMILY_ALLEGRO, ENV_FAMILY_HMF_PROTO5_HAND, ENV_FAMILY_METAWORLD
from constants import HMF_PROTO5_CTRL_DIM, HMF_PROTO5_STATE_DIM
from constants import PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
import utils
import forward_kinematics
from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos, visualize_joints

from sim_env import BOX_POSE, DEX_OBJECT_POSE, HMF_PROTO5_RANDOM_RESET_STATE
from sim_env import sample_dex_object_pose, sample_hmf_proto5_random_reset
from sim_env import episode_reward_meets_success

import IPython
e = IPython.embed


def _save_hydra_config_to_dir(hydra_cfg, output_dir: str) -> None:
    """
    Save Hydra config (including resolved interpolations) to the result dir and copy cwd .hydra if present.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) Save user config (raw and resolved)
    OmegaConf.save(hydra_cfg, os.path.join(output_dir, "config_hydra.yaml"))
    OmegaConf.save(hydra_cfg, os.path.join(output_dir, "config_hydra_resolved.yaml"), resolve=True)

    # 2) Save Hydra runtime info (may be unavailable)
    try:
        runtime_cfg = HydraConfig.get()
        OmegaConf.save(runtime_cfg, os.path.join(output_dir, "hydra_runtime.yaml"), resolve=True)
    except Exception:
        pass

    # 3) Copy .hydra from cwd if it exists
    src_hydra_dir = os.path.join(os.getcwd(), ".hydra")
    if os.path.isdir(src_hydra_dir):
        dst_hydra_dir = os.path.join(output_dir, ".hydra")
        shutil.copytree(src_hydra_dir, dst_hydra_dir, dirs_exist_ok=True)


def train_or_eval(args, hydra_cfg=None):
    # Set RNG seed from config (numpy / torch / data splits)
    set_seed(args['seed'])
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    # get task parameters
    is_sim = task_name[:4] == 'sim_'
    if is_sim:
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]
    # CLI (hydra) overrides; null falls back to the TASK_CONFIGS value
    dataset_dir = args.get('dataset_dir') or task_config['dataset_dir']
    num_episodes = args.get('num_episodes') or task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = list(args.get('camera_names') or task_config['camera_names'])
    image_size = args.get('image_size') or None
    # joint_ids: train on a subset of the 24 joints (e.g. [0..7] = 6 arm + 2 wrist, dropping
    # the 16 finger joints from both observation and action). Sets state/action dim implicitly.
    joint_ids = args.get('joint_ids', None)
    joint_ids = list(joint_ids) if joint_ids is not None else None
    action_repr = args.get('action_repr', 'absolute')
    state_dim = args.get('state_dim') or task_config.get('state_dim', DEFAULT_STATE_DIM)
    action_dim = args.get('action_dim') or task_config.get('action_dim', state_dim)
    if action_repr == 'task_space':
        # FK output (pos3+rot6d6+hand18=27), trimmed by joint_ids (arm-only) to just the
        # hand/wrist dims it keeps -- see utils._task_space_keep_idx.
        state_dim = action_dim = len(utils._task_space_keep_idx(joint_ids)) if joint_ids is not None \
            else forward_kinematics.TASKSPACE_DIM
    elif joint_ids is not None:
        state_dim = action_dim = len(joint_ids)
    env_family = task_config.get('env_family', None)

    # fixed parameters
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'latent_z_dim': args['latent_z_dim'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         'state_dim': state_dim,
                         'action_dim': action_dim,
                         'qpos_dropout': args.get('qpos_dropout', 0.0),
                         'no_encoder': args.get('no_encoder', False),
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names, 'state_dim': state_dim, 'action_dim': action_dim,}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'temporal_agg_k': args.get('temporal_agg_k', 0.01),
        'temporal_agg_newest': args.get('temporal_agg_newest', False),
        'camera_names': camera_names,
        'image_size': image_size,
        'action_repr': action_repr,
        'action_offset': args.get('action_offset', -1),
        'joint_ids': joint_ids,
        'action_stride': args.get('action_stride', 1),
        'num_workers': args.get('num_workers', 1),
        'val_episode_ids': args.get('val_episode_ids', None),
        'amp': args.get('amp', False),
        'resume': args.get('resume', False),
        'resume_every': args.get('resume_every', 100),
        'deploy_every': args.get('deploy_every', 25),
        'save_every': args.get('save_every', 500),
        'env_family': env_family,
        'real_robot': not is_sim,
        # eval-only: user-specified latent z (fixed within rollout)
        'latent_z_sample': args.get('latent_z_sample', None),
        # dataset info for evaluation-time initialization
        'dataset_dir': dataset_dir,
        'num_episodes': num_episodes,
        # whether to overwrite sim initial qpos from dataset
        'init_qpos_from_dataset': args.get('init_qpos_from_dataset', False),
        # direct replay: replay actions from dataset; no policy
        'direct_replay': args.get('direct_replay', False),
        'replay_episode': args.get('replay_episode', 0),
        # Eval: max rollouts whose video/png to save; None = all
        'max_save_episodes': args.get('max_save_episodes', None),
        # Eval: total rollouts to run
        'num_rollouts': args.get('num_rollouts', 50),
    }

    if is_eval:
        # Create eval subdir: eval+<timestamp>
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        eval_dir_name = f"eval_{timestamp}"
        eval_dir = os.path.join(ckpt_dir, eval_dir_name)
        os.makedirs(eval_dir, exist_ok=False)

        # Save eval config for reproducibility
        eval_config_path = os.path.join(eval_dir, "eval_config.yaml")
        OmegaConf.save(OmegaConf.create(config), eval_config_path)

        # Logger: stdout + log file
        log_path = os.path.join(eval_dir, "eval.log")
        log_file = open(log_path, "w")

        def _log(msg: str):
            print(msg)
            print(msg, file=log_file)
            log_file.flush()

        ckpt_names = [args.get('ckpt_name') or 'policy_best.ckpt']
        results = []

        eval_start = time.time()
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(
                config,
                ckpt_name,
                save_episode=True,
                output_dir=eval_dir,
                logger=_log,
                max_save_episodes=config.get('max_save_episodes', None),
            )
            results.append([ckpt_name, success_rate, avg_return])
        eval_elapsed_sec = time.time() - eval_start

        for ckpt_name, success_rate, avg_return in results:
            _log(f'{ckpt_name}: {success_rate=} {avg_return=}')
        _log(f"Total eval time: {eval_elapsed_sec:.2f} seconds")
        _log("")  # blank line

        # Extra JSON summary (success rate, etc.)
        summary_json = {
            "ckpt_dir": ckpt_dir,
            "eval_dir": eval_dir,
            "results": [
                {
                    "ckpt_name": ckpt_name,
                    "success_rate": float(success_rate),
                    "avg_return": float(avg_return),
                }
                for ckpt_name, success_rate, avg_return in results
            ],
            "eval_elapsed_seconds": float(eval_elapsed_sec),
        }
        summary_json_path = os.path.join(eval_dir, "eval_summary.json")
        with open(summary_json_path, "w") as f:
            json.dump(summary_json, f, indent=2)
        _log(f"Saved eval summary to {summary_json_path}")

        log_file.close()
        exit()

    # Train: refuse to overwrite an existing ckpt_dir, unless this is an explicit resume.
    if os.path.exists(ckpt_dir) and not args.get('resume', False):
        raise FileExistsError(
            f'ckpt_dir already exists: {ckpt_dir}. Refusing to overwrite in train mode '
            f'(pass resume=true to continue it instead).'
        )

    # Train: create result dir and save Hydra config
    os.makedirs(ckpt_dir, exist_ok=True)
    if hydra_cfg is not None:
        _save_hydra_config_to_dir(hydra_cfg, ckpt_dir)

    train_dataloader, val_dataloader, stats, _ = load_data(
        dataset_dir,
        num_episodes,
        camera_names,
        batch_size_train,
        batch_size_val,
        num_queries=policy_config['num_queries'],
        task_name=task_name,
        batches_per_epoch=args.get('batches_per_epoch', None),
        image_size=image_size,
        action_repr=config['action_repr'],
        action_offset=config['action_offset'],
        num_workers=config['num_workers'],
        val_episode_ids=config['val_episode_ids'],
        joint_ids=joint_ids,
        action_stride=config['action_stride'],
    )
    # Eval must reproduce the training action representation; keep it with the stats.
    stats['action_repr'] = config['action_repr']
    stats['joint_ids'] = joint_ids
    stats['action_stride'] = config['action_stride']

    # save dataset stats
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
    if 'pca' in stats:
        pca_path = os.path.join(ckpt_dir, 'pca.pkl')
        with open(pca_path, 'wb') as f:
            pickle.dump(stats['pca'], f)
        print(f'Saved PCA to {pca_path}')

    config['deploy_set'] = build_deploy_set(val_dataloader.dataset, stats)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    # policy_best.ckpt is written inside train_bc the moment val loss improves, so there is
    # nothing left to save here unless an older code path handed back a state_dict.
    if best_state_dict is not None:
        torch.save(best_state_dict, os.path.join(ckpt_dir, 'policy_best.ckpt'))
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names, image_size=None):
    curr_images = []
    for cam_name in camera_names:
        img = ts.observation['images'][cam_name]
        if image_size is not None:
            img = cv2.resize(img, (int(image_size[1]), int(image_size[0])), interpolation=cv2.INTER_AREA)
        curr_images.append(rearrange(img, 'h w c -> c h w'))
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def overwrite_sim_qpos_from_dataset(env, task_name, dataset_qpos, env_family=None):
    """
    Override sim initial joint positions with dataset trajectory start qpos.
    Sim only; no-op for real_robot.
    """
    physics = env._physics

    # Bi-manual 14-dim tasks (transfer_cube / insertion)
    if env_family == ENV_FAMILY_METAWORLD or 'sim_transfer_cube' in task_name or 'sim_insertion' in task_name:
        q = np.asarray(dataset_qpos, dtype=np.float32)
        if q.shape[0] < 14:
            raise ValueError(f"Dataset qpos dim {q.shape[0]} < 14; cannot map to sim joints.")
        q = q[:14]
        left_arm = q[:6]
        left_grip_norm = q[6]
        right_arm = q[7:13]
        right_grip_norm = q[13]

        left_grip_pos = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(float(left_grip_norm))
        right_grip_pos = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(float(right_grip_norm))

        with physics.reset_context():
            physics.data.qpos[:6] = left_arm
            physics.data.qpos[8:14] = right_arm
            physics.data.qpos[6] = left_grip_pos
            physics.data.qpos[7] = -left_grip_pos
            physics.data.qpos[14] = right_grip_pos
            physics.data.qpos[15] = -right_grip_pos
            np.copyto(physics.data.ctrl, physics.data.qpos[:16])

    elif env_family == ENV_FAMILY_ALLEGRO:
        q = np.asarray(dataset_qpos, dtype=np.float32)
        if q.shape[0] < 22:
            raise ValueError(f"dex task expects at least 22-dim hand qpos; dataset has {q.shape[0]}.")
        hand_qpos = q[:22]
        with physics.reset_context():
            physics.data.qpos[:22] = hand_qpos
            np.copyto(physics.data.ctrl, physics.data.qpos[:22])

    elif env_family == ENV_FAMILY_HMF_PROTO5_HAND:
        q = np.asarray(dataset_qpos, dtype=np.float32)
        if q.shape[0] < HMF_PROTO5_STATE_DIM:
            raise ValueError(
                f"hmf_proto5_hand expects at least {HMF_PROTO5_STATE_DIM}-dim qpos; dataset has {q.shape[0]}."
            )
        robot_qpos = q[:HMF_PROTO5_STATE_DIM]
        finger_qpos = robot_qpos[-HMF_PROTO5_CTRL_DIM:]
        with physics.reset_context():
            physics.data.qpos[:HMF_PROTO5_STATE_DIM] = robot_qpos
            if physics.model.nu != HMF_PROTO5_CTRL_DIM:
                raise ValueError(f"hmf_proto5_hand expects nu={HMF_PROTO5_CTRL_DIM}, got {physics.model.nu}.")
            physics.data.ctrl[:] = finger_qpos


def sample_dataset_start_qpos(dataset_dir, num_episodes):
    episode_idx = np.random.randint(0, num_episodes)
    dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
    with h5py.File(dataset_path, 'r') as root:
        qpos0 = root['/observations/qpos'][0]
    return np.array(qpos0, dtype=np.float32)


def build_fixed_hmf_proto5_random_reset(task_name, fixed_object_pose):
    random_reset = SIM_TASK_CONFIGS.get(task_name, {}).get("random_reset", {})
    targets = random_reset.get("random_obj_goal", [])
    expected_dim = 3 * len(targets)
    arr = np.asarray(fixed_object_pose, dtype=np.float64).reshape(-1)
    if arr.size != expected_dim:
        raise ValueError(
            f"{task_name} needs {expected_dim}-dim fixed pose "
            f"(3 xyz values per random_reset target), got {arr.size}"
        )

    state = {"random_obj_goal": []}
    for i, target in enumerate(targets):
        state["random_obj_goal"].append({
            "name": target["name"],
            "type": target["type"],
            "position": arr[3 * i:3 * i + 3],
        })

    task_reset_joint = random_reset.get("task_reset_joint")
    if task_reset_joint and task_reset_joint.get("enabled", False):
        state["task_reset_joint"] = dict(task_reset_joint)
    return state


def apply_object_pose_for_reset(task_name, fixed_object_pose, env_family=None):
    """
    Set task-specific object/goal pose before env.reset().
    fixed_object_pose None: task-specific random sample; else fixed vector.
    """
    if env_family is None:
        env_family = SIM_TASK_CONFIGS.get(task_name, {}).get("env_family")

    if fixed_object_pose is not None:
        arr = np.asarray(fixed_object_pose, dtype=np.float64).reshape(-1)
        if env_family == ENV_FAMILY_HMF_PROTO5_HAND:
            HMF_PROTO5_RANDOM_RESET_STATE[0] = build_fixed_hmf_proto5_random_reset(task_name, arr)
        elif 'sim_transfer_cube' in task_name:
            if arr.size != 7:
                raise ValueError(f"sim_transfer_cube needs 7-dim object pose, got {arr.size}")
            BOX_POSE[0] = arr
        elif 'sim_insertion' in task_name:
            if arr.size != 14:
                raise ValueError(f"sim_insertion needs 14-dim object pose, got {arr.size}")
            BOX_POSE[0] = arr
        elif env_family == ENV_FAMILY_ALLEGRO:
            if arr.size != 7:
                raise ValueError(f"dex task needs 7-dim object pose, got {arr.size}")
            DEX_OBJECT_POSE[0] = arr
        else:
            raise NotImplementedError(f"apply_object_pose_for_reset: {task_name}")
    else:
        if 'sim_transfer_cube' in task_name:
            BOX_POSE[0] = sample_box_pose()
        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose())
        elif env_family == ENV_FAMILY_ALLEGRO:
            DEX_OBJECT_POSE[0] = sample_dex_object_pose()
        elif env_family == ENV_FAMILY_HMF_PROTO5_HAND:
            HMF_PROTO5_RANDOM_RESET_STATE[0] = sample_hmf_proto5_random_reset(task_name)


def rollout_single_episode_return(
    policy,
    env,
    config,
    pre_process,
    post_process,
    *,
    pca,
    use_pca_action,
    rollout_latent_z,
    film_theta=None,
    film_pca_theta=None,
    fixed_object_pose=None,
    fixed_init_qpos=None,
    init_qpos_from_dataset=False,
    dataset_dir=None,
    num_episodes=None,
    direct_replay=False,
    replay_qpos0=None,
    replay_actions=None,
    save_episode=False,
    output_dir=None,
    rollout_id=0,
    max_save_episodes=None,
    onscreen_render=False,
    logger=print,
    quiet=False,
):
    """
    One rollout; returns (episode_return, episode_highest_reward).
    Sim: object pose from fixed_object_pose before reset; optional fixed_init_qpos or init_qpos_from_dataset.

    film_theta: flat (2*hidden_dim,) [gamma, beta] for the plain per-channel FiLM path.
    film_pca_theta: flat (2*k,) [gamma, beta] for the PCA-bottleneck FiLM path (see
    detr_vae.py's load_film_pca()); caller must have already called policy.model.load_film_pca()
    so policy.model.film_pca_k == k. Mutually exclusive with film_theta.
    """
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    image_size = config.get('image_size', None)
    state_dim = config['state_dim']
    action_dim = config.get('action_dim', state_dim)
    task_name = config['task_name']
    env_family = config.get('env_family', None)
    temporal_agg = config['temporal_agg']
    temporal_agg_k = float(config.get('temporal_agg_k', 0.01))
    temporal_agg_newest = bool(config.get('temporal_agg_newest', False))
    max_timesteps_cfg = config['episode_len']
    if env_family == ENV_FAMILY_ALLEGRO:
        onscreen_cam = 'default_cam'
    elif env_family == ENV_FAMILY_HMF_PROTO5_HAND:
        onscreen_cam = 'topview'
    else:
        onscreen_cam = 'angle'

    max_timesteps = int(max_timesteps_cfg * 2)
    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    ### direct replay
    if direct_replay and real_robot:
        raise ValueError('direct_replay is sim-only')

    if not real_robot:
        apply_object_pose_for_reset(task_name, fixed_object_pose, env_family=env_family)
    ts = env.reset()

    if not real_robot and fixed_init_qpos is not None:
        try:
            overwrite_sim_qpos_from_dataset(env, task_name, fixed_init_qpos, env_family=env_family)
            new_obs = env._task.get_observation(env._physics)
            ts = ts._replace(observation=new_obs)
            logger("Initialized sim qpos from fixed_init_qpos.")
        except Exception as ex:
            logger(f"[WARN] fixed_init_qpos failed: {ex}")
    elif not real_robot and (init_qpos_from_dataset or direct_replay):
        try:
            if direct_replay:
                dataset_qpos0 = replay_qpos0
            else:
                assert dataset_dir is not None and num_episodes is not None
                dataset_qpos0 = sample_dataset_start_qpos(dataset_dir, num_episodes)
            overwrite_sim_qpos_from_dataset(env, task_name, dataset_qpos0, env_family=env_family)
            new_obs = env._task.get_observation(env._physics)
            ts = ts._replace(observation=new_obs)
            logger("Initialized sim qpos from dataset start qpos.")
        except Exception as ex:
            logger(f"[WARN] init qpos from dataset failed; using default init. error={ex}")

    if onscreen_render:
        ax = plt.subplot()
        plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
        plt.ion()

    steps_this_rollout = min(max_timesteps, replay_actions.shape[0]) if direct_replay else max_timesteps
    if temporal_agg and not direct_replay:
        all_time_actions = torch.zeros([max_timesteps, max_timesteps + num_queries, action_dim]).cuda()

    qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()

    will_save_this_rollout = save_episode and (
        max_save_episodes is None or rollout_id < max_save_episodes
    )
    image_list = [] if will_save_this_rollout else None
    qpos_list = [] if will_save_this_rollout else None
    target_qpos_list = [] if will_save_this_rollout else None
    rewards = []

    step_iter = range(steps_this_rollout)
    if not quiet:
        step_iter = tqdm(step_iter, desc=f"Rollout {rollout_id} steps", leave=False)

    with torch.inference_mode():
        for t in step_iter:
            if onscreen_render:
                image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                plt_img.set_data(image)
                plt.pause(DT)

            obs = ts.observation
            if will_save_this_rollout:
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
            qpos_numpy = np.array(obs['qpos'])
            qpos_numpy = qpos_numpy[:state_dim]
            if will_save_this_rollout:
                qpos_list.append(qpos_numpy)
            if not direct_replay:
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(ts, camera_names, image_size)

            if direct_replay:
                target_qpos = np.asarray(replay_actions[t], dtype=np.float64)
            else:
                if policy_class == "ACT":
                    if t % query_frequency == 0:
                        if film_theta is not None and film_pca_theta is not None:
                            raise ValueError("film_theta and film_pca_theta are mutually exclusive")
                        film_gamma = film_beta = film_pca_gamma = film_pca_beta = None
                        if film_theta is not None:
                            hdim = int(policy.model.visual_film_gamma.numel())
                            theta = np.asarray(film_theta, dtype=np.float32).reshape(-1)
                            if theta.size != 2 * hdim:
                                raise ValueError(f"film_theta dim mismatch: got {theta.size}, expected {2*hdim}")
                            film_gamma = torch.from_numpy(theta[:hdim]).to(device=qpos.device, dtype=qpos.dtype).unsqueeze(0)
                            film_beta = torch.from_numpy(theta[hdim:]).to(device=qpos.device, dtype=qpos.dtype).unsqueeze(0)
                        elif film_pca_theta is not None:
                            k = int(policy.model.film_pca_k)
                            if k <= 0:
                                raise ValueError(
                                    "film_pca_theta given but policy.model has no PCA-bottleneck FiLM "
                                    "basis loaded (film_pca_k=0); call policy.model.load_film_pca() first"
                                )
                            theta = np.asarray(film_pca_theta, dtype=np.float32).reshape(-1)
                            if theta.size != 2 * k:
                                raise ValueError(f"film_pca_theta dim mismatch: got {theta.size}, expected {2*k}")
                            film_pca_gamma = torch.from_numpy(theta[:k]).to(device=qpos.device, dtype=qpos.dtype).unsqueeze(0)
                            film_pca_beta = torch.from_numpy(theta[k:]).to(device=qpos.device, dtype=qpos.dtype).unsqueeze(0)
                        all_actions = policy(
                            qpos,
                            curr_image,
                            latent_z_sample=rollout_latent_z,
                            film_gamma=film_gamma,
                            film_beta=film_beta,
                            film_pca_gamma=film_pca_gamma,
                            film_pca_beta=film_pca_beta,
                        )
                    if temporal_agg:
                        all_time_actions[[t], t:t + num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        # Upstream weights index 0 -- the OLDEST chunk still covering t --
                        # most, so at 30Hz with num_queries=50 the command lags ~0.8s behind
                        # the observation. temporal_agg_newest reverses it; see replay_eval.py.
                        n_agg = len(actions_for_curr_step)
                        order = np.arange(n_agg)[::-1] if temporal_agg_newest else np.arange(n_agg)
                        exp_weights = np.exp(-temporal_agg_k * order)
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif policy_class == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError
                raw_action = raw_action.squeeze(0).cpu().numpy()
                target_qpos = post_process(raw_action, qpos_numpy)
                if use_pca_action and pca is not None:
                    root_6 = target_qpos[:ROOT_DIM]
                    finger_pcs = target_qpos[ROOT_DIM:].reshape(1, -1)
                    finger_16 = pca.inverse_transform(finger_pcs).squeeze(0)
                    target_qpos = np.concatenate([root_6, finger_16]).astype(np.float64)

            if will_save_this_rollout:
                target_qpos_list.append(target_qpos)
            ts = env.step(target_qpos)
            rewards.append(ts.reward)

    if onscreen_render:
        plt.close()

    if real_robot:
        from aloha_scripts.robot_utils import move_grippers
        move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)

    rewards = np.array(rewards)
    episode_return = np.sum(rewards[rewards != None])
    episode_highest_reward = np.max(rewards[rewards != None])

    if will_save_this_rollout and output_dir is not None:
        save_videos(image_list, DT, video_path=os.path.join(output_dir, f'video{rollout_id}.mp4'))
        plot_path = os.path.join(output_dir, f'video{rollout_id}_qpos.png')
        visualize_joints(qpos_list, target_qpos_list, plot_path=plot_path,
                         label_overwrite=('State', 'Command'))

    return float(episode_return), float(episode_highest_reward)


def eval_bc(config, ckpt_name, save_episode=True, output_dir=None, logger=print, max_save_episodes=None):
    # Eval uses same seed for reproducibility; extend config for a separate eval seed if needed
    set_seed(config.get('seed', 0))
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    action_dim = config.get('action_dim', state_dim)
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    image_size = config.get('image_size', None)
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    env_family = config.get('env_family', None)
    temporal_agg = config['temporal_agg']
    dataset_dir = config.get('dataset_dir', None)
    num_episodes = config.get('num_episodes', None)
    init_qpos_from_dataset = config.get('init_qpos_from_dataset', False)
    direct_replay = config.get('direct_replay', False)
    replay_episode = config.get('replay_episode', 0)
    num_rollouts_cfg = int(config.get('num_rollouts', 50))

    use_pca_action = (env_family == ENV_FAMILY_ALLEGRO and action_dim < STATE_DIM_ALLEGRO and not direct_replay)

    latent_z_sample_str = config.get('latent_z_sample', None)
    latent_z_dim = int(policy_config.get('latent_z_dim', 32)) if isinstance(policy_config, dict) else 32

    def _parse_latent_z_sample(s):
        if s is None:
            return None
        s = str(s).strip()
        if s == "" or s.lower() in {"none", "null"}:
            return None
        try:
            if s.startswith("["):
                arr = json.loads(s)
            else:
                arr = [float(x) for x in s.split(",")]
        except Exception as ex:
            raise ValueError(f"Cannot parse --latent_z_sample: {s!r}. Expected 'v1,v2,...' or JSON list string. error={ex}")
        if not isinstance(arr, (list, tuple)):
            raise ValueError(f"--latent_z_sample parsed value is not list/tuple: {type(arr)}")
        if len(arr) != latent_z_dim:
            raise ValueError(f"--latent_z_sample dim mismatch: got {len(arr)}, expected latent_z_dim={latent_z_dim}")
        return torch.tensor(arr, dtype=torch.float32).cuda()

    base_latent_z = _parse_latent_z_sample(latent_z_sample_str)

    if direct_replay and real_robot:
        raise ValueError('direct_replay is sim-only; not for real_robot.')
    if direct_replay:
        assert dataset_dir is not None and num_episodes is not None, \
            "direct_replay requires dataset_dir and num_episodes."

    # Default output_dir to ckpt_dir if unset (legacy)
    if output_dir is None:
        output_dir = ckpt_dir

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    state_dict = torch.load(ckpt_path)
    loading_status = policy.load_state_dict(state_dict, strict=False)
    # Backward-compat: allow missing FiLM buffers introduced later. film_pca_* default to
    # a disabled no-op (film_pca_k=0) until load_film_pca() installs a basis, so it's safe
    # for older checkpoints saved before this buffer existed to be missing them.
    allowed_missing = {
        "model.visual_film_gamma", "model.visual_film_beta",
        # visual / memory (encoder-decoder boundary) / hs (pre-action_head) — see
        # detr_vae.py's load_film_pca(..., target=...).
        "model.film_pca_W", "model.film_pca_mu", "model.film_pca_gamma", "model.film_pca_beta",
        "model.film_pca_memory_W", "model.film_pca_memory_mu",
        "model.film_pca_memory_gamma", "model.film_pca_memory_beta",
        "model.film_pca_hs_W", "model.film_pca_hs_mu",
        "model.film_pca_hs_gamma", "model.film_pca_hs_beta",
    }
    missing = set(getattr(loading_status, "missing_keys", []))
    unexpected = set(getattr(loading_status, "unexpected_keys", []))
    missing_not_allowed = missing - allowed_missing
    if unexpected or missing_not_allowed:
        raise RuntimeError(
            "Error(s) in loading state_dict for ACTPolicy:\n"
            f"\tUnexpected key(s) in state_dict: {sorted(unexpected)}\n"
            f"\tMissing key(s) in state_dict (not allowed): {sorted(missing_not_allowed)}\n"
            f"\tMissing key(s) allowed: {sorted(missing & allowed_missing)}"
        )
    logger(loading_status)
    policy.cuda()
    policy.eval()
    logger(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    # 'delta' policies predict action - qpos[t], so un-normalizing needs the current qpos.
    if stats.get('action_repr') == 'delta':
        post_process = lambda a, q: a * stats['delta_std'] + stats['delta_mean'] + q
    else:
        post_process = lambda a, q: a * stats['action_std'] + stats['action_mean']

    pca = None
    if use_pca_action:
        pca_path = os.path.join(ckpt_dir, 'pca.pkl')
        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)
        logger(f'Loaded PCA from {pca_path} for action inverse transform')

    def load_episode_for_replay(episode_idx):
        """
        Load qpos[0] and full action sequence for an episode (direct replay).
        Returns (qpos0, actions) with actions shape (T, action_dim), raw unnormalized.
        """
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos0 = np.array(root['/observations/qpos'][0], dtype=np.float32)
            actions = np.array(root['/action'][()], dtype=np.float64)
        return qpos0, actions

    max_timesteps = int(max_timesteps * 5) # may increase for real-world tasks

    # load environment
    if real_robot:
        from aloha_scripts.real_env import make_real_env # requires aloha
        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        from sim_env import make_sim_env
        env = make_sim_env(task_name, time_limit=max_timesteps*DT)
        env_max_reward = env.task.max_reward

    num_rollouts = num_rollouts_cfg if not direct_replay else min(num_rollouts_cfg, num_episodes)
    episode_returns = []
    highest_rewards = []
    fixed_object_pose = config.get('fixed_object_pose', None)
    fixed_init_qpos = config.get('fixed_init_qpos', None)
    for rollout_id in range(num_rollouts):
        rollout_latent_z = base_latent_z
        replay_qpos0, replay_actions = None, None
        if direct_replay:
            episode_idx = (replay_episode + rollout_id) % num_episodes
            replay_qpos0, replay_actions = load_episode_for_replay(episode_idx)
            logger(f'[direct_replay] Rollout {rollout_id} replay episode_{episode_idx}, T={len(replay_actions)}')

        episode_return, episode_highest_reward = rollout_single_episode_return(
            policy,
            env,
            config,
            pre_process,
            post_process,
            pca=pca,
            use_pca_action=use_pca_action,
            rollout_latent_z=rollout_latent_z,
            fixed_object_pose=fixed_object_pose,
            fixed_init_qpos=fixed_init_qpos,
            init_qpos_from_dataset=init_qpos_from_dataset,
            dataset_dir=dataset_dir,
            num_episodes=num_episodes,
            direct_replay=direct_replay,
            replay_qpos0=replay_qpos0,
            replay_actions=replay_actions,
            save_episode=save_episode,
            output_dir=output_dir,
            rollout_id=rollout_id,
            max_save_episodes=max_save_episodes,
            onscreen_render=onscreen_render,
            logger=logger,
            quiet=False,
        )
        episode_returns.append(episode_return)
        highest_rewards.append(episode_highest_reward)
        ok = episode_reward_meets_success(task_name, episode_highest_reward, env_max_reward)
        logger(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {ok}')

    success_flags = np.array(
        [episode_reward_meets_success(task_name, h, env_max_reward) for h in highest_rewards],
        dtype=np.float64,
    )
    success_rate = float(np.mean(success_flags))
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(int(env_max_reward) + 1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    logger(summary_str)

    # save success rate and returns to txt
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(output_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return


# bf16 autocast for the train/val forward. Off by default; `amp=true` turns it on. Training is
# GPU-bound here (97% util), so this is roughly a 2x speedup, which is what buys the "train
# well past plateau" runs. bf16 needs no GradScaler. Deploy metrics stay fp32 -- 72 samples,
# and they are the numbers every conclusion is ranked on.
_AMP = False


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=_AMP):
        out = policy(qpos_data, image_data, action_data, is_pad)
    return {k: v.float() for k, v in out.items()}


def build_deploy_set(val_dataset, stats, n_per_episode=8):
    """A fixed (episode, start_ts) sample set, preloaded to GPU, for the deployment metrics.

    Fixed rather than resampled so the metric is comparable across epochs *and* across runs;
    the ordinary val loss redraws random timesteps every epoch and is noisy enough that
    min-over-8000-epochs is partly a lottery.
    """
    items = []
    for episode_id in val_dataset.episode_ids:
        path = os.path.join(val_dataset.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(path, 'r') as root:
            T = int(root['/action'].shape[0])
        for ts in np.linspace(0, T - 1, n_per_episode + 2)[1:-1].astype(int):
            items.append(val_dataset.load(episode_id, start_ts=int(ts)))
    batch = [torch.stack([it[i] for it in items]).cuda() for i in range(4)]
    to_t = lambda k: torch.as_tensor(np.asarray(stats[k], dtype=np.float32)).cuda()
    ref = {k: to_t(k) for k in ('action_mean', 'action_std', 'qpos_mean', 'qpos_std')}
    if 'delta_mean' in stats:
        ref.update(delta_mean=to_t('delta_mean'), delta_std=to_t('delta_std'))
    if 'task_space_mean' in stats:
        ref.update(task_space_mean=to_t('task_space_mean'), task_space_std=to_t('task_space_std'))
    print(f'Deployment metric set: {len(items)} samples from {len(val_dataset.episode_ids)} val episodes')
    return batch, ref


def deploy_metrics(policy, deploy_set, action_repr, chunk=64):
    """Val diagnostics in raw joint radians, on the path the robot actually runs.

    The train/val loss goes through the CVAE *posterior* (it is handed the ground-truth action
    chunk), so it can look healthy while the deployed policy -- prior, z=0, images+qpos only --
    does nothing. These use that deployed path, and being in radians they are comparable across
    action representations and camera sets:
      l1_rad        mean |pred - gt| over the chunk
      skill         l1_rad / l1_rad of "hold current qpos"; < 1 = better than freezing
      motion_ratio  mean|pred - qpos| / mean|gt - qpos|; ~0 is the "robot barely moves" failure

    For action_repr='task_space' the units are mixed (meters + unitless rot6d + radians), so
    'l1_rad' is not literally radians there; skill/motion_ratio (ratios) stay meaningful.
    """
    (image, qpos, action, is_pad), ref = deploy_set
    preds = []
    with torch.inference_mode():
        for i in range(0, len(qpos), chunk):
            preds.append(policy(qpos[i:i + chunk], image[i:i + chunk]))
    a_hat = torch.cat(preds)
    m = (~is_pad).unsqueeze(-1).float()

    if action_repr == 'task_space':
        # No absolute-pose reconstruction here: composing the 6D-rotation delta back onto qpos
        # needs a matrix multiply, not addition -- and it isn't needed anyway. qpos_raw cancels
        # out of every metric below for any delta-style repr (see 'delta' below: pred_abs and
        # gt_abs both add the same qpos_raw, so it drops out of pred_abs-gt_abs, and copy_rad's
        # qpos_raw baseline is exactly "predict zero delta"). So work directly in delta space,
        # via _chunk_metrics with a zero baseline in place of qpos_raw (also gets track_corr).
        denorm = lambda a: a * ref['task_space_std'] + ref['task_space_mean']
        pred_abs, gt_abs = denorm(a_hat), denorm(action)
        return _chunk_metrics(pred_abs, gt_abs, torch.zeros_like(gt_abs), m)

    qpos_raw = (qpos * ref['qpos_std'] + ref['qpos_mean']).unsqueeze(1)  # (B,1,D)
    if action_repr == 'delta':
        denorm = lambda a: a * ref['delta_std'] + ref['delta_mean'] + qpos_raw
    else:
        denorm = lambda a: a * ref['action_std'] + ref['action_mean']
    pred_abs, gt_abs = denorm(a_hat), denorm(action)

    out = _chunk_metrics(pred_abs, gt_abs, qpos_raw, m)
    n_arm = min(8, gt_abs.shape[-1])  # joints 0-7 = 6 arm + 2 wrist, present in every joint set
    if n_arm < gt_abs.shape[-1]:
        out.update({f'arm_{k}': v for k, v in _chunk_metrics(
            pred_abs[..., :n_arm], gt_abs[..., :n_arm], qpos_raw[..., :n_arm], m).items()})
    else:
        out.update({f'arm_{k}': v for k, v in list(out.items())})
    return out


def _chunk_metrics(pred_abs, gt_abs, qpos_raw, m):
    """l1_rad / skill / motion_ratio / track_corr for one set of joint dims. `m` is the
    (B,T,1) non-padding mask; track_corr correlates commanded vs demonstrated displacement
    per joint over all unmasked (sample, chunk-step) pairs, then averages the joints."""
    n = m.sum() * gt_abs.shape[-1]
    mae = lambda x, y: ((x - y).abs() * m).sum() / n
    l1_rad, copy_rad = mae(pred_abs, gt_abs), mae(qpos_raw, gt_abs)
    cd, gd = (pred_abs - qpos_raw) * m, (gt_abs - qpos_raw) * m
    cnt = m.sum()
    cd = (cd - cd.sum((0, 1)) / cnt) * m
    gd = (gd - gd.sum((0, 1)) / cnt) * m
    den = ((cd ** 2).sum((0, 1)) * (gd ** 2).sum((0, 1))).sqrt()
    corr = torch.where(den > 0, (cd * gd).sum((0, 1)) / den.clamp(min=1e-12),
                       torch.zeros_like(den))
    return {
        'l1_rad': l1_rad.item(),
        'skill': (l1_rad / copy_rad).item(),
        'motion_ratio': (mae(pred_abs, qpos_raw) / mae(gt_abs, qpos_raw)).item(),
        'track_corr': corr.mean().item(),
    }


def _hist_to_float(hist):
    """[{k: tensor}] -> [{k: float}], so the resume file stays small and picklable."""
    return [{k: float(v) for k, v in d.items()} for d in hist]


def _hist_to_tensor(hist):
    """Inverse of _hist_to_float; plot_history/compute_dict_mean want tensors."""
    return [{k: torch.tensor(float(v)) for k, v in d.items()} for d in hist]


def save_train_state(path, policy, optimizer, epoch, tr):
    """Everything needed to continue this run, written atomically.

    Model + AdamW state is ~1GB, so this is written every `resume_every` epochs, not every
    epoch. The best-so-far checkpoints are NOT in here -- they are written to their final
    paths the moment they improve, which is both crash-safe and cheaper than carrying three
    deepcopied state_dicts in memory for a multi-thousand-epoch run.
    """
    state = {'epoch': epoch, 'model': policy.state_dict(), 'optimizer': optimizer.state_dict(),
             'trackers': tr,
             'rng': {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(),
                     'numpy': np.random.get_state(), 'python': random.getstate()}}
    torch.save(state, path + '.tmp')
    os.replace(path + '.tmp', path)


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    set_seed(seed)

    global _AMP
    _AMP = bool(config.get('amp', False))
    if _AMP:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print('bf16 autocast enabled for train/val forward')

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    resume_path = os.path.join(ckpt_dir, 'train_state.pt')
    resume_every = int(config.get('resume_every', 100))
    start_epoch, resumed = 0, None
    if config.get('resume', False) and os.path.isfile(resume_path):
        resumed = torch.load(resume_path, map_location='cuda', weights_only=False)
        policy.load_state_dict(resumed['model'])
        optimizer.load_state_dict(resumed['optimizer'])
        start_epoch = int(resumed['epoch'])
        r = resumed['rng']
        torch.set_rng_state(r['torch'].cpu() if torch.is_tensor(r['torch']) else r['torch'])
        torch.cuda.set_rng_state_all([x.cpu() if torch.is_tensor(x) else x for x in r['cuda']])
        np.random.set_state(r['numpy'])
        random.setstate(r['python'])
        print(f'RESUMED from {resume_path} at epoch {start_epoch}')

    # TensorBoard: `tensorboard --logdir <ckpt_dir>/tb` (or the parent results/ dir to
    # compare runs side by side), then forward the port over SSH if training runs remotely.
    tb_writer = SummaryWriter(log_dir=os.path.join(ckpt_dir, 'tb'))

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    # Second, independent selection track: best on the deployed (prior, z=0) path in radians.
    # policy_best.ckpt stays selected by val loss so it matches earlier runs; the deploy pick
    # is saved alongside as policy_best_deploy.ckpt.
    deploy_set = config.get('deploy_set', None)
    deploy_every = int(config.get('deploy_every', 25))
    save_every = int(config.get('save_every', 500))  # 0 = only best/last/best_deploy
    min_l1_rad, best_deploy = np.inf, None
    # Third track: max arm track_corr. l1_rad (like val loss) rewards freezing in place, so on a
    # long run its argmin can be a do-nothing checkpoint; track_corr is scale-free and cannot be
    # gamed that way. Saved as policy_best_track.ckpt.
    max_track = -np.inf
    deploy_history = []
    global_step = 0  # cumulative optimizer.step() calls; standard TensorBoard x-axis
    # Best-so-far checkpoints are written to their final paths the moment they improve rather
    # than deepcopied and held to the end: crash-safe, and it keeps three extra model copies
    # out of RAM on runs that are thousands of epochs long.
    best_epoch = best_deploy_epoch = best_track_epoch = -1
    best_deploy_metrics = best_track_metrics = None
    if resumed is not None:
        t = resumed['trackers']
        min_val_loss, best_epoch = t['min_val_loss'], t['best_epoch']
        min_l1_rad, best_deploy_epoch, best_deploy_metrics = (
            t['min_l1_rad'], t['best_deploy_epoch'], t['best_deploy_metrics'])
        max_track, best_track_epoch, best_track_metrics = (
            t['max_track'], t['best_track_epoch'], t['best_track_metrics'])
        deploy_history, global_step = t['deploy_history'], t['global_step']
        train_history = _hist_to_tensor(t['train_history'])
        validation_history = _hist_to_tensor(t['validation_history'])

    def _trackers():
        return {'min_val_loss': float(min_val_loss), 'best_epoch': int(best_epoch),
                'min_l1_rad': float(min_l1_rad), 'best_deploy_epoch': int(best_deploy_epoch),
                'best_deploy_metrics': best_deploy_metrics,
                'max_track': float(max_track), 'best_track_epoch': int(best_track_epoch),
                'best_track_metrics': best_track_metrics,
                'deploy_history': deploy_history, 'global_step': global_step,
                'train_history': _hist_to_float(train_history),
                'validation_history': _hist_to_float(validation_history)}

    def _write_deploy_json():
        with open(os.path.join(ckpt_dir, 'deploy_metrics.json'), 'w') as f:
            json.dump({'best_deploy_epoch': best_deploy_epoch, **(best_deploy_metrics or {}),
                       'best_track_epoch': best_track_epoch,
                       'best_track_metrics': best_track_metrics,
                       'best_val_loss': float(min_val_loss),
                       'best_val_loss_epoch': int(best_epoch),
                       'history': deploy_history}, f, indent=2)

    for epoch in tqdm(range(start_epoch, num_epochs), initial=start_epoch, total=num_epochs):
        print(f'\nEpoch {epoch}')
        if deploy_set is not None and (epoch % deploy_every == 0 or epoch == num_epochs - 1):
            policy.eval()
            dm = deploy_metrics(policy, deploy_set, config.get('action_repr', 'absolute'))
            for k, v in dm.items():
                tb_writer.add_scalar(f'deploy/{k}', v, global_step)
            deploy_history.append({'epoch': epoch, **dm})
            print('deploy: ' + ' '.join(f'{k}={v:.4f}' for k, v in dm.items()))
            if dm['l1_rad'] < min_l1_rad:
                min_l1_rad, best_deploy_epoch, best_deploy_metrics = dm['l1_rad'], epoch, dm
                torch.save(policy.state_dict(), os.path.join(ckpt_dir, 'policy_best_deploy.ckpt'))
            if dm.get('arm_track_corr', dm['track_corr']) > max_track:
                max_track, best_track_epoch, best_track_metrics = (
                    dm.get('arm_track_corr', dm['track_corr']), epoch, dm)
                torch.save(policy.state_dict(), os.path.join(ckpt_dir, 'policy_best_track.ckpt'))
            _write_deploy_json()   # partial results survive a crash / an interrupted long run
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss, best_epoch = epoch_val_loss, epoch
                best_ckpt_info = (epoch, min_val_loss, None)
                torch.save(policy.state_dict(), os.path.join(ckpt_dir, 'policy_best.ckpt'))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
            tb_writer.add_scalar(f'val/{k}', v.item(), global_step)
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
            global_step += 1
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
            tb_writer.add_scalar(f'train/{k}', v.item(), global_step)
        print(summary_string)

        if save_every and epoch % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)
        if resume_every and (epoch + 1) % resume_every == 0:
            torch.save(policy.state_dict(), os.path.join(ckpt_dir, 'policy_last.ckpt'))
            save_train_state(resume_path, policy, optimizer, epoch + 1, _trackers())

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    save_train_state(resume_path, policy, optimizer, num_epochs, _trackers())
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')
    if best_track_metrics is not None:
        print(f'Best track ckpt @ epoch {best_track_epoch}: '
              + ' '.join(f'{k}={v:.4f}' for k, v in best_track_metrics.items()))
    if best_deploy_metrics is not None:
        _write_deploy_json()
        print(f'Best deploy ckpt @ epoch {best_deploy_epoch}: '
              + ' '.join(f'{k}={v:.4f}' for k, v in best_deploy_metrics.items()))

    tb_writer.close()

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


# Eval args are parsed before Hydra; ckpt_dir is always CLI-provided.
_EVAL_ARGS = None


def _parse_eval_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--init_qpos_from_dataset', action='store_true',
                        help='Eval (sim): override init qpos with a random trajectory start from the dataset')
    parser.add_argument('--direct_replay', action='store_true',
                        help='Eval: replay dataset actions in sim (no policy); init from that trajectory qpos[0]')
    parser.add_argument('--replay_episode', type=int, default=0,
                        help='direct_replay start episode index (rollout i replays episode replay_episode+i)')
    parser.add_argument('--temporal_agg', action='store_true',
                        help='Eval: temporal aggregation to smooth ACT output')
    parser.add_argument('--temporal_agg_newest', action='store_true',
                        help='Eval: weight the NEWEST chunk prediction most instead of the oldest '
                             '(upstream weights the oldest, costing ~0.8s of lag at 30Hz)')
    parser.add_argument('--temporal_agg_k', type=float, default=0.01,
                        help='Eval: temporal-aggregation decay rate (default 0.01, near-uniform)')
    parser.add_argument('--max_save_episodes', type=int, default=None,
                        help='Eval: only save video/png for first N rollouts; default all')
    parser.add_argument('--num_rollouts', type=int, default=50,
                        help='Eval total rollouts (default 50; direct_replay uses min(num_rollouts, num_episodes))')
    parser.add_argument('--latent_z_sample', type=str, default=None,
                        help='(eval only) Fixed latent z for whole rollout. Format: "v1,...,vD" or JSON list e.g. "[0,0.1,...]". Dim = latent_z_dim.')
    parser.add_argument('--ckpt_dir', type=str, required=True,
                        help='checkpoint dir (required; provided via CLI only)')
    parser.add_argument('--ckpt_name', type=str, default=None,
                        help="Eval: which checkpoint in ckpt_dir (default policy_best.ckpt; "
                             "policy_best_deploy.ckpt is the one picked by the deployed-path metric)")
    eval_args, unknown = parser.parse_known_args()
    return eval_args, unknown


@hydra.main(config_path='config', config_name='config', version_base=None)
def main(cfg):
    global _EVAL_ARGS
    args_dict = OmegaConf.to_container(cfg, resolve=True)
    args_dict['eval'] = _EVAL_ARGS.eval
    args_dict['onscreen_render'] = _EVAL_ARGS.onscreen_render
    args_dict['init_qpos_from_dataset'] = _EVAL_ARGS.init_qpos_from_dataset
    args_dict['direct_replay'] = _EVAL_ARGS.direct_replay
    args_dict['replay_episode'] = _EVAL_ARGS.replay_episode
    args_dict['temporal_agg'] = _EVAL_ARGS.temporal_agg
    args_dict['temporal_agg_newest'] = _EVAL_ARGS.temporal_agg_newest
    args_dict['temporal_agg_k'] = _EVAL_ARGS.temporal_agg_k
    args_dict['ckpt_dir'] = _EVAL_ARGS.ckpt_dir
    args_dict['ckpt_name'] = _EVAL_ARGS.ckpt_name
    args_dict['max_save_episodes'] = _EVAL_ARGS.max_save_episodes
    args_dict['num_rollouts'] = _EVAL_ARGS.num_rollouts
    args_dict['latent_z_sample'] = _EVAL_ARGS.latent_z_sample
    train_or_eval(args_dict, hydra_cfg=cfg)


if __name__ == '__main__':
    _EVAL_ARGS, unknown = _parse_eval_args()
    sys.argv = [sys.argv[0]] + unknown
    main()
