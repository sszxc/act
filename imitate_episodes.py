import os
import time
# 无显示器时使用离屏渲染，避免 "OpenGL platform library has not been loaded" 错误
if 'MUJOCO_GL' not in os.environ:
    os.environ['MUJOCO_GL'] = 'egl'  # 有 GPU 用 egl；无 GPU 可改为 'osmesa'

import sys
import shutil
import json
from datetime import datetime
import torch
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
import h5py

from constants import DT, DEFAULT_STATE_DIM, ROOT_DIM, STATE_DIM_DEX
from constants import PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos, visualize_joints

from sim_env import BOX_POSE, DEX_OBJECT_POSE, sample_dex_object_pose

import IPython
e = IPython.embed


def _save_hydra_config_to_dir(hydra_cfg, output_dir: str) -> None:
    """
    将 Hydra 的 config（含插值解析版本）保存到指定结果目录，并尽量复制当前运行目录下的 .hydra 目录。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) 保存用户配置（原始 & resolve 后）
    OmegaConf.save(hydra_cfg, os.path.join(output_dir, "config_hydra.yaml"))
    OmegaConf.save(hydra_cfg, os.path.join(output_dir, "config_hydra_resolved.yaml"), resolve=True)

    # 2) 保存 Hydra 运行时信息（不保证一定可用）
    try:
        runtime_cfg = HydraConfig.get()
        OmegaConf.save(runtime_cfg, os.path.join(output_dir, "hydra_runtime.yaml"), resolve=True)
    except Exception:
        pass

    # 3) 复制当前工作目录下的 .hydra（如果存在）
    src_hydra_dir = os.path.join(os.getcwd(), ".hydra")
    if os.path.isdir(src_hydra_dir):
        dst_hydra_dir = os.path.join(output_dir, ".hydra")
        shutil.copytree(src_hydra_dir, dst_hydra_dir, dirs_exist_ok=True)


def train_or_eval(args, hydra_cfg=None):
    # 根据配置中的 seed 统一设置随机种子（影响 numpy / torch 以及数据划分等）
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
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']
    state_dim = task_config.get('state_dim', DEFAULT_STATE_DIM)
    action_dim = task_config.get('action_dim', state_dim)

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
        'camera_names': camera_names,
        'real_robot': not is_sim,
        # dataset info for evaluation-time initialization
        'dataset_dir': dataset_dir,
        'num_episodes': num_episodes,
        # whether to overwrite sim initial qpos from dataset
        'init_qpos_from_dataset': args.get('init_qpos_from_dataset', False),
        # direct replay: 用数据集中某条轨迹的 action 序列直接控制 sim，不跑 policy
        'direct_replay': args.get('direct_replay', False),
        'replay_episode': args.get('replay_episode', 0),
        # eval 时最多保存多少条 rollout 的视频和 png；None 表示全部保存
        'max_save_episodes': args.get('max_save_episodes', None),
    }

    if is_eval:
        # 为本次 eval 创建独立子目录：eval+时间戳
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        eval_dir_name = f"eval_{timestamp}"
        eval_dir = os.path.join(ckpt_dir, eval_dir_name)
        os.makedirs(eval_dir, exist_ok=False)

        # 保存本次 eval 使用的 config 参数，便于之后复现
        eval_config_path = os.path.join(eval_dir, "eval_config.yaml")
        OmegaConf.save(OmegaConf.create(config), eval_config_path)

        # 简单的 logger：既打印到 stdout，也写入 log 文件
        log_path = os.path.join(eval_dir, "eval.log")
        log_file = open(log_path, "w")

        def _log(msg: str):
            print(msg)
            print(msg, file=log_file)
            log_file.flush()

        ckpt_names = [f'policy_best.ckpt']
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
        _log("")  # 换行

        # 额外保存一个 json，总结成功率等指标
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

    # train 模式：若 ckpt_dir 已存在则报错，避免覆盖
    if os.path.exists(ckpt_dir):
        raise FileExistsError(
            f'ckpt_dir already exists: {ckpt_dir}. Refusing to overwrite in train mode.'
        )

    # 训练开始：先创建结果目录并保存 Hydra 配置
    os.makedirs(ckpt_dir, exist_ok=False)
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
    )

    # save dataset stats
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
    if 'pca' in stats:
        pca_path = os.path.join(ckpt_dir, 'pca.pkl')
        with open(pca_path, 'wb') as f:
            pickle.dump(stats['pca'], f)
        print(f'Saved PCA to {pca_path}')

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
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


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def eval_bc(config, ckpt_name, save_episode=True, output_dir=None, logger=print, max_save_episodes=None):
    # eval 阶段也使用同一个 seed，保证可复现；如需单独 eval seed，可在 config 中扩展
    set_seed(config.get('seed', 0))
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    action_dim = config.get('action_dim', state_dim)
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    dataset_dir = config.get('dataset_dir', None)
    num_episodes = config.get('num_episodes', None)
    init_qpos_from_dataset = config.get('init_qpos_from_dataset', False)
    direct_replay = config.get('direct_replay', False)
    replay_episode = config.get('replay_episode', 0)
    onscreen_cam = 'default_cam' if 'dex' in task_name else 'angle'

    use_pca_action = ('dex' in task_name and action_dim < STATE_DIM_DEX and not direct_replay)

    if direct_replay and real_robot:
        raise ValueError('direct_replay 仅支持 sim 环境，不能用于 real_robot。')
    if direct_replay:
        assert dataset_dir is not None and num_episodes is not None, \
            "direct_replay 需要 dataset_dir 和 num_episodes。"

    # 若未显式指定 output_dir，则默认仍写入 ckpt_dir（兼容旧逻辑）
    if output_dir is None:
        output_dir = ckpt_dir

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    logger(loading_status)
    policy.cuda()
    policy.eval()
    logger(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    pca = None
    if use_pca_action:
        pca_path = os.path.join(ckpt_dir, 'pca.pkl')
        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)
        logger(f'Loaded PCA from {pca_path} for action inverse transform')

    def sample_dataset_start_qpos():
        """
        从数据集中随机采一条轨迹的起始 qpos（t=0）。
        返回原始 qpos（不做归一化），形状为 (qpos_dim,)。
        """
        assert dataset_dir is not None and num_episodes is not None, \
            "dataset_dir / num_episodes 未在 config 中提供，无法从数据集采样初始 qpos。"
        episode_idx = np.random.randint(0, num_episodes)
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos0 = root['/observations/qpos'][0]
        return np.array(qpos0, dtype=np.float32)

    def load_episode_for_replay(episode_idx):
        """
        加载指定 episode 的 qpos[0] 和整条 action 序列，用于 direct replay。
        返回 (qpos0, actions)，actions 形状 (T, action_dim)，为原始未归一化。
        """
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos0 = np.array(root['/observations/qpos'][0], dtype=np.float32)
            actions = np.array(root['/action'][()], dtype=np.float64)
        return qpos0, actions

    def overwrite_sim_qpos_from_dataset(env, task_name, dataset_qpos):
        """
        将模拟环境的初始关节位置，用数据集中某条轨迹起点的 qpos 覆盖。
        只对 sim 环境生效，对 real_robot 不做任何事。
        """
        physics = env._physics

        # bi-manual 14-dim 任务（transfer_cube / insertion）
        if 'sim_transfer_cube' in task_name or 'sim_insertion' in task_name:
            # 期望的观测维度是 14: [L_arm(6), L_grip(1), R_arm(6), R_grip(1)]
            q = np.asarray(dataset_qpos, dtype=np.float32)
            if q.shape[0] < 14:
                raise ValueError(f"数据集 qpos 维度 {q.shape[0]} < 14，无法映射到模拟关节。")
            q = q[:14]
            left_arm = q[:6]
            left_grip_norm = q[6]
            right_arm = q[7:13]
            right_grip_norm = q[13]

            # 归一化位置 -> 物理手指位置
            left_grip_pos = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(float(left_grip_norm))
            right_grip_pos = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(float(right_grip_norm))

            with physics.reset_context():
                # 手臂关节
                physics.data.qpos[:6] = left_arm
                physics.data.qpos[8:14] = right_arm
                # 夹爪：两指对称，一个正一个负（与 before_step 中 env_action 一致）
                physics.data.qpos[6] = left_grip_pos
                physics.data.qpos[7] = -left_grip_pos
                physics.data.qpos[14] = right_grip_pos
                physics.data.qpos[15] = -right_grip_pos
                # 保持其他 qpos（如 BOX_POSE）不变，只更新控制量为当前关节
                np.copyto(physics.data.ctrl, physics.data.qpos[:16])

        # dexterous hand 任务：只用手部 22 维 qpos 初始化
        elif 'dex' in task_name:
            q = np.asarray(dataset_qpos, dtype=np.float32)
            if q.shape[0] < 22:
                raise ValueError(f"dex 任务期望至少 22 维手部 qpos，但数据集给了 {q.shape[0]} 维。")
            hand_qpos = q[:22]
            with physics.reset_context():
                physics.data.qpos[:22] = hand_qpos
                # 控制量跟随当前手部状态
                np.copyto(physics.data.ctrl, physics.data.qpos[:22])

        else:
            # 未知的 sim 任务，不做覆盖
            return

    max_timesteps = int(max_timesteps * 5) # may increase for real-world tasks

    # load environment
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers # requires aloha
        from aloha_scripts.real_env import make_real_env # requires aloha
        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        from sim_env import make_sim_env
        env = make_sim_env(task_name, time_limit=max_timesteps*DT)
        env_max_reward = env.task.max_reward

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    num_rollouts = 50 if not direct_replay else min(50, num_episodes)
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        ### direct replay: 本 rollout 要回放的 episode
        replay_qpos0, replay_actions = None, None
        if direct_replay:
            episode_idx = (replay_episode + rollout_id) % num_episodes
            replay_qpos0, replay_actions = load_episode_for_replay(episode_idx)
            logger(f'[direct_replay] Rollout {rollout_id} replay episode_{episode_idx}, T={len(replay_actions)}')

        ### set task
        if 'sim_transfer_cube' in task_name:
            BOX_POSE[0] = sample_box_pose() # used in sim reset
        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset
        elif 'dex' in task_name:
            DEX_OBJECT_POSE[0] = sample_dex_object_pose()

        ts = env.reset()

        # 将模拟初始 qpos 覆盖为数据集中的某条轨迹起点（init_qpos_from_dataset 随机一条；direct_replay 用当前回放的那条）
        if not real_robot and (init_qpos_from_dataset or direct_replay):
            try:
                dataset_qpos0 = replay_qpos0 if direct_replay else sample_dataset_start_qpos()
                overwrite_sim_qpos_from_dataset(env, task_name, dataset_qpos0)
                new_obs = env._task.get_observation(env._physics)
                ts = ts._replace(observation=new_obs)
                logger("Initialized sim qpos from dataset start qpos.")
            except Exception as ex:
                logger(f"[WARN] 初始化 qpos 从数据集失败，使用默认初始化. error={ex}")

        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        ### evaluation loop
        steps_this_rollout = min(max_timesteps, replay_actions.shape[0]) if direct_replay else max_timesteps
        if temporal_agg and not direct_replay:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, action_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()

        # 是否为本 rollout 保存视频/图片
        will_save_this_rollout = save_episode and (
            max_save_episodes is None or rollout_id < max_save_episodes
        )
        image_list = [] if will_save_this_rollout else None  # for visualization
        qpos_list = [] if will_save_this_rollout else None
        target_qpos_list = [] if will_save_this_rollout else None
        rewards = []
        # 每个 rollout 内部的 step 进度条
        with torch.inference_mode():
            for t in tqdm(range(steps_this_rollout), desc=f"Rollout {rollout_id} steps", leave=False):
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
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
                    curr_image = get_image(ts, camera_names)

                ### 动作来源：direct_replay 用数据集里的 action[t]，否则用 policy 输出
                if direct_replay:
                    target_qpos = np.asarray(replay_actions[t], dtype=np.float64)
                else:
                    ### query policy
                    if config['policy_class'] == "ACT":
                        if t % query_frequency == 0:
                            all_actions = policy(qpos, curr_image)
                        if temporal_agg:
                            all_time_actions[[t], t:t+num_queries] = all_actions
                            actions_for_curr_step = all_time_actions[:, t]
                            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                            actions_for_curr_step = actions_for_curr_step[actions_populated]
                            k = 0.01
                            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                            exp_weights = exp_weights / exp_weights.sum()
                            exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                            raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                        else:
                            raw_action = all_actions[:, t % query_frequency]
                    elif config['policy_class'] == "CNNMLP":
                        raw_action = policy(qpos, curr_image)
                    else:
                        raise NotImplementedError
                    raw_action = raw_action.squeeze(0).cpu().numpy()
                    target_qpos = post_process(raw_action)
                    if use_pca_action and pca is not None:
                        root_6 = target_qpos[:ROOT_DIM]
                        finger_pcs = target_qpos[ROOT_DIM:].reshape(1, -1)
                        finger_16 = pca.inverse_transform(finger_pcs).squeeze(0)
                        target_qpos = np.concatenate([root_6, finger_16]).astype(np.float64)

                if will_save_this_rollout:
                    target_qpos_list.append(target_qpos)
                ts = env.step(target_qpos)
                rewards.append(ts.reward)

            plt.close()
        if real_robot:
            move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
            pass

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards[rewards!=None])
        highest_rewards.append(episode_highest_reward)
        logger(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

        if will_save_this_rollout:
            save_videos(image_list, DT, video_path=os.path.join(output_dir, f'video{rollout_id}.mp4'))
            # 记录并可视化关节角度：state vs command
            plot_path = os.path.join(output_dir, f'video{rollout_id}_qpos.png')
            visualize_joints(qpos_list, target_qpos_list, plot_path=plot_path,
                             label_overwrite=('State', 'Command'))

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
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


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
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
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
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
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if epoch % 500 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

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
                        help='在 eval（模拟环境）时，用数据集中随机一条轨迹的起始 qpos 覆盖模拟环境初始姿态')
    parser.add_argument('--direct_replay', action='store_true',
                        help='eval 时用数据集中某条轨迹的 action 序列直接控制 sim，不跑 policy；会同时用该条轨迹的 qpos[0] 初始化')
    parser.add_argument('--replay_episode', type=int, default=0,
                        help='direct_replay 时从第几条轨迹开始回放（rollout i 回放 episode replay_episode+i）')
    parser.add_argument('--temporal_agg', action='store_true',
                        help='eval 时使用 temporal aggregation 平滑 ACT 输出')
    parser.add_argument('--max_save_episodes', type=int, default=None,
                        help='eval 时只保存前 N 条 rollout 的视频和 png，后续不再保存以节省时间；默认保存全部')
    parser.add_argument('--ckpt_dir', type=str, required=True,
                        help='checkpoint dir (required; provided via CLI only)')
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
    args_dict['ckpt_dir'] = _EVAL_ARGS.ckpt_dir
    args_dict['max_save_episodes'] = _EVAL_ARGS.max_save_episodes
    train_or_eval(args_dict, hydra_cfg=cfg)


if __name__ == '__main__':
    _EVAL_ARGS, unknown = _parse_eval_args()
    sys.argv = [sys.argv[0]] + unknown
    main()
