"""Core sim rollout: batch-step envs under a given FiLM theta and collect episode_return (plus
optional video/frame/mocap-trajectory capture)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from constants import DT, DEFAULT_STATE_DIM
from policy import ACTPolicy
from visualize_episodes import save_videos

from src.cli_utils import _progress
from src.film_utils import _film_pca_attr, _split_film_theta_batch


def rollout_batch_episode_returns(
    policy: ACTPolicy,
    envs: list,
    eval_cfg: dict,
    pre_process,
    post_process,
    thetas: np.ndarray,
    *,
    latent_z,
    fixed_object_pose: np.ndarray,
    fixed_object_shape: str | None = None,
    fixed_object_size: np.ndarray | None = None,
    fixed_init_qpos: np.ndarray | None,
    init_qpos_from_dataset: bool,
    dataset_dir: str | None,
    num_episodes: int | None,
    show_rollout_progress: bool = False,
    rollout_desc: str = "rollout",
    video_dir: Path | None = None,
    round_label: str = "round",
    capture_frames: bool = False,
    capture_mocap_pos: bool = False,
    video_layout: str = "combined",
    film_target: str = "visual",
):
    """
    Step multiple envs in parallel (one theta each), batch observations per timestep on one GPU.
    Returns episode_return per trajectory (shape: (B,)).

    film_target: which PCA-bottleneck FiLM insertion point `thetas` searches — "visual"
    (default), "memory" (candidate 4, encoder-decoder boundary), or "hs" (candidate 5,
    pre-action_head). Must match whatever load_film_pca(..., target=film_target) installed on
    `policy.model` beforehand; see _film_pca_attr().

    If video_dir is not None, records video(s) per env to video_dir / f"{round_label}_env{i}.mp4".
    video_layout="combined" (default) writes one side-by-side mp4; "separate" writes one mp4
    per camera (e.g. round_0001_env0_default_cam.mp4).

    If capture_frames is True, also collects each env's per-timestep observation images and
    returns them: image_lists[i] is a list of {cam_name: HxWxC uint8} dicts, one per timestep,
    for env i (same format save_videos expects). Used e.g. to build VLM visual feedback without
    needing to encode/decode an mp4.

    If capture_mocap_pos is True, also collects each env's commanded end-effector (mocap) xyz
    position per timestep as an (B, T, 3) array. Only meaningful for HMF-proto5-style mocap
    tasks, where action[:3] IS the mocap_pos written verbatim into physics.data.mocap_pos (see
    sim_env.py's Proto5HMFMocapTask.before_step()) — for other task families action[:3] is not
    an end-effector position.

    capture_frames and capture_mocap_pos are independent; when either is set, the return value
    is (rewards_sum, *extras) with extras in that order (image_lists, then mocap_pos), otherwise
    just rewards_sum.
    """
    B = len(envs)
    if thetas.shape[0] != B:
        raise ValueError(f"thetas batch {thetas.shape[0]} != envs {B}")

    k_attr, g_attr, b_attr = (_film_pca_attr(film_target, n) for n in ("k", "gamma", "beta"))
    k = int(getattr(policy.model, k_attr))
    film_gamma_t, film_beta_t = _split_film_theta_batch(thetas, k)
    film_theta_kwargs = {g_attr: film_gamma_t, b_attr: film_beta_t}

    # reset all envs with fixed object pose
    from imitate_episodes import apply_object_pose_for_reset, overwrite_sim_qpos_from_dataset, sample_dataset_start_qpos

    ts_list = []
    for i, env in enumerate(envs):
        apply_object_pose_for_reset(
            eval_cfg["task_name"], fixed_object_pose,
            fixed_object_shape=fixed_object_shape, fixed_object_size=fixed_object_size,
        )
        ts = env.reset()
        if fixed_init_qpos is not None:
            overwrite_sim_qpos_from_dataset(env, eval_cfg["task_name"], fixed_init_qpos)
            new_obs = env._task.get_observation(env._physics)
            ts = ts._replace(observation=new_obs)
        elif init_qpos_from_dataset:
            if dataset_dir is None or num_episodes is None:
                raise ValueError("init_qpos_from_dataset requires dataset_dir/num_episodes")
            qpos0 = sample_dataset_start_qpos(dataset_dir, num_episodes)
            overwrite_sim_qpos_from_dataset(env, eval_cfg["task_name"], qpos0)
            new_obs = env._task.get_observation(env._physics)
            ts = ts._replace(observation=new_obs)
        ts_list.append(ts)

    max_timesteps = int(eval_cfg["episode_len"] * 2)
    state_dim = int(eval_cfg["state_dim"])
    camera_names = eval_cfg["camera_names"]
    policy_config = eval_cfg["policy_config"]
    query_frequency = int(policy_config["num_queries"])
    temporal_agg = bool(eval_cfg["temporal_agg"])
    action_dim = int(eval_cfg.get("action_dim", state_dim))
    if temporal_agg:
        query_frequency = 1
        num_queries = int(policy_config["num_queries"])
        all_time_actions = torch.zeros([B, max_timesteps, max_timesteps + num_queries, action_dim], device="cuda")
    else:
        all_time_actions = None

    rewards_sum = np.zeros(B, dtype=np.float64)
    record_video = video_dir is not None
    collect_images = record_video or capture_frames
    image_lists: list[list[dict]] | None = [[] for _ in range(B)] if collect_images else None
    mocap_pos_lists: list[list[np.ndarray]] | None = [[] for _ in range(B)] if capture_mocap_pos else None

    # Same as imitate_episodes eval: inference_mode so temporal_agg does not keep
    # autograd over the full episode when stitching multi-step policy outputs into all_time_actions
    with torch.inference_mode():
        it = _progress(
            range(max_timesteps),
            enabled=bool(show_rollout_progress),
            desc=str(rollout_desc),
            total=max_timesteps,
        )
        for t in it:
            # build batch obs
            qpos_batch = []
            img_batch = []
            for i, ts in enumerate(ts_list):
                obs = ts.observation
                qpos_np = np.asarray(obs["qpos"], dtype=np.float32)[:state_dim]
                qpos_batch.append(pre_process(qpos_np))

                if collect_images:
                    image_lists[i].append(obs["images"])

                # images: (num_cam, C, H, W) normalized to 0..1
                cams = []
                for cam_name in camera_names:
                    im = obs["images"][cam_name]  # H,W,C uint8
                    im = np.transpose(im, (2, 0, 1)).astype(np.float32) / 255.0
                    cams.append(im)
                img_batch.append(np.stack(cams, axis=0))

            qpos_t = torch.from_numpy(np.stack(qpos_batch, axis=0)).float().cuda()
            img_t = torch.from_numpy(np.stack(img_batch, axis=0)).float().cuda()

            # query policy
            if t % query_frequency == 0:
                all_actions = policy(
                    qpos_t,
                    img_t,
                    latent_z_sample=latent_z,
                    **film_theta_kwargs,
                )

            if temporal_agg:
                assert all_time_actions is not None
                all_time_actions[:, t, t : t + num_queries] = all_actions
                actions_for_curr_step = all_time_actions[:, :, t]  # (B, T, action_dim)
                # populated mask
                populated = torch.all(actions_for_curr_step != 0, dim=2)  # (B, T)
                raw_action_list = []
                for i in range(B):
                    acts_i = actions_for_curr_step[i][populated[i]]
                    if acts_i.numel() == 0:
                        raw_action_list.append(torch.zeros((1, action_dim), device="cuda"))
                        continue
                    k = 0.01
                    w = np.exp(-k * np.arange(len(acts_i)))
                    w = w / w.sum()
                    w = torch.from_numpy(w).float().cuda().unsqueeze(1)
                    raw_action_list.append((acts_i * w).sum(dim=0, keepdim=True))
                raw_action = torch.cat(raw_action_list, dim=0)  # (B, action_dim)
            else:
                raw_action = all_actions[:, t % query_frequency]  # (B, action_dim)

            raw_np = raw_action.detach().cpu().numpy()
            target_qpos_batch = post_process(raw_np)  # (B, action_dim)

            if capture_mocap_pos:
                assert mocap_pos_lists is not None
                for i in range(B):
                    mocap_pos_lists[i].append(np.asarray(target_qpos_batch[i, :3], dtype=np.float64))

            # step envs
            new_ts_list = []
            for i, env in enumerate(envs):
                ts = env.step(np.asarray(target_qpos_batch[i], dtype=np.float64))
                r = ts.reward
                if r is not None:
                    rewards_sum[i] += float(r)
                new_ts_list.append(ts)
            ts_list = new_ts_list

    if record_video:
        assert video_dir is not None and image_lists is not None
        video_dir.mkdir(parents=True, exist_ok=True)
        for i in range(B):
            save_videos(
                image_lists[i],
                DT,
                video_path=str(video_dir / f"{round_label}_env{i}.mp4"),
                layout=video_layout,
            )

    extras = []
    if capture_frames:
        assert image_lists is not None
        extras.append(image_lists)
    if capture_mocap_pos:
        assert mocap_pos_lists is not None
        extras.append(np.stack([np.stack(lst, axis=0) for lst in mocap_pos_lists], axis=0))
    if extras:
        return (rewards_sum, *extras)
    return rewards_sum


def _build_eval_config(
    task_name: str,
    task_cfg: dict,
    policy_class: str,
    policy_config: dict,
    seed: int,
    temporal_agg: bool,
    latent_z_sample,
    fixed_object_pose: np.ndarray | None,
    fixed_init_qpos: np.ndarray | None,
    init_qpos_from_dataset: bool,
):
    state_dim = task_cfg.get("state_dim", DEFAULT_STATE_DIM)
    action_dim = task_cfg.get("action_dim", state_dim)
    return {
        "real_robot": False,
        "policy_class": policy_class,
        "policy_config": policy_config,
        "task_name": task_name,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "episode_len": task_cfg["episode_len"],
        "camera_names": task_cfg["camera_names"],
        "temporal_agg": temporal_agg,
        "seed": seed,
        "latent_z_sample": None,
        "dataset_dir": task_cfg.get("dataset_dir"),
        "num_episodes": task_cfg.get("num_episodes"),
        "init_qpos_from_dataset": init_qpos_from_dataset,
        "direct_replay": False,
        "fixed_object_pose": fixed_object_pose,
        "fixed_init_qpos": fixed_init_qpos,
    }
