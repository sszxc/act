import numpy as np
import torch
import os
import cv2
import h5py
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA

from constants import ROOT_DIM, FINGER_DIM
import forward_kinematics

import IPython
e = IPython.embed

def load_cam_images(root, camera_names, start_ts, image_size=None):
    """Read one frame per camera and stack to (k, h, w, 3). image_size=(H, W) resizes every
    camera to a common size; required when cameras differ in native resolution."""
    images = []
    for cam_name in camera_names:
        img = root[f'/observations/images/{cam_name}'][start_ts]
        if image_size is not None:
            img = cv2.resize(img, (int(image_size[1]), int(image_size[0])), interpolation=cv2.INTER_AREA)
        images.append(img)
    if len({im.shape for im in images}) > 1:
        raise ValueError(
            f'cameras have different resolutions: '
            f'{dict(zip(camera_names, (im.shape for im in images)))}. '
            f'Pass image_size=[H,W] to resize them to a common size.')
    return np.stack(images, axis=0)


def _task_space_keep_idx(joint_ids):
    """action_repr='task_space' + joint_ids (arm-only): the 6 arm joints are always fused into
    the palm pose, so joint_ids only selects which of the trailing 18 wrist/finger dims survive.
    Returns the column indices into the 27-dim task-space vector (pos3+rot6d6+hand18) to keep."""
    joint_ids = np.asarray(joint_ids, dtype=int)
    hand_keep = [j - forward_kinematics.N_ARM for j in joint_ids if j >= forward_kinematics.N_ARM]
    return np.array(list(range(forward_kinematics.POSE_DIM)) +
                     [forward_kinematics.POSE_DIM + h for h in hand_keep], dtype=int)


class EpisodicDataset(torch.utils.data.Dataset):
    """action_repr:
      'absolute'   - target is the raw action, normalized by action_mean/std (original ACT).
      'delta'      - target is action - qpos[start_ts], normalized by delta_mean/std. Removes the
                     copy-qpos shortcut, which this dataset invites because action[t] == qpos[t+1].
      'task_space' - task-space output (see forward_kinematics.py): state is qpos[start_ts] run
                     through FK (palm pos(3) + rot6d(6) + the 18 remaining joints unchanged, 27-dim
                     total); target is the FK'd delta from that state (same start-relative semantics
                     as 'delta', but the pose part is a proper SE(3) delta, not a raw subtraction).
    action_offset: chunk starts at action[start_ts + offset]. -1 is the upstream ALOHA
    'timestep alignment' hack; with action == qpos shifted by one it makes the first chunk
    element exactly qpos[start_ts] (a guaranteed no-op step), so 0 is the honest choice here.
    action_stride: chunk element j targets action[start + offset + j*stride], i.e. the policy
    commands at 30/stride Hz. At stride 1 the per-step motion (0.003 rad) is at the error
    floor; stride 3 makes each commanded step 3x larger without changing the wall-clock span
    a chunk covers (chunk_size 10 x stride 3 = the same 1.0 s as chunk_size 30 x stride 1).
    """

    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, num_queries, image_size=None,
                 action_repr='absolute', action_offset=-1, joint_ids=None, action_stride=1):
        super(EpisodicDataset).__init__()
        self.action_stride = int(action_stride)
        self.joint_ids = None if joint_ids is None else np.asarray(joint_ids, dtype=int)
        if self.joint_ids is not None and action_repr == 'task_space':
            # FK fuses all N_ARM arm joints into the palm pose, so none of them can be dropped
            # individually; joint_ids may only additionally restrict which hand/wrist dims survive.
            if not set(range(forward_kinematics.N_ARM)).issubset(self.joint_ids.tolist()):
                raise ValueError(
                    f"action_repr='task_space' fuses joints 0..{forward_kinematics.N_ARM - 1} into "
                    f"the palm pose; joint_ids must include all of them (got {joint_ids})")
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.num_queries = int(num_queries)
        self.image_size = image_size
        self.action_repr = action_repr
        self.action_offset = int(action_offset)
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        return self.load(self.episode_ids[index], start_ts=None)

    def load(self, episode_id, start_ts=None):
        """start_ts=None draws a random timestep (training); an int pins it, which is what the
        fixed deployment-metric set in imitate_episodes.py uses."""
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape  # (T, action_dim)
            episode_len = int(original_action_shape[0])
            action_dim = int(original_action_shape[1])
            if start_ts is None:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            all_cam_images = load_cam_images(root, self.camera_names, start_ts, self.image_size)
            # get a fixed-length action chunk starting near start_ts
            if is_sim:
                action_start_ts = start_ts
            else:
                action_start_ts = max(0, start_ts + self.action_offset)
            if self.action_stride == 1:
                action_end_ts = min(episode_len, action_start_ts + self.num_queries)
                action = root['/action'][action_start_ts:action_end_ts]
                action_len = action_end_ts - action_start_ts
            else:
                stop = min(episode_len, action_start_ts + self.num_queries * self.action_stride)
                action = root['/action'][action_start_ts:stop:self.action_stride]
                action_len = len(action)

        self.is_sim = is_sim
        # task_space needs the full 24-dim qpos/action for FK; joint_ids (arm-only) is instead
        # applied after FK, to the trailing hand/wrist dims of the 27-dim task-space vector.
        if self.joint_ids is not None and self.action_repr != 'task_space':
            qpos = qpos[self.joint_ids]
            action = action[:, self.joint_ids]
            action_dim = len(self.joint_ids)
        if self.action_repr == 'delta':
            action = action - qpos[None, :]
        elif self.action_repr == 'task_space':
            action = forward_kinematics.task_action_delta_batch(qpos, action)
            if self.joint_ids is not None:
                action = action[:, _task_space_keep_idx(self.joint_ids)]
            action_dim = action.shape[1]  # 27 (or fewer, arm-only), replaces the raw hdf5 action_dim (24)
        padded_action = np.zeros((self.num_queries, action_dim), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(self.num_queries, dtype=np.float32)
        is_pad[action_len:] = 1

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        if self.action_repr == 'task_space':
            state = forward_kinematics.task_state(qpos)
            if self.joint_ids is not None:
                state = state[_task_space_keep_idx(self.joint_ids)]
        else:
            state = qpos
        qpos_data = torch.from_numpy(state).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float32 for model
        image_data = (image_data / 255.0).float()
        key = self.action_repr if self.action_repr in ('delta', 'task_space') else 'action'
        action_data = (action_data - self.norm_stats[f"{key}_mean"]) / self.norm_stats[f"{key}_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        action_data = action_data.float()
        qpos_data = qpos_data.float()

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes, num_queries=None, action_offset=-1, action_repr='absolute',
                   joint_ids=None, action_stride=1):
    all_qpos_data = []
    all_action_data = []
    per_ep = []
    # task_space needs the full 24-dim qpos/action for FK (see EpisodicDataset.load); joint_ids
    # is instead applied after FK, below, to the trailing hand/wrist dims of the 27-dim vector.
    keep = _task_space_keep_idx(joint_ids) if (joint_ids is not None and action_repr == 'task_space') else None
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        if joint_ids is not None and action_repr != 'task_space':
            qpos, action = qpos[:, joint_ids], action[:, joint_ids]
        # qpos_mean/std must match what EpisodicDataset actually feeds the model as state.
        qpos_for_stats = forward_kinematics.task_state_batch(qpos) if action_repr == 'task_space' else qpos
        if keep is not None:
            qpos_for_stats = qpos_for_stats[:, keep]
        all_qpos_data.append(torch.from_numpy(qpos_for_stats))
        all_action_data.append(torch.from_numpy(action))
        per_ep.append((qpos, action))
    # Allow variable episode lengths by concatenating along time dimension.
    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)

    # normalize action data
    action_mean = all_action_data.mean(dim=0, keepdim=True)
    action_std = all_action_data.std(dim=0, keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=0, keepdim=True)
    qpos_std = all_qpos_data.std(dim=0, keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    if num_queries is not None:
        if action_repr == 'task_space':
            ts_stats = _task_space_stats(per_ep, int(num_queries), int(action_offset))
            if keep is not None:
                ts_stats = {k: v[keep] for k, v in ts_stats.items()}
            stats.update(ts_stats)
        else:
            stats.update(_delta_stats(per_ep, int(num_queries), int(action_offset),
                                      int(action_stride)))
    return stats


def _delta_stats(per_ep, num_queries, action_offset, action_stride=1):
    """Stats for action_repr='delta': over every (start_ts, j) the sampler can draw,
    delta_j = action[start_ts + action_offset + j] - qpos[start_ts]. Scale is ~50x smaller than
    the absolute action's, which is the whole point of the representation."""
    s = np.zeros(per_ep[0][0].shape[1]); s2 = np.zeros_like(s); n = 0
    for qpos, action in per_ep:
        T = len(action)
        for j in range(num_queries):
            off = action_offset + j * action_stride
            lo = max(0, -off)                         # start_ts values with a valid target
            hi = min(T, T - off)
            if hi <= lo:
                continue
            d = action[lo + off:hi + off] - qpos[lo:hi]
            s += d.sum(0); s2 += (d ** 2).sum(0); n += len(d)
    mean = s / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 0.0))
    return {"delta_mean": mean.astype(np.float32),
            "delta_std": np.clip(std, 1e-4, np.inf).astype(np.float32)}


def _task_space_stats(per_ep, num_queries, action_offset):
    """Stats for action_repr='task_space': same (start_ts, j) grid as _delta_stats, but the
    target is the FK'd task-space delta (forward_kinematics.task_action_delta_batch), not a raw
    subtraction. Loops per start_ts (not per j) so each start's FK is computed once and reused
    across its whole queried chunk. This calls mj_forward ~1-2M times on good_41-sized data
    (~60us/call), so expect ~1-2 min the first time norm stats are computed for a given run."""
    s = np.zeros(forward_kinematics.TASKSPACE_DIM); s2 = np.zeros_like(s); n = 0
    for qpos, action in per_ep:
        T = len(action)
        for start_ts in range(T):
            lo = max(0, start_ts + action_offset)
            hi = min(T, start_ts + action_offset + num_queries)
            if hi <= lo:
                continue
            d = forward_kinematics.task_action_delta_batch(qpos[start_ts], action[lo:hi])
            s += d.sum(0); s2 += (d ** 2).sum(0); n += len(d)
    mean = s / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 0.0))
    return {"task_space_mean": mean.astype(np.float32),
            "task_space_std": np.clip(std, 1e-4, np.inf).astype(np.float32)}


def fit_finger_pca(dataset_dir, num_episodes, n_components=3):
    """Collect finger action (indices 6:22) from all episodes and fit PCA."""
    finger_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            action = root['/action'][()]  # (T, 22)
        finger_data.append(action[:, ROOT_DIM:ROOT_DIM + FINGER_DIM])
    finger_data = np.concatenate(finger_data, axis=0).astype(np.float64)
    pca = PCA(n_components=n_components)
    pca.fit(finger_data)
    return pca


def get_norm_stats_with_pca(dataset_dir, num_episodes, pca_finger_dim):
    """Get norm_stats on PCA-transformed action (root 6 + finger PCs) and return PCA object."""
    pca = fit_finger_pca(dataset_dir, num_episodes, n_components=pca_finger_dim)
    all_qpos_data = []
    all_action_transformed = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            action = root['/action'][()]  # (T, 22)
        root_6 = action[:, :ROOT_DIM]
        finger_pcs = pca.transform(action[:, ROOT_DIM:ROOT_DIM + FINGER_DIM])
        action_transformed = np.concatenate([root_6, finger_pcs], axis=1)
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_transformed.append(torch.from_numpy(action_transformed.astype(np.float32)))
    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_transformed = torch.cat(all_action_transformed, dim=0)

    action_mean = all_action_transformed.mean(dim=0, keepdim=True)
    action_std = all_action_transformed.std(dim=0, keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)
    qpos_mean = all_qpos_data.mean(dim=0, keepdim=True)
    qpos_std = all_qpos_data.std(dim=0, keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)

    norm_stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "qpos_mean": qpos_mean.numpy().squeeze(),
        "qpos_std": qpos_std.numpy().squeeze(),
        "example_qpos": all_qpos_data[0].numpy(),
    }
    return norm_stats, pca


class EpisodicDatasetPCA(torch.utils.data.Dataset):
    """Like EpisodicDataset but transforms 22d action to root 6 + finger PCs via PCA."""

    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, num_queries, pca, pca_finger_dim,
                 image_size=None):
        super(EpisodicDatasetPCA).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.num_queries = int(num_queries)
        self.image_size = image_size
        self.pca = pca
        self.pca_finger_dim = pca_finger_dim
        self.action_dim_out = ROOT_DIM + pca_finger_dim
        self.is_sim = None
        self.__getitem__(0)

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False
        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = int(original_action_shape[0])
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            all_cam_images = load_cam_images(root, self.camera_names, start_ts, self.image_size)
            if is_sim:
                action_start_ts = start_ts
            else:
                action_start_ts = max(0, start_ts - 1)
            action_end_ts = min(episode_len, action_start_ts + self.num_queries)
            action = root['/action'][action_start_ts:action_end_ts]  # (action_len, 22)
            action_len = action_end_ts - action_start_ts

        self.is_sim = is_sim
        root_6 = action[:, :ROOT_DIM]
        finger_pcs = self.pca.transform(action[:, ROOT_DIM:ROOT_DIM + FINGER_DIM])
        action_transformed = np.concatenate([root_6, finger_pcs], axis=1).astype(np.float32)
        padded_action = np.zeros((self.num_queries, self.action_dim_out), dtype=np.float32)
        padded_action[:action_len] = action_transformed
        is_pad = np.zeros(self.num_queries, dtype=np.float32)
        is_pad[action_len:] = 1

        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        image_data = torch.einsum('k h w c -> k c h w', image_data)
        image_data = (image_data / 255.0).float()
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        action_data = action_data.float()
        qpos_data = qpos_data.float()

        return image_data, qpos_data, action_data, is_pad


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, num_queries,
              task_name=None, batches_per_epoch=None, image_size=None,
              action_repr='absolute', action_offset=-1, num_workers=1, val_episode_ids=None,
              joint_ids=None, action_stride=1):
    print(f'\nData from: {dataset_dir}\n')
    if val_episode_ids is not None:
        # Explicit held-out set instead of a seeded 80/20 split. Needed whenever runs with
        # different dataset sizes must be compared (the human/scripted mixes): a random split
        # would give each mix a different val set, and their losses would not be comparable.
        val_indices = np.array(sorted(val_episode_ids))
        train_indices = np.setdiff1d(np.arange(num_episodes), val_indices)
    else:
        train_ratio = 0.8
        shuffled_indices = np.random.permutation(num_episodes)
        train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
        val_indices = shuffled_indices[int(train_ratio * num_episodes):]
    print(f'train episodes: {len(train_indices)}, val episodes: {sorted(val_indices.tolist())}')

    if task_name == 'sim_dexgrasp_pca_cube_teleop':
        from constants import SIM_TASK_CONFIGS
        pca_finger_dim = SIM_TASK_CONFIGS[task_name]['pca_finger_dim']
        norm_stats, pca = get_norm_stats_with_pca(dataset_dir, num_episodes, pca_finger_dim)
        stats = {**norm_stats, 'pca': pca}
        train_dataset = EpisodicDatasetPCA(
            train_indices, dataset_dir, camera_names, norm_stats, num_queries=num_queries,
            pca=pca, pca_finger_dim=pca_finger_dim, image_size=image_size)
        val_dataset = EpisodicDatasetPCA(
            val_indices, dataset_dir, camera_names, norm_stats, num_queries=num_queries,
            pca=pca, pca_finger_dim=pca_finger_dim, image_size=image_size)
    else:
        norm_stats = get_norm_stats(dataset_dir, num_episodes, num_queries=num_queries,
                                    action_offset=action_offset, action_repr=action_repr,
                                    joint_ids=joint_ids, action_stride=action_stride)
        stats = norm_stats
        train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats,
                                        num_queries=num_queries, image_size=image_size,
                                        action_repr=action_repr, action_offset=action_offset,
                                        joint_ids=joint_ids, action_stride=action_stride)
        val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats,
                                      num_queries=num_queries, image_size=image_size,
                                      action_repr=action_repr, action_offset=action_offset,
                                      joint_ids=joint_ids, action_stride=action_stride)

    if batches_per_epoch is None:
        # Original behavior: one pass over train_dataset per epoch (each episode sampled once,
        # at a random timestep). batch_size is effectively capped at num train episodes.
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True,
                                       pin_memory=True, num_workers=num_workers, prefetch_factor=2,
                                       persistent_workers=num_workers > 0)
    else:
        # Sample with replacement so batch_size can exceed num train episodes: the same episode
        # can appear more than once per batch/epoch, each time at an independently random
        # timestep (EpisodicDataset.__getitem__ redraws start_ts on every call regardless of
        # index). samples_per_epoch is fixed so batches/epoch stays constant across training.
        samples_per_epoch = batch_size_train * batches_per_epoch
        train_sampler = torch.utils.data.RandomSampler(
            train_dataset, replacement=True, num_samples=samples_per_epoch)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, sampler=train_sampler,
                                       pin_memory=True, num_workers=num_workers, prefetch_factor=2,
                                       persistent_workers=num_workers > 0)
    # Validation always does one plain pass per epoch (no replacement): no reason to inflate it.
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True,
                                num_workers=num_workers, prefetch_factor=2,
                                persistent_workers=num_workers > 0)

    return train_dataloader, val_dataloader, stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
