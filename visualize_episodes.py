import os
import numpy as np
import h5py
import argparse
import imageio
from tqdm import tqdm

import matplotlib.pyplot as plt
from constants import DT

import IPython
e = IPython.embed

JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
STATE_NAMES = JOINT_NAMES + ["gripper"]

def load_hdf5(dataset_dir, dataset_name):
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        is_sim = root.attrs['sim']
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        action = root['/action'][()]
        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]

    return qpos, qvel, action, image_dict

def main(args):
    dataset_dir = args['dataset_dir']
    episode_idx = args['episode_idx']
    dataset_name = f'episode_{episode_idx}'

    qpos, qvel, action, image_dict = load_hdf5(dataset_dir, dataset_name)
    save_videos(image_dict, DT, video_path=os.path.join(dataset_dir, dataset_name + '_video.mp4'))
    visualize_joints(qpos, action, plot_path=os.path.join(dataset_dir, dataset_name + '_qpos.png'))
    # visualize_timestamp(t_list, dataset_path) # TODO addn timestamp back


def _save_video(frames, output_path, fps=30, verbose=False):
    """使用 imageio 保存视频，尝试多种编码器以保证可播放"""
    if not output_path.endswith('.mp4'):
        output_path += '.mp4'
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if verbose:
        print(f"使用 imageio 保存视频到: {output_path}")
        print(f"帧数: {len(frames)}, 帧率: {fps}")

    codecs_to_try = ["libx264", "libx264rgb", "mpeg4", "libvpx-vp9"]
    for codec in codecs_to_try:
        try:
            if verbose:
                print(f"尝试使用 {codec} 编码器...")
            with imageio.get_writer(output_path, fps=fps, codec=codec) as writer:
                for frame in tqdm(frames, desc="写入视频帧", unit="帧", disable=not verbose):
                    writer.append_data(frame)
            if verbose:
                print(f"✓ 视频已保存 (使用 {codec} 编码器): {output_path}")
            print(f'Saved video to: {output_path}')
            return
        except Exception as e:
            print(f"✗ {codec} 编码器失败: {e}")
            continue
    raise RuntimeError(f"所有编码器均失败，无法保存视频: {output_path}")


def save_videos(video, dt, video_path=None):
    """将 video（list 或 dict 格式的多相机帧）转为帧列表并用 imageio 保存"""
    if video_path is None:
        video_path = 'output.mp4'
    fps = int(1 / dt) if dt > 0 else 30

    if isinstance(video, list):
        cam_names = list(video[0].keys())
        frames = []
        for image_dict in video:
            images = []
            for cam_name in cam_names:
                image = image_dict[cam_name].copy()
                if image.shape[2] == 3:
                    image = image[:, :, [2, 1, 0]]  # BGR -> RGB
                images.append(image)
            frame = np.concatenate(images, axis=1)
            frames.append(frame)
    elif isinstance(video, dict):
        cam_names = list(video.keys())
        all_cam_videos = np.concatenate([video[cam_name] for cam_name in cam_names], axis=2)
        n_frames = all_cam_videos.shape[0]
        frames = []
        for t in range(n_frames):
            image = all_cam_videos[t].copy()
            if image.shape[2] == 3:
                image = image[:, :, [2, 1, 0]]  # BGR -> RGB
            frames.append(image)
    else:
        raise TypeError("video 必须是 list 或 dict")

    _save_video(frames, video_path, fps=fps, verbose=False)


def visualize_joints(qpos_list, command_list, plot_path=None, ylim=None, label_overwrite=None):
    if label_overwrite:
        label1, label2 = label_overwrite
    else:
        label1, label2 = 'State', 'Command'

    qpos = np.array(qpos_list) # ts, dim
    command = np.array(command_list)
    num_ts, num_dim = qpos.shape
    h, w = 2, num_dim
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))

    # plot joint state
    all_names = [name + '_left' for name in STATE_NAMES] + [name + '_right' for name in STATE_NAMES]
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(qpos[:, dim_idx], label=label1)
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
        ax.legend()

    # plot arm command
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(command[:, dim_idx], label=label2)
        ax.legend()

    if ylim:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved qpos plot to: {plot_path}')
    plt.close()

def visualize_timestamp(t_list, dataset_path):
    plot_path = dataset_path.replace('.pkl', '_timestamp.png')
    h, w = 4, 10
    fig, axs = plt.subplots(2, 1, figsize=(w, h*2))
    # process t_list
    t_float = []
    for secs, nsecs in t_list:
        t_float.append(secs + nsecs * 10E-10)
    t_float = np.array(t_float)

    ax = axs[0]
    ax.plot(np.arange(len(t_float)), t_float)
    ax.set_title(f'Camera frame timestamps')
    ax.set_xlabel('timestep')
    ax.set_ylabel('time (sec)')

    ax = axs[1]
    ax.plot(np.arange(len(t_float)-1), t_float[:-1] - t_float[1:])
    ax.set_title(f'dt')
    ax.set_xlabel('timestep')
    ax.set_ylabel('time (sec)')

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved timestamp plot to: {plot_path}')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', required=False)
    main(vars(parser.parse_args()))
