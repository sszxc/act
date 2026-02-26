import os
import numpy as np
import h5py
import argparse
import imageio
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt
from constants import DT


def _draw_camera_name(image, name):
    """在图像左上角绘制相机名（image 为 RGB numpy (H,W,3) uint8）"""
    if image.size == 0 or image.shape[2] != 3:
        return image
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except Exception:
        font = ImageFont.load_default()
    # 描边效果：先画黑边再画白字，保证在任意背景下可读
    x, y = 10, 10
    for dx, dy in [(-1,-1),(-1,1),(1,-1),(1,1),(-1,0),(1,0),(0,-1),(0,1)]:
        draw.text((x+dx, y+dy), name, font=font, fill=(0, 0, 0))
    draw.text((x, y), name, font=font, fill=(255, 255, 255))
    return np.array(pil)

import IPython
e = IPython.embed

# ALOHA 双臂关节名（仅作参考，绘图不再使用，以支持任意构型数据集）
# JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
# STATE_NAMES = JOINT_NAMES + ["gripper"]

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
    save_videos(
        image_dict,
        DT,
        video_path=os.path.join(dataset_dir, dataset_name + '_video.mp4'),
        input_bgr=args.get('input_bgr', False),
    )
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
                for frame in tqdm(frames, desc="写入视频帧", unit="帧"):
                    writer.append_data(frame)
            if verbose:
                print(f"✓ 视频已保存 (使用 {codec} 编码器): {output_path}")
            print(f'Saved video to: {output_path}')
            return
        except Exception as e:
            print(f"✗ {codec} 编码器失败: {e}")
            continue
    raise RuntimeError(f"所有编码器均失败，无法保存视频: {output_path}")


def save_videos(video, dt, video_path=None, input_bgr=False):
    """将 video（list 或 dict 格式的多相机帧）转为帧列表并用 imageio 保存。
    input_bgr: 若 True 表示 HDF5 里存的是 BGR，会转成 RGB 再写；默认 False 表示已是 RGB，不转换。
    """
    if video_path is None:
        video_path = 'output.mp4'
    fps = int(1 / dt) if dt > 0 else 30

    if isinstance(video, list):
        cam_names = list(video[0].keys())
        frames = []
        for image_dict in tqdm(video, desc="生成视频帧", unit="帧"):
            images = []
            for cam_name in cam_names:
                image = image_dict[cam_name].copy()
                if image.shape[2] == 3 and input_bgr:
                    image = image[:, :, [2, 1, 0]]  # BGR -> RGB
                image = _draw_camera_name(image, cam_name)
                images.append(image)
            frame = np.concatenate(images, axis=1)
            frames.append(frame)
    elif isinstance(video, dict):
        cam_names = list(video.keys())
        n_frames = video[cam_names[0]].shape[0]
        frames = []
        for t in tqdm(range(n_frames), desc="生成视频帧", unit="帧"):
            images = []
            for cam_name in cam_names:
                image = video[cam_name][t].copy()
                if image.shape[2] == 3 and input_bgr:
                    image = image[:, :, [2, 1, 0]]  # BGR -> RGB
                image = _draw_camera_name(image, cam_name)
                images.append(image)
            frame = np.concatenate(images, axis=1)
            frames.append(frame)
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
    num_ts, qpos_dim = qpos.shape
    command_dim = command.shape[1]
    num_dim = max(qpos_dim, command_dim)  # 允许 qpos 与 action 维度不一致
    h, w = 2, num_dim
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))

    # plot joint state（不绑定具体 joint 名称，支持任意机械臂构型）
    # all_names = [name + '_left' for name in STATE_NAMES] + [name + '_right' for name in STATE_NAMES]
    for dim_idx in tqdm(range(num_dim), desc="绘制关节状态", unit="轴"):
        ax = axs[dim_idx]
        if dim_idx < qpos_dim:
            ax.plot(qpos[:, dim_idx], label=label1)
        ax.set_title(f'Joint {dim_idx}')
        ax.legend()

    # plot arm command
    for dim_idx in tqdm(range(num_dim), desc="绘制关节指令", unit="轴"):
        ax = axs[dim_idx]
        if dim_idx < command_dim:
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
    parser.add_argument(
        '--input_bgr',
        action='store_true',
        help='HDF5 中图像为 BGR 时指定，会转为 RGB 再写视频；默认假定已是 RGB。',
    )
    main(vars(parser.parse_args()))
