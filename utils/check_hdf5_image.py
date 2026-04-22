import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

# 改成你自己的 hdf5 路径
HDF5_PATH = "data/sim_dexgrasp_cube_teleop/20260304_123721/episode_0.hdf5"
DATASET_KEY = "observations/images/top"


def parse_args():
    parser = argparse.ArgumentParser(
        description="检查 HDF5 文件：默认遍历并打印所有数据的维度；可选为指定数据集添加 HDF5 IMAGE 标签。"
    )
    parser.add_argument(
        "--hdf5-path",
        type=str,
        default=HDF5_PATH,
        help=f"HDF5 文件路径（默认: {HDF5_PATH}）",
    )
    parser.add_argument(
        "--dataset-key",
        type=str,
        default=DATASET_KEY,
        help=f"图像数据集的 key，仅在使用 --add-image-tag 时有效（默认: {DATASET_KEY}）",
    )
    parser.add_argument(
        "--add-image-tag",
        action="store_true",
        help="额外功能：复制 HDF5 并为指定图像数据集添加 HDF5 IMAGE 标签（供 H5Web 识别）",
    )
    return parser.parse_args()


def print_all_shapes(f, prefix=""):
    """递归遍历 HDF5 中所有数据集，打印 key 与维度。"""
    for key in f.keys():
        path = f"{prefix}/{key}" if prefix else key
        obj = f[key]
        if isinstance(obj, h5py.Dataset):
            print(f"  {path}: shape={obj.shape}, dtype={obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print_all_shapes(obj, path)


def run_add_image_tag(src_path: Path, dataset_key: str):
    """复制 HDF5 并为指定数据集添加 IMAGE 标签。"""
    dst_path = src_path.with_name(
        src_path.stem + "_with_image_attr" + src_path.suffix
    )
    shutil.copy2(src_path, dst_path)
    print(f"已复制 HDF5 文件到新路径: {dst_path}")

    with h5py.File(dst_path, "r+") as f:
        dset = f[dataset_key]
        print(f"dataset key: {dataset_key}")
        print(f"shape: {dset.shape}")
        print(f"dtype: {dset.dtype}")

        current_class = dset.attrs.get("CLASS", None)
        if current_class is None or (
            isinstance(current_class, (bytes, str))
            and current_class != b"IMAGE"
            and current_class != "IMAGE"
        ):
            dset.attrs["CLASS"] = np.string_("IMAGE")
            print('已为数据集添加属性 CLASS="IMAGE"')
        else:
            print(f'数据集已包含 CLASS 属性: {current_class!r}')

        if dset.ndim >= 4:
            sample = dset[0]
        else:
            sample = dset[()]

        sample = np.array(sample)
        print(f"sample shape: {sample.shape}")
        print(f"min: {sample.min()}, max: {sample.max()}")

        if (
            sample.dtype == np.uint8
            and sample.ndim == 3
            and sample.shape[-1] in (1, 3, 4)
        ):
            print("看起来是标准的 uint8 图像格式 (H, W, C)。")
        elif (
            sample.dtype in (np.float32, np.float64)
            and sample.min() >= 0
            and sample.max() <= 1
        ):
            print("看起来是 0~1 浮点图像，可以乘 255 转成 uint8 图像。")
        else:
            print("数据格式不像典型的标准图像，需要根据 shape/dtype 进一步判断。")

        img = sample
        if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[0] < img.shape[-1]:
            img = np.transpose(img, (1, 2, 0))

        if img.dtype != np.uint8:
            img = img.astype(np.float32)
            img = img - img.min()
            if img.max() > 0:
                img = img / img.max()
            img = (img * 255).astype(np.uint8)

        out_path = dst_path.with_suffix(".top_sample.png")
        Image.fromarray(img).save(out_path)
        print(f"已保存示例图像到: {out_path}")


def main():
    args = parse_args()
    src_path = Path(args.hdf5_path)

    # 默认：遍历所有数据并打印维度
    print(f"HDF5 文件: {src_path}")
    print("所有数据集及其维度:")
    with h5py.File(src_path, "r") as f:
        print_all_shapes(f)

    # 可选：添加 image 标签
    if args.add_image_tag:
        print("\n--- 执行额外功能：添加 image 标签 ---")
        run_add_image_tag(src_path, args.dataset_key)


if __name__ == "__main__":
    main()
