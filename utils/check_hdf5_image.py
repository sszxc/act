import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

# Set this to your HDF5 path
HDF5_PATH = "data/sim_dexgrasp_cube_teleop/20260304_123721/episode_0.hdf5"
DATASET_KEY = "observations/images/top"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect HDF5: print all dataset shapes by default; optionally add HDF5 IMAGE tag to a dataset."
    )
    parser.add_argument(
        "--hdf5-path",
        type=str,
        default=HDF5_PATH,
        help=f"HDF5 file path (default: {HDF5_PATH})",
    )
    parser.add_argument(
        "--dataset-key",
        type=str,
        default=DATASET_KEY,
        help=f"Image dataset key; only used with --add-image-tag (default: {DATASET_KEY})",
    )
    parser.add_argument(
        "--add-image-tag",
        action="store_true",
        help="Extra: copy HDF5 and add HDF5 IMAGE tag to the image dataset (for H5Web)",
    )
    return parser.parse_args()


def print_all_shapes(f, prefix=""):
    """Recursively walk HDF5 datasets and print keys and shapes."""
    for key in f.keys():
        path = f"{prefix}/{key}" if prefix else key
        obj = f[key]
        if isinstance(obj, h5py.Dataset):
            print(f"  {path}: shape={obj.shape}, dtype={obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print_all_shapes(obj, path)


def run_add_image_tag(src_path: Path, dataset_key: str):
    """Copy HDF5 and add IMAGE tag to the given dataset."""
    dst_path = src_path.with_name(
        src_path.stem + "_with_image_attr" + src_path.suffix
    )
    shutil.copy2(src_path, dst_path)
    print(f"Copied HDF5 to: {dst_path}")

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
            dset.attrs["CLASS"] = np.bytes_("IMAGE")
            print('Added dataset attribute CLASS="IMAGE"')
        else:
            print(f"Dataset already has CLASS attribute: {current_class!r}")

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
            print("Looks like standard uint8 image (H, W, C).")
        elif (
            sample.dtype in (np.float32, np.float64)
            and sample.min() >= 0
            and sample.max() <= 1
        ):
            print("Looks like 0~1 float image; multiply by 255 for uint8.")
        else:
            print("Format does not look like a typical image; check shape/dtype.")

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
        print(f"Saved sample image to: {out_path}")


def main():
    args = parse_args()
    src_path = Path(args.hdf5_path)

    # Default: walk and print shapes
    print(f"HDF5 file: {src_path}")
    print("All datasets and shapes:")
    with h5py.File(src_path, "r") as f:
        print_all_shapes(f)

    # Optional: add image tag
    if args.add_image_tag:
        print("\n--- Extra: add IMAGE tag ---")
        run_add_image_tag(src_path, args.dataset_key)


if __name__ == "__main__":
    main()
