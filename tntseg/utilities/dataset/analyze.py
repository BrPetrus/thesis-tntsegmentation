from dataclasses import dataclass
from numpy.typing import NDArray
from pathlib import Path
from split_data import load_images
from typing import Tuple
import numpy as np


def main(input_folder: Path, output_folder: Path):
    input_folder_path = Path(input_folder)

    gt_path = input_folder_path / "01_GT" / "SEG"
    orig_path = input_folder_path / "01"
    if not gt_path.exists():
        raise RuntimeError(f"Path to GT '{str(gt_path)}' does not exists!")
    if not orig_path.exists():
        raise RuntimeError(f"Path to orig. '{str(orig_path)}' does not exists!")

    # Load
    gt, imgs, times = load_images(gt_path, orig_path)
    print(
        f"Total unique values across all channels: {sum(np.unique(gt[i]).size for i in range(gt.shape[0]))}"
    )
    print(f"Found {times} as time slots with annotations given")

    # Split
    evaluate_overlaps(output_folder, gt)

    # Run dataset info
    print(find_stats(imgs, gt))


@dataclass(frozen=True)
class DatasetStats:
    std: float
    mean: float
    class_frequencies: Tuple[int, int]

    def __str__(self):
        return (
            f"standard devation: {self.std}\n"
            f"mean value: {self.mean}\n"
            f"class frequencies: {self.class_frequencies}"
        )


def find_stats(imgs: NDArray[np.integer], gt: NDArray[np.integer]) -> DatasetStats:
    # if imgs.dtype != np.uint8:
    if not np.issubdtype(imgs.dtype, np.integer):
        raise ValueError(f"Expected integer type but got {imgs.dtype}")
    if not np.issubdtype(gt.dtype, np.integer):
        raise ValueError(f"Expected uint8 type but got {gt.dtype}")

    imgs = imgs.copy()
    imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())

    dataset_std = imgs.std()
    dataset_mean = imgs.mean()

    # Now find the weights for cross-entropy
    frequencies = np.bincount((gt != 0).flatten())  # Merge all labels

    return DatasetStats(
        std=dataset_std, mean=dataset_mean, class_frequencies=frequencies
    )


def evaluate_overlaps(output_folder_path, gt):
    print(gt.shape)
    for t in range(gt.shape[0]):
        other_parts = []
        other_ids = set()
        gt_parts = []
        gt_ids = set()

        other_parts.append(gt[t, :, 0:256, 0:256])
        gt_parts = gt[t, :, 0:256, 256:]
        other_parts.append(gt[t, :, 256:, 0:256])
        other_parts.append(gt[t, :, 256:, 256:])
        other_parts_joined = np.stack(other_parts)

        gt_ids = gt_ids.union(set(np.unique(gt[t, :, 0:256, 256:])))
        other_ids = other_ids.union(set(np.unique(gt[t, :, 0:256, 0:256])))
        other_ids = other_ids.union(set(np.unique(gt[t, :, 256:, 0:256])))
        other_ids = other_ids.union(set(np.unique(gt[t, :, 256:, 256:])))
        print(f"gt_ids: {len(gt_ids)}")
        print(f"other_ids: {len(other_ids)}")

        # Find how many are present on both gt and other_ids
        intersecting = gt_ids.intersection(other_ids)
        print(f"Found {len(intersecting)} intersecting")

        # Evaluate how much do they overlap in terms of areas
        for overlapping_tunnel in intersecting:
            area_in_gt = np.sum(gt_parts[gt_parts == overlapping_tunnel])
            area_in_other = np.sum(
                other_parts_joined[other_parts_joined == overlapping_tunnel]
            )
            print(
                f"Id={overlapping_tunnel} - percent in GT: {area_in_gt / (area_in_gt + area_in_other) * 100:.3}%"
            )
            # Save each part as a TIFF file
            output_dir = output_folder_path / "splits"
            output_dir.mkdir(exist_ok=True)

            # Save GT part
            gt_mask = gt_parts == overlapping_tunnel
            if gt_mask.any():
                tifffile.imwrite(
                    output_dir / f"time_{t}_tunnel_{overlapping_tunnel}_gt.tiff",
                    gt_parts * gt_mask,
                )

            # Save other parts
            other_mask = other_parts_joined == overlapping_tunnel
            if other_mask.any():
                tifffile.imwrite(
                    output_dir / f"time_{t}_tunnel_{overlapping_tunnel}_other.tiff",
                    other_parts_joined * other_mask,
                )


if __name__ == "__main__":
    import argparse
    import tifffile

    parser = argparse.ArgumentParser(description="Analyze dataset")
    parser.add_argument(
        "input_folder", type=Path, help="Path to the folder containing images."
    )
    parser.add_argument(
        "--output_folder",
        type=Path,
        help="Path to the folder containing images.",
        default=Path(".") / "output",
    )
    args = parser.parse_args()

    if not Path(args.input_folder):
        raise RuntimeError(f"Folder {args.input_folder} does not exists.")
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)

    main(Path(args.input_folder), Path(args.output_folder))
