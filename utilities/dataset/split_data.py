from pathlib import Path
import numpy as np
import os
from typing import List, Tuple, Optional, Any
import tifffile
from numpy.typing import NDArray

def load_images(gt_path: Path, orig_path: Path) -> Tuple[NDArray, NDArray, List[str]]:
    gt_images = []
    orig_images = []
    times = []
    for gt_file in gt_path.iterdir():
        gt_img = tifffile.imread(gt_path / gt_file)
        gt_images.append(gt_img)

        # Extract id (=time)
        id = gt_file.name[4:-4]
        try:
            int(id)
        except ValueError:
            raise ValueError(f"Could not extract a valid integer id from filename '{gt_file.name}'. Expected format: 'mask<id>.tif'")
            
        # Find the corresponding original file
        orig_img = tifffile.imread(orig_path / f"t{id}.tif")
        orig_images.append(orig_img)

        times.append(int(id))  # left leading 0 are removed
    gt = np.array(gt_images)
    imgs = np.array(orig_images)

    if gt.shape != imgs.shape:
        raise RuntimeError(f"GT shape '{gt.shape}' is incompatible with original images '{imgs.shape}'!")

    return gt, imgs, times


def bbox_3d(img: NDArray) -> List[Tuple[int, int]]:
    # idea is by https://stackoverflow.com/a/31402351
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.nonzero(r)[0][[0, -1]]
    cmin, cmax = np.nonzero(c)[0][[0, -1]]
    zmin, zmax = np.nonzero(z)[0][[0, -1]]
    return [(rmin, rmax), (cmin, cmax), (zmin, zmax)]


def process(gt: NDArray, imgs: NDArray, time_dim: int = 0) -> List[Tuple[NDArray, NDArray]]:
    extracted_tunnels = []

    # Keep track of largest size
    largest_row_range = 0
    largest_row_range_id = 0,0
    largest_col_range = 0
    largest_col_range_id = 0,0
    largest_z_range = 0
    largest_z_range_id = 0,0


    # Go through each time frame
    for t_idx in range(gt.shape[time_dim]):
        gt_timeslice = gt[t_idx]
        img_timeslice = imgs[t_idx]
        
        # Find unique cell ids
        tunnel_ids = np.unique(gt_timeslice)

        # Find bounding boxes of all tunnels
        for tunnel_id in tunnel_ids:
            rows, cols, zs = bbox_3d(gt_timeslice == tunnel_id)
            extracted_tunnel_gt = gt_timeslice[rows[0]:rows[1]+1, cols[0]:cols[1]+1, zs[0]:zs[1]+1]
            extracted_tunnel_img = img_timeslice[rows[0]:rows[1]+1, cols[0]:cols[1]+1, zs[0]:zs[1]+1]
            extracted_tunnels.append(
                (extracted_tunnel_gt, extracted_tunnel_img)
            )

            if tunnel_id == 0:
                continue  # Skip BG

            if largest_row_range < rows[1]-rows[0]+1:
                largest_row_range = rows[1]-rows[0]+1
                largest_row_range_id = tunnel_id, t_idx
            if largest_col_range < cols[1]-cols[0]+1:
                largest_col_range = cols[1]-cols[0]+1
                largest_col_range_id = tunnel_id, t_idx
            if largest_z_range < zs[1]-zs[0]+1:
                largest_z_range = zs[1]-zs[0]+1
                largest_z_range_id = tunnel_id, t_idx
        
    print(f"Largest row range {largest_row_range} found for tunnel {largest_row_range_id[0]} at time slot {largest_row_range_id[1]}")
    print(f"Largest col range {largest_col_range} found for tunnel {largest_col_range_id[0]} at time slot {largest_col_range_id[1]}")
    print(f"Largest z range {largest_z_range} found for tunnel {largest_z_range_id[0]} at time slot {largest_z_range_id[1]}")
    return extracted_tunnels


def main(input_folder: str, output_folder: str, overwrite_output_flag: bool = False) -> None:
    input_folder_path = Path(input_folder)
    output_folder_path = Path(output_folder)

    os.makedirs(output_folder_path, exist_ok=overwrite_output_flag)
    os.makedirs(output_folder_path / "GT", exist_ok=overwrite_output_flag)
    os.makedirs(output_folder_path / "IMG", exist_ok=overwrite_output_flag)
    os.makedirs(output_folder_path / "GT_MERGED_LABELS", exist_ok=overwrite_output_flag)

    gt_path = input_folder_path / "01_GT" / "SEG"
    orig_path = input_folder_path / "01"
    if not gt_path.exists():
        raise RuntimeError(f"Path to GT '{str(gt_path)}' does not exists!")
    if not orig_path.exists():
        raise RuntimeError(f"Path to orig. '{str(orig_path)}' does not exists!")

    # Load
    gt, imgs, times = load_images(gt_path, orig_path)
    print(f"Total unique values across all channels: {sum(np.unique(gt[i]).size for i in range(gt.shape[0]))}")
    print(f"Found {times} as time slots with annotations given")

    # Extract the tunnels
    extracted_tunnels = process(gt, imgs, 0)

    # Save them in the output folder
    for i, (tunnel_gt, tunnel_img) in enumerate(extracted_tunnels):
        tifffile.imwrite(output_folder_path / "GT" / f"{i}.tif", tunnel_gt)
        tifffile.imwrite(output_folder_path / "IMG" / f"{i}.tif", tunnel_img)
        tifffile.imwrite(output_folder_path / "GT_MERGED_LABELS" / f"{i}.tif", (tunnel_gt != 0).astype(np.bool))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Split dataset images with padding.")
    parser.add_argument("input_folder", type=Path, help="Path to the folder containing images.")
    parser.add_argument("output_folder", type=Path, help="Path to the output folder.")
    parser.add_argument('--output_overwrite', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    main(args.input_folder, args.output_folder, args.output_overwrite)
