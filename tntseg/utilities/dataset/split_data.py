from pathlib import Path
import numpy as np
import os
from typing import List, Tuple, Optional, Any
import tifffile
from numpy.typing import NDArray
from torch import Value

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


def add_padding(img: NDArray, z_from, z_span, r_from, r_span, c_from, c_span, minimum_patch_size: Tuple[int,int,int]) -> NDArray:
    def _zero_fill(img: NDArray, z, r, c):
        z_left_pad = z // 2
        z_right_pad = z - (z_left_pad)
        r_left_pad = r // 2
        r_right_pad = r - (r_left_pad)
        c_left_pad = c // 2
        c_right_pad = c - (c_left_pad)
        return np.pad(img, ((z_left_pad, z_right_pad), (r_left_pad, r_right_pad), (c_left_pad, c_right_pad)))

    z_minimum_size = minimum_patch_size[0]
    patch = img.copy()
    if z_minimum_size > z_span:
        # return img[z_from:z_span+z_from, r_from:r_from+r_span, c_from:c_from+c_span]
        z_pad_left = (z_minimum_size - z_span) // 2
        z_pad_right = z_minimum_size - (z_span + z_pad_left)  # Note: both even and odd spans must be handled

        # Greedily try to expand as much as we can
        z_expand_right = min(img.shape[0], z_from+z_span+z_pad_right) - z_from - z_span
        z_expand_left = z_from - max(0, z_from-z_pad_left)
        patch = patch[z_from-z_expand_left:z_from+z_span+z_expand_right]
    else:
        patch = patch[z_from:z_from+z_span, :, :]

    r_minimum_size = minimum_patch_size[1]
    if r_minimum_size > r_span:
        r_pad_left = (r_minimum_size - r_span) // 2
        r_pad_right = r_minimum_size - (r_span + r_pad_left)

        # Greedily try to expand as much as we can
        r_expand_right = min(patch.shape[1], r_from+r_span+r_pad_right) - r_from - r_span
        r_expand_left = r_from - max(0, r_from-r_pad_left)
        patch = patch[:, r_from-r_expand_left:r_from+r_span+r_expand_right]
    else:
        patch = patch[:, r_from:r_from+r_span, :]

    c_minimum_size = minimum_patch_size[2]
    if c_minimum_size > c_span:
        c_pad_left = (c_minimum_size - c_span) // 2
        c_pad_right = c_minimum_size - (c_span + c_pad_left)

        # Greedily try to expand as much as we can
        c_expand_right = min(patch.shape[2], c_from+c_span+c_pad_right) - c_from - c_span
        c_expand_left = c_from - max(0, c_from-c_pad_left)
        patch = patch[:, :, c_from-c_expand_left:c_from+c_span+c_expand_right]
    else:
        patch = patch[:, :, c_from:c_from+c_span]

    # Now fill with zeros
    return _zero_fill(
        patch,
        max(minimum_patch_size[0] - patch.shape[0], 0),
        max(minimum_patch_size[1] - patch.shape[1], 0),
        max(minimum_patch_size[2] - patch.shape[2], 0) 
    )


def process(gt: NDArray, imgs: NDArray, minimum_patch_size: Tuple[int, int, int], time_dim: int = 0) -> List[Tuple[NDArray, NDArray]]:
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
            zs, rows, cols = bbox_3d(gt_timeslice == tunnel_id)
            # extracted_tunnel_gt = gt_timeslice[zs[0]:zs[1]+1, rows[0]:rows[1]+1, cols[0]:cols[1]+1]
            # extracted_tunnel_img = img_timeslice[zs[0]:zs[1]+1, rows[0]:rows[1]+1, cols[0]:cols[1]+1]
            # extracted_tunnels.append(
            #     (extracted_tunnel_gt, extracted_tunnel_img)
            # )

            if tunnel_id == 0:
                continue  # Skip BG

            if largest_row_range < zs[1]-zs[0]+1:
                largest_row_range = zs[1]-zs[0]+1
                largest_row_range_id = tunnel_id, t_idx
            if largest_col_range < rows[1]-rows[0]+1:
                largest_col_range = rows[1]-rows[0]+1
                largest_col_range_id = tunnel_id, t_idx
            if largest_z_range < cols[1]-cols[0]+1:
                largest_z_range = cols[1]-cols[0]+1
                largest_z_range_id = tunnel_id, t_idx

            # Now figure out the correct padding
            z_span = zs[1]-zs[0]+1
            row_span = rows[1]-rows[0]+1
            col_span = cols[1]-cols[0]+1
            extracted_gt_padded = add_padding(gt_timeslice, zs[0], z_span, rows[0], row_span, cols[0], col_span, minimum_patch_size)
            extracted_img_padded = add_padding(img_timeslice, zs[0], z_span, rows[0], row_span, cols[0], col_span, minimum_patch_size)

            extracted_tunnels.append(
                (extracted_gt_padded, extracted_img_padded)
            )
            
        
    print(f"Largest row range {largest_row_range} found for tunnel {largest_row_range_id[0]} at time slot {largest_row_range_id[1]}")
    print(f"Largest col range {largest_col_range} found for tunnel {largest_col_range_id[0]} at time slot {largest_col_range_id[1]}")
    print(f"Largest z range {largest_z_range} found for tunnel {largest_z_range_id[0]} at time slot {largest_z_range_id[1]}")
    return extracted_tunnels


def main(input_folder: str, output_folder: str, minimum_patch_size: List[int], overwrite_output_flag: bool = False) -> None:
    if len(minimum_patch_size) != 3:
        raise ValueError(f"Invalid minimum patch size. Expected 3 dimensions!")
    print(f"Using minimum patch size: {minimum_patch_size}")

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
    extracted_tunnels = process(gt, imgs, minimum_patch_size=minimum_patch_size)

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
    parser.add_argument('--min_size', type=int, nargs=3, metavar=('MIN_Z', 'MIN_Y', 'MIN_X'),
                        default=[7, 32, 32],
                        help="Minimum size (z, y, x) for each patch. If the bounding box is smaller, it will be padded around the centroid. If not possible, mirror padding is used.")
    args = parser.parse_args()


    main(args.input_folder, args.output_folder, args.min_size, args.output_overwrite)
