from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import os
import tifffile
from typing import List, Tuple, Optional, Dict
from numpy.typing import NDArray
import logging

from torch.distributions.constraints import positive_integer
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def load_images(gt_path: Path, orig_path: Path) -> Tuple[NDArray, NDArray, List[int]]:
    """
    Load ground truth and original images from specified paths.
    
    Args:
        gt_path: Path to the ground truth masks
        orig_path: Path to the original images
        
    Returns:
        Tuple of (gt_array, img_array, time_points)
    """
    gt_images = []
    orig_images = []
    times = []
    
    for gt_file in gt_path.iterdir():
        if not gt_file.name.startswith('mask') or not gt_file.name.endswith('.tif'):
            logger.warning(f"Skipping non-conforming file: {gt_file.name}")
            continue
            
        # Extract id (=time)
        id_str = gt_file.stem[4:]  # Remove 'mask' prefix and extension
        try:
            time_id = int(id_str)
        except ValueError:
            logger.error(f"Could not extract a valid integer id from filename '{gt_file.name}'. Expected format: 'mask<id>.tif'")
            continue
            
        # Load GT image
        gt_img = tifffile.imread(gt_file)
        gt_images.append(gt_img)
        
        # Find and load the corresponding original file
        orig_file = orig_path / f"t{time_id:03}.tif"
        if not orig_file.exists():
            logger.error(f"Missing original image for GT {gt_file.name} (expected {orig_file})")
            continue
            
        orig_img = tifffile.imread(orig_file)
        orig_images.append(orig_img)
        times.append(time_id)
    
    if not gt_images:
        raise ValueError(f"No valid image pairs found in {gt_path} and {orig_path}")
        
    gt = np.array(gt_images)
    imgs = np.array(orig_images)

    if gt.shape != imgs.shape:
        raise RuntimeError(f"GT shape '{gt.shape}' is incompatible with original images '{imgs.shape}'!")

    logger.info(f"Loaded {len(times)} image pairs with shape {gt.shape[1:]}")
    return gt, imgs, times


def get_quadrant_limits(quadrant: int, image_shape: Tuple[int, int, int]) -> Dict[str, Tuple[int, int]]:
        """
        Get the z, row, and column limits for the specified quadrant.
        
        Args:
            quadrant: Quadrant number (1-4)
            image_shape: Shape of the image (z, rows, cols)
            
        Returns:
            Dictionary with 'z', 'r', and 'c' keys mapping to (min, max) tuples
            
        Note:
            Quadrants are arranged as:
               |
            1  |  2
            ---+---
            3  |  4
               |
        """
        z_depth, rows, cols = image_shape
        half_rows, half_cols = rows // 2, cols // 2
        
        # Z limits are always the full depth
        z_limits = (0, z_depth)
        
        if quadrant == 2:  # Top right
            r_limits = (0, half_rows)
            c_limits = (half_cols, cols)
        elif quadrant == 1:  # Top left
            r_limits = (0, half_rows)
            c_limits = (0, half_cols)
        elif quadrant == 3:  # Bottom left
            r_limits = (half_rows, rows)
            c_limits = (0, half_cols)
        elif quadrant == 4:  # Bottom right
            r_limits = (half_rows, rows)
            c_limits = (half_cols, cols)
        else:
            raise ValueError(f"Invalid quadrant: {quadrant}. Must be 1, 2, 3, or 4.")
            
        return {'z': z_limits, 'r': r_limits, 'c': c_limits}

# NOTE: was the previous implementation better?
def bbox_3d(img: NDArray) -> List[Tuple[int, int]]:
    """
    Compute the 3D bounding box of a binary image.
    
    Args:
        img: Binary image
        
    Returns:
        List of (min, max) tuples for each dimension (z, row, col)
    """
    # Get non-zero voxel coordinates
    coords = np.where(img)
    
    if len(coords[0]) == 0:
        raise ValueError("Empty binary image, cannot compute bounding box")
        
    # Get min and max for each dimension
    mins = [coord.min() for coord in coords]
    maxs = [coord.max() for coord in coords]
    
    # Return as list of (min, max) tuples
    return [(min_val, max_val) for min_val, max_val in zip(mins, maxs)]

def compute_overlap_percentage(mask: NDArray, limits: Dict[str, Tuple[int, int]]) -> float:
    """
    Compute the percentage of a mask that falls within the specified limits and return the number of pixels in overlap.
    
    Args:
        mask: Binary mask
        limits: Dictionary with 'z', 'r', and 'c' keys mapping to (min, max) tuples
        
    Returns:
        Percentage of mask voxels within limits (0.0 to 1.0)
    """
    total_voxels = np.sum(mask)
    if total_voxels == 0:
        return 0.0
        
    # Create a mask for the specified region
    region_mask = np.zeros_like(mask, dtype=bool)
    region_mask[
        limits['z'][0]:limits['z'][1],
        limits['r'][0]:limits['r'][1],
        limits['c'][0]:limits['c'][1]
    ] = True
    
    # Count voxels in both the mask and the region
    overlap_voxels = np.sum(mask & region_mask)
    
    return overlap_voxels / total_voxels, overlap_voxels

def extract_patch_with_padding(
    image: NDArray, 
    bbox: List[Tuple[int, int]], 
    min_size: Tuple[int, int, int]
) -> Tuple[NDArray, List[Tuple[int, int]]]:
    """
    Extract a patch from an image using the bounding box and ensure it meets the minimum size.
    
    Args:
        image: Input image
        bbox: Bounding box as [(z_min, z_max), (r_min, r_max), (c_min, c_max)]
        min_size: Minimum size (z, rows, cols)
        
    Returns:
        Tuple of (extracted_patch, extraction_coords)
        Where extraction_coords is [(z_start, z_end), (r_start, r_end), (c_start, c_end)]
    """
    # Extract the original bounds
    (z_min, z_max), (r_min, r_max), (c_min, c_max) = bbox
    
    # Calculate current sizes
    z_size = z_max - z_min + 1
    r_size = r_max - r_min + 1
    c_size = c_max - c_min + 1
    
    # Calculate how much padding we need
    z_pad = max(0, min_size[0] - z_size)
    r_pad = max(0, min_size[1] - r_size)
    c_pad = max(0, min_size[2] - c_size)
    
    # Calculate padding for each side
    z_pad_before = z_pad // 2
    z_pad_after = z_pad - z_pad_before
    r_pad_before = r_pad // 2
    r_pad_after = r_pad - r_pad_before
    c_pad_before = c_pad // 2
    c_pad_after = c_pad - c_pad_before
    
    # Adjust bounds to include padding, while staying within image bounds
    z_start = max(0, z_min - z_pad_before)
    z_end = min(image.shape[0], z_max + 1 + z_pad_after)
    r_start = max(0, r_min - r_pad_before)
    r_end = min(image.shape[1], r_max + 1 + r_pad_after)
    c_start = max(0, c_min - c_pad_before)
    c_end = min(image.shape[2], c_max + 1 + c_pad_after)
    
    # Store the actual extraction coordinates
    extraction_coords = [(z_start, z_end), (r_start, r_end), (c_start, c_end)]
    
    # Extract the patch
    patch = image[z_start:z_end, r_start:r_end, c_start:c_end]
    
    # If the patch is still smaller than the minimum size, try expanding in both directions
    if patch.shape[0] < min_size[0] or patch.shape[1] < min_size[1] or patch.shape[2] < min_size[2]:
        # Calculate remaining size differences
        z_remaining = max(0, min_size[0] - patch.shape[0])
        r_remaining = max(0, min_size[1] - patch.shape[1]) 
        c_remaining = max(0, min_size[2] - patch.shape[2])
        
        # Try expanding extraction coordinates first
        if z_remaining > 0:
            # Try expanding half to left and half to right
            z_left = min(z_remaining//2, z_start)  # How much we can expand left
            z_right = min(z_remaining - z_left, image.shape[0] - z_end)  # Remaining expansion to right
            # If we couldn't expand fully to left, try expanding more to right
            if z_left < z_remaining//2:
                z_right = min(z_remaining - z_left, image.shape[0] - z_end)
            # If we still can't expand enough to right, expand more to left
            if z_right < (z_remaining - z_left):
                z_left = min(z_remaining - z_right, z_start)
            new_z_start = z_start - z_left
            new_z_end = z_end + z_right
            patch = image[new_z_start:new_z_end, r_start:r_end, c_start:c_end]
            extraction_coords[0] = (new_z_start, new_z_end)
            
        if r_remaining > 0:
            # Similar logic for rows
            r_left = min(r_remaining//2, r_start)
            r_right = min(r_remaining - r_left, image.shape[1] - r_end)
            if r_left < r_remaining//2:
                r_right = min(r_remaining - r_left, image.shape[1] - r_end)
            if r_right < (r_remaining - r_left):
                r_left = min(r_remaining - r_right, r_start)
            new_r_start = r_start - r_left
            new_r_end = r_end + r_right
            patch = image[extraction_coords[0][0]:extraction_coords[0][1], 
                 new_r_start:new_r_end, c_start:c_end]
            extraction_coords[1] = (new_r_start, new_r_end)
            
        if c_remaining > 0:
            # Similar logic for columns
            c_left = min(c_remaining//2, c_start)
            c_right = min(c_remaining - c_left, image.shape[2] - c_end)
            if c_left < c_remaining//2:
                c_right = min(c_remaining - c_left, image.shape[2] - c_end)
            if c_right < (c_remaining - c_left):
                c_left = min(c_remaining - c_right, c_start)
            new_c_start = c_start - c_left
            new_c_end = c_end + c_right
            patch = image[extraction_coords[0][0]:extraction_coords[0][1],
                 extraction_coords[1][0]:extraction_coords[1][1],
                 new_c_start:new_c_end]
            extraction_coords[2] = (new_c_start, new_c_end)

        # If still too small, pad with mirror values
        if patch.shape[0] < min_size[0] or patch.shape[1] < min_size[1] or patch.shape[2] < min_size[2]:
            z_pad = max(0, min_size[0] - patch.shape[0])
            r_pad = max(0, min_size[1] - patch.shape[1])
            c_pad = max(0, min_size[2] - patch.shape[2])
            
            pad_width = (
                (z_pad//2, z_pad - z_pad//2),
                (r_pad//2, r_pad - r_pad//2),
                (c_pad//2, c_pad - c_pad//2)
            )
            patch = np.pad(patch, pad_width, mode='reflect')
    
    return patch, extraction_coords

def generate_random_crop(
    image: NDArray, 
    gt_image: NDArray, 
    min_size: Tuple[int, int, int],
    non_training_limits: Dict[str, Tuple[int, int]],
    invert_training_limits: bool = False
) -> Optional[Tuple[NDArray, NDArray, Dict[str, Tuple[int, int]]]]:
    """
    Generate a completely random crop from the non-training region.
    
    Args:
        image: Original image
        gt_image: Ground truth image
        min_size: Minimum crop size (z, rows, cols)
        non_training_limits: Limits of the non-training region
        invert_training_limits: Invert the meaning of training limits
        
    Returns:
        Tuple of (gt_crop, img_crop, crop_coords) where crop_coords is a dict object holding the bbox coordinates
        Returns None if crop is not possible
    """
    if non_training_limits['z'] != (0, 7):
        raise ValueError("Limits on z dimenstion is not supported for creating random crops")
    possible_crop_mask = np.ones(image.shape[1:], dtype=bool)  # This will represent a map of all possible places for the top right coordinate of the patch
    possible_crop_mask[non_training_limits['r'][0]:non_training_limits['r'][1], non_training_limits['c'][0]:non_training_limits['c'][1]] = False
    if invert_training_limits:
        possible_crop_mask = ~possible_crop_mask
        possible_crop_mask[non_training_limits['r'][0]:non_training_limits['r'][1], non_training_limits['c'][0]:non_training_limits['c'][0]+min_size[2]] = False
        possible_crop_mask[non_training_limits['r'][1]-min_size[1]:non_training_limits['r'][1], non_training_limits['c'][0]:non_training_limits['c'][1]] = False
    else:
        possible_crop_mask[:, :min_size[2]] = False
        possible_crop_mask[-min_size[1]:, :] = False

        slice_r = slice(max(0, non_training_limits['r'][0] - min_size[1]), non_training_limits['r'][0]-1)
        if slice_r.start != slice_r.stop:
            slice_c = non_training_limits['c'][0], non_training_limits['c'][1]-1
            possible_crop_mask[slice_r, slice_c] = False

        slice_r = (non_training_limits['r'][0], non_training_limits['r'][1]-1)
        slice_c = slice(non_training_limits['c'][1], min(non_training_limits['c'][1]+min_size[2], image.shape[2]))
        if slice_c.start != slice_c.stop:
            possible_crop_mask[slice_r, slice_c] = False

    # Now pick the topright coordinate randomly
    valid_indices = np.argwhere(possible_crop_mask == True)
    top_right_corner = valid_indices[np.random.randint(valid_indices.shape[0])]
    bottom_left_corner = top_right_corner[0] + min_size[1], top_right_corner[1] - min_size[2]

    slice_r = slice(top_right_corner[0], bottom_left_corner[0])
    slice_c = slice(bottom_left_corner[1], top_right_corner[1])
    gt_crop = gt_image[0:7, slice_r, slice_c]
    img_crop = image[0:7, slice_r, slice_c]

    # Log whether the crop has any tunnels (for information only)
    has_tunnels = np.any(gt_crop > 0)
    logger.debug(f"Generated random crop with tunnels: {has_tunnels}")
    return gt_crop, img_crop, {'z': (0,7), 'r': (slice_r.start, slice_r.stop), 'c': (slice_c.start, slice_c.stop)}

def extract_patches(
    gt: NDArray, 
    imgs: NDArray, 
    min_size: Tuple[int, int, int], 
    train_quad: int,
    overlap_threshold: float = 0.5,
    overlap_threshold_abs: int = 100,
    num_random_crops: int = 0
) -> List[Tuple[NDArray, NDArray, str, List[Tuple[int, int]], List[Tuple[int, int]]]]:
    """
    Extract patches containing tunnels from the non-training quadrant.
    
    Args:
        gt: Ground truth images
        imgs: Original images
        min_size: Minimum patch size (z, rows, cols)
        train_quad: Quadrant to use for training (1-4)
        overlap_threshold: Threshold for determining if a tunnel is in the training quadrant
        num_random_crops: Number of random crops to add
        
    Returns:
        List of (gt_patch, img_patch, patch_id, bbox, extraction_coords) tuples
    """
    logger.info(f"Overlap threshold set to {overlap_threshold}")
    extracted_patches = []
    patch_stats = {
        'z_max': 0, 'r_max': 0, 'c_max': 0,
        'z_avg': 0, 'r_avg': 0, 'c_avg': 0,
        'total': 0
    }
    
    # Calculate non-training quadrant limits (we want tunnels NOT in the training quadrant)
    train_limits = get_quadrant_limits(train_quad, gt.shape[1:])
    
    # Process each time frame
    for t_idx in range(gt.shape[0]):
        gt_slice = gt[t_idx]
        img_slice = imgs[t_idx]
        
        # Find unique tunnel IDs (excluding background which is 0)
        tunnel_ids = np.unique(gt_slice)
        tunnel_ids = tunnel_ids[tunnel_ids != 0]
        
        logger.info(f"Time {t_idx}: Found {len(tunnel_ids)} tunnels")
        
        # Process each tunnel
        for tunnel_id in tunnel_ids:
            # Create binary mask for this tunnel
            tunnel_mask = gt_slice == tunnel_id
            
            # Check overlap with training quadrant
            overlap_perc, overlap_size = compute_overlap_percentage(tunnel_mask, train_limits)
            
            # Skip tunnels that are mostly in the training quadrant
            if overlap_perc > overlap_threshold or overlap_size > overlap_threshold_abs:
                logger.info(f"Skipping tunnel {tunnel_id} ({overlap_perc:.1%}%/{overlap_size}px in training quadrant)")
                continue
                
            # Find bounding box for this tunnel
            try:
                bbox = bbox_3d(tunnel_mask)
            except ValueError:
                logger.warning(f"Empty tunnel mask for ID {tunnel_id}, skipping")
                continue
                
            # Extract patches with padding to meet minimum size
            gt_patch, gt_coords = extract_patch_with_padding(gt_slice, bbox, min_size)
            img_patch, _ = extract_patch_with_padding(img_slice, bbox, min_size)

            # Check if now it overlaps too much with the testing patch
            patch_mask = np.zeros_like(gt_slice, dtype=bool)
            patch_mask[gt_coords[0][0]:gt_coords[0][1], gt_coords[1][0]:gt_coords[1][1], gt_coords[2][0]:gt_coords[2][1]] = True
            patch_overlap_perc, patch_overlap_px = compute_overlap_percentage(patch_mask, train_limits)
            if patch_overlap_perc > overlap_threshold or patch_overlap_px > overlap_threshold_abs:
                logger.warning(f"Padded patch for tunnel {tunnel_id} now has {patch_overlap_perc:.1%}/{patch_overlap_px}px overlap with training quadrant")
                continue
            
            # Create patch ID from time index and tunnel ID
            patch_id = f"t{t_idx}_id{tunnel_id}"
            
            # Update statistics
            patch_stats['total'] += 1
            patch_stats['z_max'] = max(patch_stats['z_max'], gt_patch.shape[0])
            patch_stats['r_max'] = max(patch_stats['r_max'], gt_patch.shape[1])
            patch_stats['c_max'] = max(patch_stats['c_max'], gt_patch.shape[2])
            patch_stats['z_avg'] += gt_patch.shape[0]
            patch_stats['r_avg'] += gt_patch.shape[1]
            patch_stats['c_avg'] += gt_patch.shape[2]
            
            # Log the bounding box and extraction coordinates
            logger.info(f"Tunnel {tunnel_id} bbox: z={bbox[0]}, r={bbox[1]}, c={bbox[2]}")
            logger.info(f"Extraction coords: z={gt_coords[0]}, r={gt_coords[1]}, c={gt_coords[2]}")
            
            extracted_patches.append((gt_patch, img_patch, patch_id, bbox, gt_coords))
    
    # Add random crops if requested
    if num_random_crops > 0:
        logger.info(f"Adding {num_random_crops} random crops")

        for r_idx in range(num_random_crops):
            # Pick random time slot
            t_idx = np.random.randint(0, gt.shape[0])

            # Generate a random crop
            crop_result = generate_random_crop(imgs[t_idx], gt[t_idx], min_size, train_limits)
            gt_crop, img_crop, bbox = crop_result
            patch_id = f"r{r_idx}_t{t_idx}"  # Include time index in random crop ID
            
            # For random crops, the extraction coordinates are the actual crop coordinates
            extraction_coords = [bbox['z'], bbox['r'], bbox['c']]
            
            # The bbox is the same as the extraction coords for random crops
            random_bbox = extraction_coords
            
            logger.info(f"Random crop {patch_id} extraction coords: z={extraction_coords[0]}, r={extraction_coords[1]}, c={extraction_coords[2]}")
            extracted_patches.append((gt_crop, img_crop, patch_id, random_bbox, extraction_coords))
    
    
    # Calculate average patch sizes
    if patch_stats['total'] > 0:
        patch_stats['z_avg'] /= patch_stats['total']
        patch_stats['r_avg'] /= patch_stats['total']
        patch_stats['c_avg'] /= patch_stats['total']
        
        logger.info(f"Extracted {patch_stats['total']} patches")
        logger.info(f"Max sizes (z,r,c): {patch_stats['z_max']}, {patch_stats['r_max']}, {patch_stats['c_max']}")
        logger.info(f"Avg sizes (z,r,c): {patch_stats['z_avg']:.1f}, {patch_stats['r_avg']:.1f}, {patch_stats['c_avg']:.1f}")
    
    return extracted_patches

def extract_test_patches(
    gt: NDArray, 
    imgs: NDArray, 
    min_size: Tuple[int, int, int], 
    train_quad: int,
    overlap_threshold_perc: float = 0.5,
    overlap_threshold_size: int = 100,
    num_random_crops: int = 0
) -> List[Tuple[NDArray, NDArray, str, List[Tuple[int, int]], List[Tuple[int, int]]]]:
    """
    Extract test patches from the training quadrant.
    
    Args:
        gt: Ground truth images
        imgs: Original images
        min_size: Minimum patch size (z, rows, cols)
        train_quad: Quadrant to use for training/testing (1-4)
        overlap_threshold: Threshold for determining if a tunnel is in the training quadrant
        num_random_crops: Number of random crops to add
        
    Returns:
        List of (gt_patch, img_patch, patch_id, bbox, extraction_coords) tuples from the test quadrant
    """
    extracted_patches = []
    patch_stats = {
        'z_max': 0, 'r_max': 0, 'c_max': 0,
        'z_avg': 0, 'r_avg': 0, 'c_avg': 0,
        'total': 0
    }
    
    # Get the training quadrant limits - this is our test region
    test_limits = get_quadrant_limits(train_quad, gt.shape[1:])
    
    logger.info(f"Extracting test patches from quadrant {train_quad}")
    
    # Process each time frame
    for t_idx in range(gt.shape[0]):
        gt_slice = gt[t_idx]
        img_slice = imgs[t_idx]
        
        # Find unique tunnel IDs (excluding background which is 0)
        tunnel_ids = np.unique(gt_slice)
        tunnel_ids = tunnel_ids[tunnel_ids != 0]
        
        logger.info(f"Time {t_idx}: Found {len(tunnel_ids)} tunnels")
        
        # Process each tunnel
        for tunnel_id in tunnel_ids:
            # Create binary mask for this tunnel
            tunnel_mask = gt_slice == tunnel_id
            
            # Check overlap with training/test quadrant
            overlap_perc, overlap_px = compute_overlap_percentage(tunnel_mask, test_limits)
            
            # Only include tunnels that are MOSTLY IN the test quadrant
            if overlap_perc < overlap_threshold_perc:
                logger.debug(f"Skipping tunnel {tunnel_id} (only {overlap_perc:.1%} in test quadrant)")
                continue
                
            # Find bounding box for this tunnel
            try:
                bbox = bbox_3d(tunnel_mask)
            except ValueError:
                logger.warning(f"Empty tunnel mask for ID {tunnel_id}, skipping")
                continue
                
            # Extract patches with padding to meet minimum size
            gt_patch, gt_coords = extract_patch_with_padding(gt_slice, bbox, min_size)
            img_patch, _ = extract_patch_with_padding(img_slice, bbox, min_size)
            
            # Create patch ID from time index and tunnel ID
            patch_id = f"t{t_idx}_id{tunnel_id}"
            
            # Update statistics
            patch_stats['total'] += 1
            patch_stats['z_max'] = max(patch_stats['z_max'], gt_patch.shape[0])
            patch_stats['r_max'] = max(patch_stats['r_max'], gt_patch.shape[1])
            patch_stats['c_max'] = max(patch_stats['c_max'], gt_patch.shape[2])
            patch_stats['z_avg'] += gt_patch.shape[0]
            patch_stats['r_avg'] += gt_patch.shape[1]
            patch_stats['c_avg'] += gt_patch.shape[2]
            
            # Log the bounding box and extraction coordinates
            logger.info(f"Test tunnel {tunnel_id} bbox: z={bbox[0]}, r={bbox[1]}, c={bbox[2]}")
            logger.info(f"Test extraction coords: z={gt_coords[0]}, r={gt_coords[1]}, c={gt_coords[2]}")
            
            extracted_patches.append((gt_patch, img_patch, patch_id, bbox, gt_coords))
    
    # Add random crops from the test quadrant if requested
    if num_random_crops > 0:
        logger.info(f"Adding {num_random_crops} random crops from test quadrant")
        
        for r_idx in range(num_random_crops):
            # Pick a random time slice
            t_idx = np.random.randint(0, gt.shape[0])
            
            # Generate a random crop from the test quadrant
            crop_result = generate_random_crop(
                imgs[t_idx], gt[t_idx], min_size, test_limits, invert_training_limits=True
            )
            
            if crop_result is not None:
                gt_crop, img_crop, bbox = crop_result
                patch_id = f"r{r_idx}_t{t_idx}"  # Include time index
                
                random_coords = [bbox['z'], bbox['r'], bbox['c']]
                
                logger.info(f"Random test crop {patch_id} coords: z={random_coords[0]}, r={random_coords[1]}, c={random_coords[2]}")
                extracted_patches.append((gt_crop, img_crop, patch_id, random_coords, random_coords))
    
    # Calculate average patch sizes
    if patch_stats['total'] > 0:
        patch_stats['z_avg'] /= patch_stats['total']
        patch_stats['r_avg'] /= patch_stats['total']
        patch_stats['c_avg'] /= patch_stats['total']
        
        logger.info(f"Extracted {patch_stats['total']} test patches")
        logger.info(f"Test max sizes (z,r,c): {patch_stats['z_max']}, {patch_stats['r_max']}, {patch_stats['c_max']}")
        logger.info(f"Test avg sizes (z,r,c): {patch_stats['z_avg']:.1f}, {patch_stats['r_avg']:.1f}, {patch_stats['c_avg']:.1f}")
    
    return extracted_patches

def main(
    input_folder: str, 
    output_folder: str, 
    min_size: List[int], 
    train_quad: int,
    overwrite: bool = False, 
    num_random_crops_train: int = 0,
    num_random_crops_test: int = 0,
    overlap_threshold: float = 0.5,
    overlap_threshold_px: int = 100
) -> None:
    """
    Main function to extract and save training and testing patches.
    
    Args:
        input_folder: Path to input folder
        output_folder: Path to output folder
        min_size: Minimum patch size [z, rows, cols]
        train_quad: Quadrant to use for training (1-4)
        overwrite: Whether to overwrite existing output
        num_random_crops_train: Number of random crops to add to the training set
        num_random_crops_test: Number of random crops to add to the testing set
        overlap_threshold: Threshold for determining if a tunnel is in the training quadrant
    """
    if len(min_size) != 3:
        raise ValueError(f"Invalid minimum patch size. Expected 3 dimensions!")

    input_folder_path = Path(input_folder)
    output_folder_path = Path(output_folder)
    
    # Create output directories for both training and testing
    if output_folder_path.exists() and not overwrite:
        raise ValueError(f"Output folder {output_folder} already exists. Use --overwrite to overwrite.")
    output_folder_path.mkdir(parents=True, exist_ok=overwrite)

    # Configure output logging to file
    fh = logging.FileHandler(str(output_folder_path / 'dataset_split.log'))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    
    logger.info(f"Using minimum patch size: {min_size}")
    logger.info(f"Using training quadrant: {train_quad}")
    
    
    # Training directories
    train_path = output_folder_path / "train"
    os.makedirs(train_path / "GT", exist_ok=overwrite)
    os.makedirs(train_path / "IMG", exist_ok=overwrite)
    os.makedirs(train_path / "GT_MERGED_LABELS", exist_ok=overwrite)
    
    # Testing directories
    test_path = output_folder_path / "test"
    os.makedirs(test_path / "GT", exist_ok=overwrite)
    os.makedirs(test_path / "IMG", exist_ok=overwrite)
    os.makedirs(test_path / "GT_MERGED_LABELS", exist_ok=overwrite)
    
    # Find input paths
    gt_path = input_folder_path / "01_GT" / "SEG"
    orig_path = input_folder_path / "01"
    
    if not gt_path.exists():
        raise RuntimeError(f"Path to GT '{str(gt_path)}' does not exist!")
    if not orig_path.exists():
        raise RuntimeError(f"Path to orig. '{str(orig_path)}' does not exist!")
    
    # Load images
    gt, imgs, times = load_images(gt_path, orig_path)
    logger.info(f"Loaded {len(times)} time points with annotations")
    
    # Extract training patches
    train_patches = extract_patches(
        gt, imgs, 
        min_size=tuple(min_size), 
        train_quad=train_quad,
        overlap_threshold=overlap_threshold,
        overlap_threshold_abs=overlap_threshold_px,
        num_random_crops=num_random_crops_train
    )
    
    # Extract testing patches
    test_patches = extract_test_patches(
        gt, imgs,
        min_size=tuple(min_size),
        train_quad=train_quad,
        overlap_threshold_perc=overlap_threshold,
        overlap_threshold_size=overlap_threshold_px,
        num_random_crops=num_random_crops_test
    )
    
    # Save training patches
    logger.info(f"Saving {len(train_patches)} training patches")
    for i, (patch_gt, patch_img, patch_id, bbox, extraction_coords) in enumerate(train_patches):
        logger.info(f"Saving training patch {patch_id}")
        logger.info(f"  Original bbox: z={bbox[0]}, r={bbox[1]}, c={bbox[2]}")
        logger.info(f"  Extraction coords: z={extraction_coords[0]}, r={extraction_coords[1]}, c={extraction_coords[2]}")
        
        # Save the original patch ID which includes time and tunnel ID info
        tifffile.imwrite(train_path / "GT" / f"{patch_id}.tif", patch_gt)
        tifffile.imwrite(train_path / "IMG" / f"{patch_id}.tif", patch_img)
        binary_mask = (patch_gt > 0).astype(np.uint8) * 255
        tifffile.imwrite(train_path / "GT_MERGED_LABELS" / f"{patch_id}.tif", binary_mask)

    # Save testing patches
    logger.info(f"Saving {len(test_patches)} testing patches")
    for i, (patch_gt, patch_img, patch_id, bbox, extraction_coords) in enumerate(test_patches):
        logger.info(f"Saving testing patch {patch_id}")
        logger.info(f"  Original bbox: z={bbox[0]}, r={bbox[1]}, c={bbox[2]}")
        logger.info(f"  Extraction coords: z={extraction_coords[0]}, r={extraction_coords[1]}, c={extraction_coords[2]}")
        
        tifffile.imwrite(test_path / "GT" / f"{patch_id}.tif", patch_gt)
        tifffile.imwrite(test_path / "IMG" / f"{patch_id}.tif", patch_img)
        binary_mask = (patch_gt > 0).astype(np.uint8) * 255
        tifffile.imwrite(test_path / "GT_MERGED_LABELS" / f"{patch_id}.tif", binary_mask)
    
    # Create visualization of train/test split
    logger.info("Creating visualization of train/test split")
    
    # Create a visualization for each time point
    for t_idx in range(gt.shape[0]):
        # Create RGB image (3 channels)
        vis_img = np.zeros((*gt[t_idx].shape, 3), dtype=np.uint8)
        
        # Create copy of GT for visualization
        vis_gt = gt[t_idx].copy()
        
        # Color training patches in blue, testing patches in red
        for patch_gt, _, patch_id, _, coords in train_patches:
            if patch_id.startswith(f't{t_idx}_'):  # Match time index
                z_coords, r_coords, c_coords = coords
                # Visualization
                vis_img[z_coords[0]:z_coords[1], r_coords[0]:r_coords[1], c_coords[0]:c_coords[1], 2] = 128
                
                # Extract tunnel ID from patch_id (format: t<time>_id<tunnel_id>)
                tunnel_id = int(patch_id.split('_id')[1])
                # Remove this tunnel ID from the GT visualization
                vis_gt[gt[t_idx] == tunnel_id] = 128

        for _, _, patch_id, _, coords in test_patches:
            if patch_id.startswith(f't{t_idx}_'):  # Match time index
                z_coords, r_coords, c_coords = coords
                # Visualization
                vis_img[z_coords[0]:z_coords[1], r_coords[0]:r_coords[1], c_coords[0]:c_coords[1], 0] = 128
                
                # Extract tunnel ID from patch_id
                tunnel_id = int(patch_id.split('_id')[1])
                vis_gt[gt[t_idx] == tunnel_id] = 256

        
        # Save visualization for each z-slice
        vis_path = output_folder_path / "visualizations"
        os.makedirs(vis_path, exist_ok=True)
        for z in range(vis_img.shape[0]):
            tifffile.imwrite(vis_path / f"split_vis_t{t_idx}_z{z}.tif", vis_img[z])
            # Also save the modified GT showing which objects went where
            tifffile.imwrite(vis_path / f"split_gt_t{t_idx}_z{z}.tif", vis_gt[z])

        # Do a max projection on the original
        maxproj_img = np.max(gt[t_idx]!=0, axis=0)
        plt.figure()
        plt.imshow(maxproj_img, 'gray')
        plt.imshow(np.max(vis_img, axis=0), alpha=0.5)
        plt.savefig(vis_path / f"t{t_idx}.png", dpi=300)
        plt.show()
        

    # Create separate visualization for random crops
    logger.info("Creating visualization of random crops")
    rand_vis_path = output_folder_path / "random_crops_vis"
    os.makedirs(rand_vis_path, exist_ok=True)
    
    for t_idx in range(gt.shape[0]):
        # Create RGB image for random crops
        rand_vis_img = np.zeros((*gt[t_idx].shape, 3), dtype=np.uint8)
        
        # Mark training random crops in green
        for _, _, patch_id, _, coords in train_patches:
            if patch_id.startswith(f'r') and f't{t_idx}' in patch_id:
                z_coords, r_coords, c_coords = coords
                rand_vis_img[z_coords[0]:z_coords[1], r_coords[0]:r_coords[1], c_coords[0]:c_coords[1], 1] = 128
        
        # Mark testing random crops in yellow
        for _, _, patch_id, _, coords in test_patches:
            if patch_id.startswith(f'r') and f't{t_idx}' in patch_id:
                z_coords, r_coords, c_coords = coords
                rand_vis_img[z_coords[0]:z_coords[1], r_coords[0]:r_coords[1], c_coords[0]:c_coords[1], 1:] = 128
        
        # Save visualization for each z-slice
        for z in range(rand_vis_img.shape[0]):
            tifffile.imwrite(rand_vis_path / f"random_crops_vis_t{t_idx}_z{z}.tif", rand_vis_img[z])

    logger.info("Done!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Split dataset images with padding.")
    parser.add_argument("input_folder", type=str, help="Path to the folder containing images.")
    parser.add_argument("output_folder", type=str, help="Path to the output folder.")
    parser.add_argument("--train_quad", type=int, choices=[1, 2, 3, 4], default=1,
                        help="Quadrant to use for training (1=top-right, 2=top-left, 3=bottom-left, 4=bottom-right)")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite output folder if it exists")
    parser.add_argument('--min_size', type=int, nargs=3, metavar=('MIN_Z', 'MIN_Y', 'MIN_X'),
                        default=[7, 32, 32],
                        help="Minimum size (z, y, x) for each patch.")
    parser.add_argument("--random_crops_train", type=int, default=0, help="Number of random crops to add to the training set.")
    parser.add_argument("--random_crops_test", type=int, default=0, help="Number of random crops to add to the testing set.")
    parser.add_argument("--overlap_threshold", type=float, default=0.2, 
                        help="Threshold for determining if a tunnel is in the training quadrant.")
    parser.add_argument("--overlap_threshold_px", type=int, default=100)
    
    args = parser.parse_args()

    main(
        args.input_folder, 
        args.output_folder, 
        args.min_size, 
        args.train_quad,
        args.overwrite, 
        args.random_crops_train,
        args.random_crops_test,
        args.overlap_threshold,
        args.overlap_threshold_px
    )
