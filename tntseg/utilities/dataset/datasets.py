from pathlib import Path
import matplotlib
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import os
import pandas as pd
import tifffile
from typing import Optional, Tuple, List
import numpy as np
from numpy.typing import NDArray


def load_dataset_metadata(img_folder: str, mask_folder: Optional[str] = None) -> pd.DataFrame:
    """
    Load dataset metadata into a pandas DataFrame.
    
    Args:
        img_folder: Path to the image folder
        mask_folder: Optional path to the mask folder
        
    Returns:
        DataFrame with columns: ['img_path', 'mask_path'] or ['img_path'] if no masks
    """
    img_folder_path = Path(img_folder)
    
    if not img_folder_path.exists():
        raise ValueError(f"Image folder does not exist: {img_folder}")
    
    # Get all image files
    img_files = list(img_folder_path.glob("*.tif")) + list(img_folder_path.glob("*.tiff"))
    
    if not img_files:
        raise ValueError(f"No TIFF files found in {img_folder}")
    
    data = []
    for img_file in img_files:
        row = {'img_path': str(img_file)}
        
        if mask_folder is not None:
            mask_folder_path = Path(mask_folder)
            mask_file = mask_folder_path / img_file.name
            
            if not mask_file.exists():
                raise ValueError(f"Mask file not found: {mask_file}")
            
            row['mask_path'] = str(mask_file)
        
        data.append(row)
    
    df = pd.DataFrame(data)
    return df.sort_values('img_path').reset_index(drop=True)


# TODO: add transformation pipeline option
class TNTDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, load_masks: bool = True, transforms: Optional[List] = None, 
                 tile: bool = False, tile_size: Optional[Tuple[int, int, int]] = None, overlap: int = 0,
                 quad_mode: bool = False, quad_idx: Optional[int] = None):
        """
        Dataset for TNT segmentation with support for tiling and quadrant extraction.
        
        Args:
            dataframe: DataFrame with columns ['img_path'] and optionally ['mask_path']
            load_masks: Whether to load mask images
            transforms: Albumentations transforms to apply
            tile: Whether to tile images
            tile_size: Size of tiles as (depth, height, width)
            overlap: Overlap between tiles in pixels
            quad_mode: Whether to extract a quadrant from each image
            quad_idx: Which quadrant to extract (0: top-left, 1: top-right, 2: bottom-left, 3: bottom-right)
        """
        self.dataframe = dataframe.copy()
        self.load_masks = load_masks
        self.transforms = transforms
        self.tile = tile
        self.tile_size = tile_size
        self.overlap = overlap
        self.quad_mode = quad_mode
        self.quad_idx = quad_idx
        self.tile_metadata = []

        if self.load_masks and 'mask_path' not in self.dataframe.columns:
            raise ValueError("load_masks=True requires 'mask_path' column in dataframe")
            
        if self.quad_mode and self.quad_idx not in [0, 1, 2, 3]:
            raise ValueError("quad_idx must be 0, 1, 2, or 3 when quad_mode=True")
            
        # Load and process the data
        self.data = []
        self.mask_data = []
        
        if self.quad_mode and self.tile:
            # Extract quadrant then tile it
            self._extract_quad_and_tile()
        elif self.quad_mode:
            # Just extract quadrant
            self._extract_quad()
        elif self.tile:
            # Tile the whole image
            self._generate_tiles()
        else:
            # Regular mode - load full images
            self._load_full_images()
    
    def _load_full_images(self):
        """Load full images without tiling or quad extraction"""
        for idx, row in self.dataframe.iterrows():
            img = tifffile.imread(row['img_path'])
            if img.dtype != np.uint16:
                img = img.astype(np.float32)
            else:
                img = img.astype(np.float32)
            self.data.append(img)

            if self.load_masks:
                mask = tifffile.imread(row['mask_path'])
                if img.shape != mask.shape:
                    raise RuntimeError(f"Size mismatch {img.shape} != {mask.shape}: {row['img_path']} and {row['mask_path']}")

                # Convert to float32 (handle uint8 masks)
                if mask.dtype == np.uint8 and not set(np.unique(mask)).issubset(set([0,255])):
                    raise RuntimeError(f"Expected just 0 and 255 in file at {row['mask_path']}, got {np.unique(mask)}")
                elif mask.dtype == np.uint8:
                    mask = mask / 255.0

                mask = mask.astype(np.float32)
                self.mask_data.append(mask)
    
    def _extract_quad(self):
        """Extract a specific quadrant from each image"""
        for idx, row in self.dataframe.iterrows():
            img = tifffile.imread(row['img_path'])
            if img.dtype != np.uint16:
                img = img.astype(np.float32)
            else:
                img = img.astype(np.float32)
                
            # Calculate quadrant bounds
            quad_y = 0 if self.quad_idx < 2 else img.shape[1] // 2
            quad_x = 0 if self.quad_idx % 2 == 0 else img.shape[2] // 2
            quad_h = img.shape[1] // 2
            quad_w = img.shape[2] // 2
            
            # Extract quadrant
            quad_img = img[:, quad_y:quad_y+quad_h, quad_x:quad_x+quad_w]
            self.data.append(quad_img)
            
            # Add metadata for reconstruction
            self.tile_metadata.append({
                'image_idx': idx,
                'image_path': row['img_path'],
                'is_quad': True,
                'quad_idx': self.quad_idx,
                'quad_y': quad_y,
                'quad_x': quad_x,
                'full_height': img.shape[1],
                'full_width': img.shape[2],
                'quad_height': quad_h,
                'quad_width': quad_w
            })
            
            if self.load_masks:
                mask = tifffile.imread(row['mask_path'])
                if img.shape != mask.shape:
                    raise RuntimeError(f"Size mismatch {img.shape} != {mask.shape}: {row['img_path']} and {row['mask_path']}")
                
                # Convert mask to float32
                if mask.dtype == np.uint8 and not set(np.unique(mask)).issubset(set([0,255])):
                    raise RuntimeError(f"Expected just 0 and 255 in file at {row['mask_path']}, got {np.unique(mask)}")
                elif mask.dtype == np.uint8:
                    mask = mask / 255.0
                
                mask = mask.astype(np.float32)
                quad_mask = mask[:, quad_y:quad_y+quad_h, quad_x:quad_x+quad_w]
                self.mask_data.append(quad_mask)
    
    def _generate_tiles(self):
        """Generate tiles from full images"""
        if not self.tile_size:
            raise ValueError("tile_size must be specified when tile=True")
            
        for img_idx, row in self.dataframe.iterrows():
            img = tifffile.imread(row['img_path'])
            if img.dtype != np.uint16:
                img = img.astype(np.float32)
            else:
                img = img.astype(np.float32)
            
            mask = None
            if self.load_masks:
                mask = tifffile.imread(row['mask_path'])
                if img.shape != mask.shape:
                    raise RuntimeError(f"Size mismatch {img.shape} != {mask.shape}: {row['img_path']} and {row['mask_path']}")
                
                if mask.dtype == np.uint8 and not set(np.unique(mask)).issubset(set([0,255])):
                    raise RuntimeError(f"Expected just 0 and 255 in file at {row['mask_path']}, got {np.unique(mask)}")
                elif mask.dtype == np.uint8:
                    mask = mask / 255.0
                
                mask = mask.astype(np.float32)
            
            # Calculate number of tiles in each dimension
            d, h, w = img.shape
            td, th, tw = self.tile_size
            
            # Calculate steps with overlap
            step_h = th - self.overlap
            step_w = tw - self.overlap
            
            # Number of tiles in each dimension
            n_h = max(1, (h - th) // step_h + 1) if h > th else 1
            n_w = max(1, (w - tw) // step_w + 1) if w > tw else 1
            
            for i in range(n_h):
                for j in range(n_w):
                    # Calculate tile coordinates
                    start_h = min(i * step_h, h - th)
                    start_w = min(j * step_w, w - tw)
                    
                    # Extract tile
                    tile = img[:td, start_h:start_h+th, start_w:start_w+tw]
                    self.data.append(tile)
                    
                    # Save metadata for reconstruction
                    self.tile_metadata.append({
                        'image_idx': img_idx,
                        'image_path': row['img_path'],
                        'is_quad': False,
                        'is_tile': True,
                        'row': i,
                        'col': j,
                        'start_h': start_h,
                        'start_w': start_w,
                        'height': th,
                        'width': tw,
                        'depth': td,
                        'full_height': h,
                        'full_width': w
                    })
                    
                    if self.load_masks:
                        mask_tile = mask[:td, start_h:start_h+th, start_w:start_w+tw]
                        self.mask_data.append(mask_tile)

    def _extract_quad_and_tile(self):
        """Extract a quadrant and then tile it"""
        if not self.tile_size:
            raise ValueError("tile_size must be specified when tile=True")
            
        for img_idx, row in self.dataframe.iterrows():
            img = tifffile.imread(row['img_path'])
            if img.dtype != np.uint16:
                img = img.astype(np.float32)
            else:
                img = img.astype(np.float32)
            
            # Calculate quadrant bounds
            quad_y = 0 if self.quad_idx < 2 else img.shape[1] // 2
            quad_x = 0 if self.quad_idx % 2 == 0 else img.shape[2] // 2
            quad_h = img.shape[1] // 2
            quad_w = img.shape[2] // 2
            
            # Extract quadrant
            quad_img = img[:, quad_y:quad_y+quad_h, quad_x:quad_x+quad_w]
            
            mask_quad = None
            if self.load_masks:
                mask = tifffile.imread(row['mask_path'])
                if img.shape != mask.shape:
                    raise RuntimeError(f"Size mismatch {img.shape} != {mask.shape}: {row['img_path']} and {row['mask_path']}")
                
                if mask.dtype == np.uint8 and not set(np.unique(mask)).issubset(set([0,255])):
                    raise RuntimeError(f"Expected just 0 and 255 in file at {row['mask_path']}, got {np.unique(mask)}")
                elif mask.dtype == np.uint8:
                    mask = mask / 255.0
                
                mask = mask.astype(np.float32)
                mask_quad = mask[:, quad_y:quad_y+quad_h, quad_x:quad_x+quad_w]
            
            # Tile the quadrant
            d, h, w = quad_img.shape
            td, th, tw = self.tile_size
            
            # Calculate steps with overlap
            step_h = th - self.overlap
            step_w = tw - self.overlap
            
            # Number of tiles in each dimension
            n_h = max(1, (h - th) // step_h + 1) if h > th else 1
            n_w = max(1, (w - tw) // step_w + 1) if w > tw else 1
            
            for i in range(n_h):
                for j in range(n_w):
                    # Calculate tile coordinates
                    start_h = min(i * step_h, h - th)
                    start_w = min(j * step_w, w - tw)
                    
                    # Extract tile from quadrant
                    tile = quad_img[:td, start_h:start_h+th, start_w:start_w+tw]
                    self.data.append(tile)
                    
                    # Save metadata for reconstruction
                    self.tile_metadata.append({
                        'image_idx': img_idx,
                        'image_path': row['img_path'],
                        'is_quad': True,
                        'is_tile': True,
                        'quad_idx': self.quad_idx,
                        'quad_y': quad_y,
                        'quad_x': quad_x,
                        'row': i,
                        'col': j,
                        'start_h': start_h,
                        'start_w': start_w,
                        'height': th,
                        'width': tw,
                        'depth': td,
                        'full_height': img.shape[1],
                        'full_width': img.shape[2],
                        'quad_height': quad_h,
                        'quad_width': quad_w
                    })
                    
                    if self.load_masks:
                        mask_tile = mask_quad[:td, start_h:start_h+th, start_w:start_w+tw]
                        self.mask_data.append(mask_tile)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise ValueError(f"Index {idx} out of range [0, {len(self)})")
        
        data = self.data[idx]
        data = (data - data.min()) / (data.max() - data.min())
        
        if self.load_masks:
            mask = self.mask_data[idx]
            transformed = self.transforms(volume=data, mask3d=mask)
            if self.tile or self.quad_mode:
                return transformed['volume'], transformed['mask3d'], self.tile_metadata[idx]
            return transformed['volume'], transformed['mask3d']
        else:
            transformed = self.transforms(volume=data)
            if self.tile or self.quad_mode:
                return transformed['volume'], self.tile_metadata[idx]
            return transformed['volume']
    
    @staticmethod
    def stitch_predictions(predictions, metadata, output_path=None):
        """
        Stitch together predictions based on tile/quad metadata.
        
        Args:
            predictions: List of prediction arrays
            metadata: List of metadata dictionaries
            output_path: Optional path to save the stitched prediction
            
        Returns:
            Dictionary mapping image_idx to stitched prediction
        """
        # Group by image_idx
        image_groups = {}
        for pred, meta in zip(predictions, metadata):
            img_idx = meta['image_idx']
            if img_idx not in image_groups:
                image_groups[img_idx] = []
            image_groups[img_idx].append((pred, meta))
        
        # Stitch each image
        stitched_results = {}
        for img_idx, group in image_groups.items():
            # Get first metadata to determine output size
            _, meta = group[0]
            d = meta.get('depth', pred.shape[0])
            h = meta['full_height'] 
            w = meta['full_width']
            
            # Create empty output array
            stitched = np.zeros((d, h, w), dtype=np.float32)
            weights = np.zeros((d, h, w), dtype=np.float32)
            
            for pred, meta in group:
                if meta.get('is_quad', False) and not meta.get('is_tile', False):
                    # Just a quadrant, no tiling
                    quad_y = meta['quad_y']
                    quad_x = meta['quad_x']
                    quad_h = meta['quad_height']
                    quad_w = meta['quad_width']
                    
                    stitched[:, quad_y:quad_y+quad_h, quad_x:quad_x+quad_w] = pred
                    weights[:, quad_y:quad_y+quad_h, quad_x:quad_x+quad_w] = 1
                
                elif meta.get('is_tile', False) and not meta.get('is_quad', False):
                    # Tile from full image
                    start_h = meta['start_h']
                    start_w = meta['start_w']
                    th = meta['height']
                    tw = meta['width']
                    
                    stitched[:, start_h:start_h+th, start_w:start_w+tw] += pred
                    weights[:, start_h:start_h+th, start_w:start_w+tw] += 1
                
                elif meta.get('is_quad', False) and meta.get('is_tile', False):
                    # Tile from a quadrant
                    quad_y = meta['quad_y']
                    quad_x = meta['quad_x']
                    start_h = meta['start_h']
                    start_w = meta['start_w']
                    th = meta['height']
                    tw = meta['width']
                    
                    # Calculate global coordinates
                    global_start_h = quad_y + start_h
                    global_start_w = quad_x + start_w
                    
                    stitched[:, global_start_h:global_start_h+th, global_start_w:global_start_w+tw] += pred
                    weights[:, global_start_h:global_start_h+th, global_start_w:global_start_w+tw] += 1
            
            # Average overlapping regions
            mask = weights > 0
            stitched[mask] /= weights[mask]
            
            stitched_results[img_idx] = stitched
            
            # Save if output path is provided
            if output_path:
                os.makedirs(output_path, exist_ok=True)
                output_file = Path(output_path) / f"stitched_img_{img_idx}.tif"
                tifffile.imwrite(str(output_file), stitched)
        
        return stitched_results

# TODO: check if this still works
if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    matplotlib.use("QtAgg")

    # Test the loading functionality
    parser = argparse.ArgumentParser(description="Test TNTDataset loading and visualization.")
    parser.add_argument("--img_folder", type=str, required=True, help="Path to the image folder.")
    parser.add_argument("--mask_folder", type=str, default=None, help="Path to the mask folder (required for training).")
    parser.add_argument("--is_training", action="store_true", help="Set this flag if loading training data (with masks).")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of random samples to visualize.")
    parser.add_argument("--z_slice", type=int, default=4, help="Z-slice index to visualize.")

    args = parser.parse_args()

    # Load dataset
    df = load_dataset_metadata(args.img_folder, args.mask_folder)
    print(f"Loaded {len(df)} samples")
    dataset = TNTDataset(df, load_masks=True)

    num_samples = min(args.num_samples, len(dataset))
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    for idx in indices:
        if args.is_training:
            img, mask = dataset[idx]
        else:
            img = dataset[idx]
            mask = None

        z = args.z_slice if img.ndim == 3 else 0
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2 if mask is not None else 1, 1)
        plt.title(f"Image idx={idx}, z={z}")
        plt.imshow(img[z], cmap="gray")
        if mask is not None:
            plt.subplot(1, 2, 2)
            plt.title(f"Mask idx={idx}, z={z}")
            plt.imshow(mask[z], cmap="gray")
        plt.show()

