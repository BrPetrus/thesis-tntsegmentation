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
    # def __init__(self, img_folder: str, mask_folder: Optional[str] = None, is_training: bool = False):
    def __init__(self, dataframe: pd.DataFrame, load_masks: bool = True):
        self.dataframe = dataframe.copy()
        self.load_masks = load_masks

        # if self.is_training and mask_folder is None:
        if self.load_masks and 'mask_path' not in self.dataframe.columns:
            raise ValueError("load_masks=True requires 'mask_path' column in dataframe")

        self.data = []
        self.mask_data = []

        # for img_file in img_folder_path.iterdir():
            # Load both mask and original img
            # img_full_path = img_folder_path / img_file
        for idx, row in self.dataframe.iterrows():
            img = tifffile.imread(row['img_path'])  # NOTE: img_file shoud contain just the id + suffix
            if img.dtype != np.uint16:
                raise RuntimeError(f"Expected unsigned 16bit integer got {img.dtype} for image at path {row['img_path']}")
            self.data.append(img)

            if self.load_masks:
                # mask_full_path = mask_folder_path / img_file.name
                mask = tifffile.imread(row['mask_path'])
                if img.shape != mask.shape:
                    raise RuntimeError(f"Size mismatch {img.shape} != {mask.shape}: {row['img_path']} and {row['mask_path']}")

                # Convert to boolean array
                if mask.dtype == np.uint8 and not set(np.unique(mask)).issubset(set(0,255)):
                    raise RuntimeError(f"Expected just 0 and 255 in file at {row['mask_path']}, got {np.unique(mask)}")
                elif mask.dtype == np.uint8:
                    mask //= 255
                if mask.dtype != np.bool:
                    raise RuntimeError(f"Expected the mask at {row['mask_path']} to be a boolean data!")

                mask = mask.astype(np.bool)
                self.mask_data.append(mask)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> NDArray[np.uint16] | Tuple[NDArray[np.uint16], NDArray[np.bool]]:
        if idx < 0 or idx >= len(self):
            raise ValueError(f"Index {idx} out of range [0, {len(self)})")
        if self.load_masks:
            return self.data[idx], self.mask_data[idx]
        return self.data[idx]

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

