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

# TODO: add transformation pipeline option
class TNTDataset(Dataset):
    def __init__(self, img_folder: str, mask_folder: Optional[str] = None, is_training: bool = False):
        self.is_training = is_training

        if self.is_training and mask_folder is None:
            raise ValueError("When training, mask_folder parameter must be provided.")

        self.data = []
        self.mask_data = []
        img_folder_path = Path(img_folder)
        if self.is_training:
            mask_folder_path = Path(mask_folder)

        for img_file in img_folder_path.iterdir():
            # Load both mask and original img
            # img_full_path = img_folder_path / img_file
            img = tifffile.imread(img_file)  # NOTE: img_file shoud contain just the id + suffix
            if img.dtype != np.uint16:
                raise RuntimeError(f"Expected unsigned 16bit integer got {img.dtype} for image at path {img_file}")
            self.data.append(img)

            if self.is_training:
                mask_full_path = mask_folder_path / img_file.name
                mask = tifffile.imread(mask_full_path)
                if img.shape != mask.shape:
                    raise RuntimeError(f"Size mismatch {img.shape} != {mask.shape}: {img_folder_path / img_file} and {mask_folder_path / img_file}")

                # Convert to boolean array
                if mask.dtype == np.uint8 and not set(np.unique(mask)).issubset(set(0,255)):
                    raise RuntimeError(f"Expected just 0 and 255 in file at {mask_full_path}, got {np.unique(mask)}")
                elif mask.dtype == np.uint8:
                    mask //= 255
                if mask.dtype != np.bool:
                    raise RuntimeError(f"Expected the mask at {mask_full_path} to be a boolean data!")

                mask = mask.astype(np.bool)
                self.mask_data.append(mask)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> NDArray[np.uint16] | Tuple[NDArray[np.uint16], NDArray[np.bool]]:
        if idx < 0 or idx > len(self):
            raise ValueError(f"Index {idx} out of range [0, {len(self)})")
        if self.is_training:
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
    parser.add_argument("--num_samples", type=int, default=3, help="Number of random samples to visualize.")
    parser.add_argument("--z_slice", type=int, default=0, help="Z-slice index to visualize.")

    args = parser.parse_args()

    dataset = TNTDataset(
        img_folder=args.img_folder,
        mask_folder=args.mask_folder,
        is_training=args.is_training
    )

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

