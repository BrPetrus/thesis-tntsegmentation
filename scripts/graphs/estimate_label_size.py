"""
Copyright 2025 Bruno Petrus

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from pathlib import Path
import tifffile
import numpy as np
import matplotlib.pyplot as plt


def read_all_tiffs(directory_path):
    path = Path(directory_path)

    # Find all TIFF files
    tiff_files = list(path.glob("*.tif"))

    images = []
    filenames = []

    for tiff_file in tiff_files:
        try:
            image = tifffile.imread(tiff_file)
            images.append(image)
            filenames.append(tiff_file.name)
            print(f"Loaded: {tiff_file.name}, shape: {image.shape}")
        except Exception as e:
            print(f"Error loading {tiff_file}: {e}")

    return images, filenames


def plot_histogram(sizes):
    """Plot histogram of label sizes."""
    plt.figure(figsize=(10, 4))

    # Set larger font sizes for thesis readability
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        }
    )

    plt.hist(sizes, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
    plt.xlabel("Label Size (pixels)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Label Sizes")
    plt.grid(True, alpha=0.3)

    # Add statistics
    mean_size = np.mean(sizes)
    median_size = np.median(sizes)

    plt.axvline(
        mean_size,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_size:.1f}",
    )
    plt.axvline(
        median_size,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_size:.1f}",
    )
    plt.legend()

    plt.tight_layout()
    plt.savefig("label_sizes.png")
    plt.show()


def main(path: Path):
    imgs, _ = read_all_tiffs(path)

    sizes = []

    for img in imgs:
        for lab in np.unique(img):
            if lab == 0:
                continue
            sizes.append(np.sum(img == lab))

    print(f"Mean area: {sum(sizes) / len(sizes)}")

    plot_histogram(sizes)


if __name__ == "__main__":
    main("/home/bruno/DP/dataset/180322_Sqh-mCh Tub-GFP 16h_110 Annotations/01_GT/SEG/")
