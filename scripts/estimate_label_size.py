from pathlib import Path
import tifffile
import numpy as np
import matplotlib.pyplot as plt

def read_all_tiffs(directory_path):
    path = Path(directory_path)

    # Find all TIFF files
    tiff_files = list(path.glob('*.tif'))

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
    plt.figure(figsize=(10, 6))
    plt.hist(sizes, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Label Size (pixels)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Label Sizes')
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mean_size = np.mean(sizes)
    median_size = np.median(sizes)
    
    plt.axvline(mean_size, color='red', linestyle='--', label=f'Mean: {mean_size:.1f}')
    plt.axvline(median_size, color='orange', linestyle='--', label=f'Median: {median_size:.1f}')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def main(path: Path):
    imgs, _ = read_all_tiffs(path)

    sizes = []

    for img in imgs:
        for lab in np.unique(img):
            if lab == 0:
                continue
            sizes.append(np.sum(img==lab))
        
    
    print(f"Mean area: {sum(sizes) / len(sizes)}")

    plot_histogram(sizes)

if __name__ == "__main__":
    main("/home/bruno/DP/dataset/180322_Sqh-mCh Tub-GFP 16h_110 Annotations/01_GT/SEG/")

