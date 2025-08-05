import numpy as np
import scipy
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Any, Optional, List, Tuple
import scipy.linalg
import skimage.filters as skfilters
import skimage.morphology as skmorph
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from torch import Value
from morphology import create_umbra, fillhole_morphology
from filters import percentile_stretch
from PIL import Image
import argparse
from dataclasses import dataclass
from pathlib import Path
import tifffile
import skimage
from sklearn.metrics import f1_score, jaccard_score, accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay

@dataclass
class ParametersSettings:
    sigma: List[float] | float

def cell_border_heuristic(segmentation: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if segmentation.dtype != np.uint8:
        raise ValueError(f"Expected uint8 type, got {segmentation.dtype}")
    segmentation = segmentation.copy()

    # for z_slice in segmentation.shape[0]:
    #     # Find cell 
    


def find(img: NDArray, output_folder: Path, params: ParametersSettings) -> NDArray[np.uint8]:
    if img.ndim != 3:
        raise ValueError("Expected 3D data")

    # Preprocessing
    img = img.copy()
    if not output_folder.exists():
        output_folder.mkdir(parents=True)
    stretched = (255 * percentile_stretch(img)).astype(np.uint8)

    # Remove tunnels
    # 2d_se = skmorph.
    # se = skmorph.disk(7)
    # eroded = stretched.copy()
    # for z_index in range(img.shape[0]):
    #     # eroded[z_index] = skmorph.erosion(eroded[z_index], se)
    #     # Remove tunnels
    #     eroded[z_index] = skmorph.erosion(eroded[z_index], se)
    # se = skmorph.footprint_rectangle([3, 6, 6])
    # se = skmorph.disk(3)
    # se = np.stack([se, se, se])
    # print(se.shape)
    # eroded = skmorph.opening(stretched, se, mode='min')

    # se = skmorph.disk(3)
    # eroded = stretched.copy()
    # for z_index in range(img.shape[0]):
    #     # Remove tunnels
    #     eroded[z_index] = skmorph.opening(eroded[z_index], se)
    se = skmorph.footprint_rectangle([5, 4, 4])
    eroded = skmorph.opening(stretched, se, mode='mirror')

    # Filling holes
    filled = eroded.copy()
    for z_index in range(img.shape[0]):
        slice = filled[z_index].copy()
        filled_slice, _ = fillhole_morphology(slice)
        filled[z_index] = filled_slice
    
    # Binarise
    threshold = skfilters.threshold_otsu(filled)
    thresholded = np.zeros_like(filled)
    thresholded[filled > threshold] = 255
    thresholded = skmorph.closing(thresholded, skmorph.ball(3))
    thresholded = thresholded.astype(np.bool)

    # Now do a low threshold and mask out
    # low_eroded = skmorph.erosion(stretched, skmorph.ball(2))

    se = skmorph.disk(1)
    eroded_low = stretched.copy()
    for z_index in range(img.shape[0]):
        # Remove tunnels
        eroded_low[z_index] = skmorph.opening(eroded_low[z_index], se)

    threshold_low = skfilters.threshold_otsu(eroded_low)
    threshold_low_mask = np.zeros_like(filled, dtype=np.bool)
    threshold_low_mask[eroded_low > threshold_low] = True
    threshold_low_mask[thresholded] = False 

    # Now remove small objects
    thresholded_no_small = skmorph.remove_small_objects(threshold_low_mask, 100)

    result = np.zeros_like(stretched, dtype=np.uint8)
    result[thresholded] = 255
    result[thresholded_no_small] = 127

    
    # Saving
    tifffile.imwrite(output_folder / "stretched.tif", stretched)
    tifffile.imwrite(output_folder / "filled.tif", filled)
    tifffile.imwrite(output_folder / "eroded.tif", eroded)
    tifffile.imwrite(output_folder / "thresholded.tif", thresholded)
    tifffile.imwrite(output_folder / "result.tif", result)
    tifffile.imwrite(output_folder / "eroded_low.tif", eroded_low)
    tifffile.imwrite(output_folder / "no_small.tif", thresholded_no_small)

    # tifffile.imwrite(output_folder / "blurred.tif", blurred)

    return result

def evaluate(pred: NDArray[np.floating | np.integer], label: NDArray[np.uint16]):
    if label.dtype == np.uint16:
        label[label > 0] = 255  # Convert instance segmentation to semantic segmentation
        # raise ValueError(f"Expected uint16, got {label.dtype} for ground truth.")
    # if label.max() != 255 or label.min() != 0:
        # raise ValueError(f"Ground truth values must belong to set {0, 1}!")

    if pred.dtype == np.float32:
        if pred.max() > 1.0 or pred.min() < 0.0:
            raise ValueError(f"Unrecognizable prediction. The values must be in range [0, 1]")
        pred = (pred * 255).astype(np.uint8)
    elif pred.dtype == np.uint8:
        if pred.max() != 255 or pred.min() != 0:
            raise ValueError(f"Ground truth values must belong to set {0, 255}!")
    else:
        raise ValueError(f"Expected float32 or uint8, got {pred.dtype}")
    # Visualize 4 random slices
    # indices = np.random.choice(pred.shape[0], size=4, replace=False)
    nrows=7
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(10, 10))
    for i in range(nrows):
        axes[i][0].imshow(pred[i])
        axes[i][1].imshow(label[i])
    plt.tight_layout()
    pred = pred.flatten()
    label = label.flatten()

    prec = precision_score(label, pred, pos_label=255)
    acc = accuracy_score(label, pred)
    recall = recall_score(label, pred, pos_label=255)
    f1 = f1_score(label, pred, pos_label=255)
    jaccard = jaccard_score(label, pred, pos_label=255)

    print(f"precision: {prec}\n"
          f"accuracy: {acc}\n"
          f"recall: {recall}\n"
          f"f1: {f1}\n"
          f"jaccard: {jaccard}")

    ConfusionMatrixDisplay.from_predictions(label, pred)

    plt.show()
    return prec, acc, recall, f1, jaccard



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run algorithm to find cells")
    parser.add_argument("image", type=Path, help="Path to the image")
    parser.add_argument("--output", type=Path, help="Output folder", default=Path('.') / "output")
    parser.add_argument("--groundtruth", type=Path, help="Ground Truth", default=None)
    args = parser.parse_args()

    # Load the image into memory
    img = tifffile.imread(args.image)

    # Params
    params = ParametersSettings(
        sigma=[0, 2, 2]
    )

    res = find(img, args.output, params)
    res[res == 255] = 0  # Remove cells
    res[res == 127] = 255  # Just the tunnels

    # enter evaluations
    if args.groundtruth != None:
        gt = tifffile.imread(args.groundtruth)
        evaluate(res, gt)