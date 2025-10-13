from dataclasses import dataclass
from typing import Any, List, Dict
import numpy as np
from numpy.typing import NDArray

import skimage.morphology as skmorph
import skimage.filters as skfilt
import skimage.measure as skmeas

import tntseg.utilities.metrics.metrics as tntmetrics

from sklearn.externals.array_api_compat.numpy import False_
import tifffile

from pathlib import Path
import argparse

import matplotlib.pyplot as plt



@dataclass(frozen=True)
class PostprocessConfig:
    minimum_size_px: int
    se_type: Any
    se_size: int

def post_process_output(predictions: NDArray[np.bool], config: PostprocessConfig) -> List[Dict]:

    if not np.issubdtype(predictions.dtype, np.bool):
        raise ValueError("Expected a boolean array.")
    if predictions.ndim != 3:
        raise ValueError("Expected 3D data")

    # # Construct the SE
    # se = skmorph.disk(config.se_size)
    
    # # Clean up small stuff
    # skmorph.binary_opening(predictions)    

    ## Cleanup small parts
    #removed_small = skmorph.remove_small_objects(predictions, config.minimum_size_px)

    # Find connected components
    labelled = skmorph.label(predictions, background=0)
    regions = skmeas.regionprops(labelled)

    # Filter small regions
    big_regions = []
    labeled_regions = np.zeros_like(labelled, dtype=np.uint8)
    for region in regions:
        if region['area'] < config.minimum_size_px:
            continue
        big_regions.append(region)
        labeled_regions[labelled == region['label']] = region['label']
    
    return big_regions, labeled_regions

    # # Visualise
    # fig, ax = plt.subplots()
    # result_vis = np.zeros_like(labelled, dtype=np.uint8)
    # all_regions = np.zeros_like(labelled, dtype=np.uint8)
    # # ax.imshow(np.max(predictions, axis=0)*255, 'gray')
    # result_vis[predictions == True] = 255
    # for region in big_regions:
    #     # ax.imshow(np.max(region['label'] == labelled, axis=0), alpha=0.5, cmap='gray')
    #     result_vis[region['label'] == labelled] = 128

    #     all_regions[region['label'] == labelled] =         
    # ax.imshow(np.max(result_vis, axis=0))
    # fig.savefig(str(output_folder / 'post_process.tif'))
    # plt.show()
    # # tifffile.imwrite(output / '')


@dataclass(frozen=True)
class TunnelDetectionResult:
    tp: int
    fp: int
    tn: int
    fn: int
    spec: float
    sens: float
    f1: float
    recall: float
    prec: float

def detect_tunnels(predictions: NDArray[np.bool], labels: NDArray[np.uint8], image: NDArray, config: PostprocessConfig, output_folder: Path) -> TunnelDetectionResult:
    img_q3 = np.quantile(img, 0.03)
    img_q97 = np.quantile(img, 0.97)
    image = image.copy()
    image[image <= img_q3] = img_q3
    image[image >= img_q97] = img_q97
    image = (image - image.min()) / (image.max() - image.min())
    image = image *255
    image = image.astype(np.uint8) 

    regions_prop, labeled_prediction = post_process_output(predictions, config)
    binary_filtered_pred = labeled_prediction != 0
    binary_gt = labels != 0

    # Calculate basic metrics
    TP = np.sum((binary_filtered_pred == True) & (binary_gt == True))
    TN = np.sum((binary_filtered_pred == False) & (binary_gt == False))
    FP = np.sum((binary_filtered_pred == True) & (binary_gt == False))
    FN = np.sum((binary_filtered_pred == False) & (binary_gt == True))

    prec = tntmetrics.precision(TP, FP)
    recall = tntmetrics.recall(TP, FN)
    f1 = tntmetrics.dice_coefficient(TP, FP, FN)

    print(f"prec={prec}, recall={recall}, f1={f1}")

    # Tunnel matching logic
    mapping = {}
    for label_id in np.unique(labels):
        if label_id == 0:
            continue  # skip background
    
        gt_tunnel_mask = labels == label_id
            
        for segment_label_id in np.unique(labeled_prediction):
            if segment_label_id == 0:
                continue
            pred_tunnel_mask = labeled_prediction == segment_label_id

            overlap = np.sum((pred_tunnel_mask == True) & (gt_tunnel_mask == True))
            contains = overlap / np.sum(gt_tunnel_mask)

            if contains > 0.2:  # TODO: think about threshold
                _, best_yet_overlap = mapping.get(label_id, (-1, 0))
                if contains > best_yet_overlap:
                    mapping[label_id] = (segment_label_id, contains)
    
    # Find unmatched labels
    unmatched_labels = []
    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        if label_id not in mapping.keys():
            unmatched_labels.append(label_id)
    
    # Find unmatched predictions
    matched_predictions = set(seg_id for seg_id, _ in mapping.values())
    unmatched_predictions = []
    for seg_id in np.unique(labeled_prediction):
        if seg_id == 0:
            continue
        if seg_id not in matched_predictions:
            unmatched_predictions.append(seg_id)

    print(f"Matched {len(mapping)} tunnels ({len(mapping) / (len(mapping)+len(unmatched_labels))*100:.1f}%)")
    print(f"Unmatched ground truth tunnels: {len(unmatched_labels)}")
    print(f"Unmatched predictions: {len(unmatched_predictions)}")

    # === PREPARE BACKGROUND IMAGE ===
    # Normalize image to 0-255 range and convert to RGB
    if image.dtype == np.float32 or image.dtype == np.float64:
        # Normalize to 0-255
        img_min, img_max = image.min(), image.max()
        if img_max > img_min:
            image_normalized = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            image_normalized = np.zeros_like(image, dtype=np.uint8)
    else:
        image_normalized = image.astype(np.uint8)
    
    # Convert grayscale to RGB if needed
    if len(image_normalized.shape) == 3:
        # Already 3D, convert to RGB by stacking
        background_rgb = np.stack([image_normalized, image_normalized, image_normalized], axis=-1)
    else:
        # 2D image, need to handle 3D volume differently
        background_rgb = np.stack([image_normalized, image_normalized, image_normalized], axis=-1)

    # === ENHANCED VISUALIZATION WITH BACKGROUND ===
    
    # 1. Basic confusion matrix visualization with background
    basic_vis = background_rgb.copy()
    alpha = 0.7  # Transparency for overlay
    
    # Create colored overlays
    tp_color = np.array([255, 255, 255])  # TP: White
    fp_color = np.array([255, 0, 0])      # FP: Red
    fn_color = np.array([0, 0, 255])      # FN: Blue
    
    # Apply overlays with alpha blending
    tp_mask = (binary_filtered_pred == True) & (binary_gt == True)
    fp_mask = (binary_filtered_pred == True) & (binary_gt == False)
    fn_mask = (binary_filtered_pred == False) & (binary_gt == True)
    
    basic_vis[tp_mask] = (alpha * tp_color + (1-alpha) * basic_vis[tp_mask]).astype(np.uint8)
    basic_vis[fp_mask] = (alpha * fp_color + (1-alpha) * basic_vis[fp_mask]).astype(np.uint8)
    basic_vis[fn_mask] = (alpha * fn_color + (1-alpha) * basic_vis[fn_mask]).astype(np.uint8)
    
    tifffile.imwrite(str(output_folder / 'basic_confusion_matrix_overlay.tif'), basic_vis)

    # 2. Matched tunnels visualization with background
    matched_vis = background_rgb.copy()
    colors = [
        (255, 100, 100),  # Light red
        (100, 255, 100),  # Light green  
        (100, 100, 255),  # Light blue
        (255, 255, 100),  # Yellow
        (255, 100, 255),  # Magenta
        (100, 255, 255),  # Cyan
        (200, 150, 100),  # Brown
        (150, 100, 200),  # Purple
    ]
    
    for i, (gt_label, (pred_label, overlap_score)) in enumerate(mapping.items()):
        color = np.array(colors[i % len(colors)])
        
        # Color both GT and prediction with same color, but different intensities
        gt_mask = labels == gt_label
        pred_mask = labeled_prediction == pred_label
        
        # GT regions: blend with background
        matched_vis[gt_mask] = (0.6 * color + 0.4 * matched_vis[gt_mask]).astype(np.uint8)
        
        # Prediction regions: blend with different intensity
        pred_color = color * 0.7
        matched_vis[pred_mask] = (0.5 * pred_color + 0.5 * matched_vis[pred_mask]).astype(np.uint8)
        
        # Overlap regions: bright white overlay
        overlap_mask = gt_mask & pred_mask
        overlap_color = np.array([255, 255, 255])
        matched_vis[overlap_mask] = (0.8 * overlap_color + 0.2 * matched_vis[overlap_mask]).astype(np.uint8)
    
    tifffile.imwrite(str(output_folder / 'matched_tunnels_overlay.tif'), matched_vis)

    # 3. Unmatched regions visualization with background
    unmatched_vis = background_rgb.copy()
    
    # Unmatched GT tunnels in bright red
    unmatched_gt_color = np.array([255, 0, 0])
    for gt_label in unmatched_labels:
        gt_mask = labels == gt_label
        unmatched_vis[gt_mask] = (0.7 * unmatched_gt_color + 0.3 * unmatched_vis[gt_mask]).astype(np.uint8)
    
    # Unmatched predictions in bright blue
    unmatched_pred_color = np.array([0, 0, 255])
    for pred_label in unmatched_predictions:
        pred_mask = labeled_prediction == pred_label
        unmatched_vis[pred_mask] = (0.7 * unmatched_pred_color + 0.3 * unmatched_vis[pred_mask]).astype(np.uint8)
    
    tifffile.imwrite(str(output_folder / 'unmatched_regions_overlay.tif'), unmatched_vis)

    # 4. Combined overview visualization with background
    overview_vis = background_rgb.copy()
    
    # Matched GT in green (semi-transparent)
    matched_gt_color = np.array([0, 255, 0])
    for gt_label, _ in mapping.items():
        gt_mask = labels == gt_label
        overview_vis[gt_mask] = (0.5 * matched_gt_color + 0.5 * overview_vis[gt_mask]).astype(np.uint8)
    
    # Matched predictions in yellow (semi-transparent)
    matched_pred_color = np.array([255, 255, 0])
    for _, (pred_label, _) in mapping.items():
        pred_mask = labeled_prediction == pred_label
        current_color = overview_vis[pred_mask].astype(float)
        blended_color = 0.4 * matched_pred_color + 0.6 * current_color
        overview_vis[pred_mask] = np.clip(blended_color, 0, 255).astype(np.uint8)
    
    # Unmatched GT in magenta (more visible)
    unmatched_gt_color = np.array([255, 100, 255])
    for gt_label in unmatched_labels:
        gt_mask = labels == gt_label
        overview_vis[gt_mask] = (0.7 * unmatched_gt_color + 0.3 * overview_vis[gt_mask]).astype(np.uint8)
    
    # Unmatched predictions in blue (more visible)
    unmatched_pred_color = np.array([0, 0, 255])
    for pred_label in unmatched_predictions:
        pred_mask = labeled_prediction == pred_label
        overview_vis[pred_mask] = (0.7 * unmatched_pred_color + 0.3 * overview_vis[pred_mask]).astype(np.uint8)
    
    tifffile.imwrite(str(output_folder / 'overview_matching_overlay.tif'), overview_vis)

    # 5. Save original background for reference
    tifffile.imwrite(str(output_folder / 'original_image.tif'), background_rgb)

    # 6. Generate matching report (same as before)
    with open(output_folder / 'matching_report.txt', 'w') as f:
        f.write("Tunnel Matching Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total GT tunnels: {len(np.unique(labels)) - 1}\n")
        f.write(f"Total predicted regions: {len(np.unique(labeled_prediction)) - 1}\n")
        f.write(f"Matched tunnels: {len(mapping)}\n")
        f.write(f"Unmatched GT tunnels: {len(unmatched_labels)}\n")
        f.write(f"Unmatched predictions: {len(unmatched_predictions)}\n")
        f.write(f"Match rate: {len(mapping) / (len(mapping)+len(unmatched_labels))*100:.1f}%\n\n")
        
        f.write("Detailed Matches:\n")
        f.write("-" * 30 + "\n")
        for gt_label, (pred_label, overlap_score) in mapping.items():
            gt_size = np.sum(labels == gt_label)
            pred_size = np.sum(labeled_prediction == pred_label)
            f.write(f"GT {gt_label} -> Pred {pred_label}: {overlap_score:.2f} overlap "
                   f"(GT size: {gt_size}, Pred size: {pred_size})\n")
        
        f.write(f"\nUnmatched GT tunnels: {unmatched_labels}\n")
        f.write(f"Unmatched predictions: {unmatched_predictions}\n")

    print(f"Visualizations saved to {output_folder}")
    print("Files generated:")
    print("  - original_image.tif: Original image reference")
    print("  - basic_confusion_matrix_overlay.tif: TP/FP/FN over original image")
    print("  - matched_tunnels_overlay.tif: Matched pairs over original image")
    print("  - unmatched_regions_overlay.tif: Unmatched regions over original image")  
    print("  - overview_matching_overlay.tif: Complete matching overview over original image")
    print("  - matching_report.txt: Detailed matching statistics")

    return TunnelDetectionResult(
        tp=TP,
        fp=FP, 
        tn=TN,
        fn=FN,
        # spec=tntmetrics.specificity(TN, FP),
        spec=0,  # TODO: fix
        sens=recall,
        f1=f1,
        recall=recall,
        prec=prec
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Post-process tunnel predictions')
    parser.add_argument('prediction', type=Path, help='Path to prediction file')
    parser.add_argument('label', type=Path, help='Path to label file')
    parser.add_argument('input', type=Path, help="Path to the original img")
    parser.add_argument('output', type=Path)
    parser.add_argument('--min-size', type=int, default=100, help='Minimum object size in pixels')
    parser.add_argument('--se-size', type=int, default=3, help='Structuring element size')
    args = parser.parse_args()

    output_folder = Path(args.output)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Load the files
    prediction = tifffile.imread(args.prediction).astype(np.bool)
    label = tifffile.imread(args.label).astype(np.uint8)
    img = tifffile.imread(args.input)

    # Create config
    config = PostprocessConfig(
        minimum_size_px=args.min_size,
        se_type='disk',
        se_size=args.se_size, 
    )

    # Run postprocessing
    processed = detect_tunnels(prediction, label, img, config, output_folder=output_folder)