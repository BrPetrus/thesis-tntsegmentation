from dataclasses import dataclass
from typing import Any, List, Dict, Tuple
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
    recall_threshold: float
    prediction_threshold: float

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
    f1: float
    recall: float
    prec: float

def detect_tunnels(prediction: NDArray[np.float32], gt_labeled: NDArray[np.uint8], image: NDArray, config: PostprocessConfig, output_folder: Path) -> TunnelDetectionResult:
    output_folder.mkdir(parents=True, exist_ok=True)

    image = _preproces_raw_image(image) 
    binary_prediction = prediction > config.prediction_threshold
    regions_prop, labeled_prediction = post_process_output(binary_prediction, config)
    tifffile.imwrite(output_folder / 'processed_prediction.tif', labeled_prediction)
    tifffile.imwrite(output_folder / 'processed_prediction_binary.tif', labeled_prediction != 0)
    binary_filtered_pred = labeled_prediction != 0
    binary_gt = gt_labeled != 0

    # Calculate basic metrics
    TP = np.sum((binary_filtered_pred == True) & (binary_gt == True))
    TN = np.sum((binary_filtered_pred == False) & (binary_gt == False))
    FP = np.sum((binary_filtered_pred == True) & (binary_gt == False))
    FN = np.sum((binary_filtered_pred == False) & (binary_gt == True))
    prec = tntmetrics.precision(TP, FP)
    recall = tntmetrics.recall(TP, FN)
    f1 = tntmetrics.dice_coefficient(TP, FP, FN)
    jacc = tntmetrics.jaccard_index(TP, FP, FN)

    print(f"prec={prec}, recall={recall}, f1={f1}, jaccard={jacc}")

    # Tunnel matching logic
    mapping = map_tunnels_gt_to_pred(gt_labeled, labeled_prediction, config.recall_threshold)
    
    # Check for prediction labels that are mapped to multiple GT tunnels
    pred_to_gt_mapping = invert_mapping(mapping)
    
    # Find duplicates
    duplicate_predictions, unmatched_labels, unmatched_predictions = find_unmatched_and_duplicates(gt_labeled, labeled_prediction, mapping, pred_to_gt_mapping)
    
    # Report duplicates
    if duplicate_predictions:
        print(f"\n*** DUPLICATE MAPPINGS DETECTED ***")
        print(f"Found {len(duplicate_predictions)} prediction regions mapped to multiple GT tunnels:")
        for pred_label, gt_matches in duplicate_predictions.items():
            print(f"  Prediction {pred_label} matches:")
            for gt_label, overlap_score in gt_matches:
                gt_size = np.sum(gt_labeled == gt_label)
                pred_size = np.sum(labeled_prediction == pred_label)
                print(f"    -> GT {gt_label}: {overlap_score:.2f} overlap (GT size: {gt_size}, Pred size: {pred_size})")
    
    print(f"\n=== METRICS FOR ALL MATCHED TUNNELS")
    matched_metrics = calculate_matched_metrics(gt_labeled, labeled_prediction, mapping) 
    print(f" 1-1 matches: {matched_metrics['num_clean_matches']}/{len(mapping)} total matches")
    print(f"Matched Dice: {matched_metrics['matched_dice']:.4f}")
    print(f"Matched Jaccard: {matched_metrics['matched_jaccard']:.4f}")
    print(f"Matched Precision: {matched_metrics['matched_precision']:.4f}")
    print(f"Matched Recall: {matched_metrics['matched_recall']:.4f}")
    print(f"TP: {matched_metrics['TP_matched']}, FP: {matched_metrics['FP_matched']}, FN: {matched_metrics['FN_matched']}")

    print(f"Matched GT {len(mapping)} tunnels ({len(mapping) / (len(mapping)+len(unmatched_labels))*100:.1f}%)")
    print(f"Unmatched ground truth tunnels: {len(unmatched_labels)}")
    print(f"Unmatched predictions: {len(unmatched_predictions)}")
    
    # === CALCULATE METRICS ON CLEAN MAPPINGS ===
    clean_mapping = filter_clean_mappings(mapping, duplicate_predictions)
    matched_metrics = calculate_matched_metrics(gt_labeled, labeled_prediction, clean_mapping)
    individual_metrics = calculate_individual_tunnel_metrics(gt_labeled, labeled_prediction, clean_mapping)
    
    print(f"\n=== METRICS FOR 1-1 MATCHED TUNNELS ONLY ===")
    print(f"Clean 1-1 matches: {matched_metrics['num_clean_matches']}/{len(mapping)} total matches")
    print(f"Matched Dice: {matched_metrics['matched_dice']:.4f}")
    print(f"Matched Jaccard: {matched_metrics['matched_jaccard']:.4f}")
    print(f"Matched Precision: {matched_metrics['matched_precision']:.4f}")
    print(f"Matched Recall: {matched_metrics['matched_recall']:.4f}")
    print(f"TP: {matched_metrics['TP_matched']}, FP: {matched_metrics['FP_matched']}, FN: {matched_metrics['FN_matched']}")

    print(f"Matched GT {len(mapping)} tunnels ({len(mapping) / (len(mapping)+len(unmatched_labels))*100:.1f}%)")
    print(f"Unmatched ground truth tunnels: {len(unmatched_labels)}")
    print(f"Unmatched predictions: {len(unmatched_predictions)}")

    visualise_matching(gt_labeled, image, output_folder, labeled_prediction, binary_filtered_pred, binary_gt, mapping, duplicate_predictions, unmatched_labels, unmatched_predictions)

    return TunnelDetectionResult(
        tp=TP,
        fp=FP, 
        tn=TN,
        fn=FN,
        f1=f1,
        recall=recall,
        prec=prec
    )

def visualise_matching(gt_labeled, image, output_folder, labeled_prediction, binary_filtered_pred, binary_gt, mapping, duplicate_predictions, unmatched_labels, unmatched_predictions):
    if image.dtype == np.float32 or image.dtype == np.float64:
        # Normalize to 0-255
        img_min, img_max = image.min(), image.max()
        image_normalized = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        image_normalized = image.astype(np.uint8)
    
    # Convert grayscale to RGB if needed
    background_rgb = np.stack([image_normalized, image_normalized, image_normalized], axis=-1)

    # 1. Basic confusion matrix visualization with background
    basic_vis = background_rgb.copy()
    alpha = 0.7  # Transparency for overlay
    
    # Create colored overlays
    tp_color = np.array([0, 255, 0])  # TP: Green
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
        gt_mask = gt_labeled == gt_label
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
        gt_mask = gt_labeled == gt_label
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
        gt_mask = gt_labeled == gt_label
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
        gt_mask = gt_labeled == gt_label
        overview_vis[gt_mask] = (0.7 * unmatched_gt_color + 0.3 * overview_vis[gt_mask]).astype(np.uint8)
    
    # Unmatched predictions in blue (more visible)
    unmatched_pred_color = np.array([0, 0, 255])
    for pred_label in unmatched_predictions:
        pred_mask = labeled_prediction == pred_label
        overview_vis[pred_mask] = (0.7 * unmatched_pred_color + 0.3 * overview_vis[pred_mask]).astype(np.uint8)
    
    tifffile.imwrite(str(output_folder / 'overview_matching_overlay.tif'), overview_vis)

    # 5. NEW: Duplicate mappings visualization
    if duplicate_predictions:
        duplicate_vis = background_rgb.copy()
        
        # Use bright, distinctive colors for duplicates
        duplicate_colors = [
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Cyan  
            (255, 165, 0),    # Orange
            (255, 20, 147),   # Deep pink
            (50, 205, 50),    # Lime green
            (255, 69, 0),     # Red orange
        ]
        
        for i, (pred_label, gt_matches) in enumerate(duplicate_predictions.items()):
            color = np.array(duplicate_colors[i % len(duplicate_colors)])
            
            # Highlight the prediction region in the chosen color
            pred_mask = labeled_prediction == pred_label
            duplicate_vis[pred_mask] = (0.8 * color + 0.2 * duplicate_vis[pred_mask]).astype(np.uint8)
            
            # Highlight each GT region it matches with a slightly different shade
            for j, (gt_label, overlap_score) in enumerate(gt_matches):
                gt_mask = gt_labeled == gt_label
                # Use progressively darker shades for multiple GT matches
                shade_factor = 0.8 - (j * 0.2)  # 0.8, 0.6, 0.4, etc.
                gt_color = color * shade_factor
                duplicate_vis[gt_mask] = (0.6 * gt_color + 0.4 * duplicate_vis[gt_mask]).astype(np.uint8)
        
        tifffile.imwrite(str(output_folder / 'duplicate_mappings_overlay.tif'), duplicate_vis)
    
    # 6. Enhanced overview with duplicate highlighting
    overview_vis = background_rgb.copy()
    
    # # Regular matched GT in green (semi-transparent)
    # matched_gt_color = np.array([0, 255, 0])
    # regular_matches = {gt_label: pred_data for gt_label, pred_data in mapping.items() 
    #                   if pred_data[0] not in duplicate_predictions}
    
    # for gt_label, _ in regular_matches.items():
    #     gt_mask = labels == gt_label
    #     overview_vis[gt_mask] = (0.5 * matched_gt_color + 0.5 * overview_vis[gt_mask]).astype(np.uint8)
    
    # Duplicate matches in blue
    duplicate_gt_color = np.array([0, 0, 255])
    for gt_label, pred_data in mapping.items():
        if pred_data[0] in duplicate_predictions:
            gt_mask = gt_labeled == gt_label
            overview_vis[gt_mask] = (0.7 * duplicate_gt_color + 0.3 * overview_vis[gt_mask]).astype(np.uint8)
    
    # # Regular matched predictions in yellow
    # matched_pred_color = np.array([255, 255, 0])
    # for _, (pred_label, _) in regular_matches.items():
    #     pred_mask = labeled_prediction == pred_label
    #     current_color = overview_vis[pred_mask].astype(float)
    #     blended_color = 0.4 * matched_pred_color + 0.6 * current_color
    #     overview_vis[pred_mask] = np.clip(blended_color, 0, 255).astype(np.uint8)
    
    # Duplicate predictions in bright red
    duplicate_pred_color = np.array([255, 100, 100])
    for pred_label in duplicate_predictions.keys():
        pred_mask = labeled_prediction == pred_label
        overview_vis[pred_mask] = (0.8 * duplicate_pred_color + 0.2 * overview_vis[pred_mask]).astype(np.uint8)
    
    # # Unmatched GT in magenta
    # unmatched_gt_color = np.array([255, 100, 255])
    # for gt_label in unmatched_labels:
    #     gt_mask = labels == gt_label
    #     overview_vis[gt_mask] = (0.7 * unmatched_gt_color + 0.3 * overview_vis[gt_mask]).astype(np.uint8)
    
    # # Unmatched predictions in blue
    # unmatched_pred_color = np.array([0, 0, 255])
    # for pred_label in unmatched_predictions:
    #     pred_mask = labeled_prediction == pred_label
    #     overview_vis[pred_mask] = (0.7 * unmatched_pred_color + 0.3 * overview_vis[pred_mask]).astype(np.uint8)
    
    tifffile.imwrite(str(output_folder / 'enhanced_overview_overlay.tif'), overview_vis)

    # Original background for reference
    tifffile.imwrite(str(output_folder / 'original_image.tif'), background_rgb)

    # === ENHANCED MATCHING REPORT WITH DUPLICATES ===
    with open(output_folder / 'matching_report.txt', 'w') as f:
        f.write("Tunnel Matching Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total GT tunnels: {len(np.unique(gt_labeled)) - 1}\n")
        f.write(f"Total predicted regions: {len(np.unique(labeled_prediction)) - 1}\n")
        f.write(f"Matched tunnels: {len(mapping)}\n")
        f.write(f"Unmatched GT tunnels: {len(unmatched_labels)}\n")
        f.write(f"Unmatched predictions: {len(unmatched_predictions)}\n")
        f.write(f"Duplicate predictions: {len(duplicate_predictions)}\n")
        f.write(f"Match rate: {len(mapping) / (len(mapping)+len(unmatched_labels))*100:.1f}%\n\n")
        
        if duplicate_predictions:
            f.write("*** DUPLICATE MAPPINGS ***\n")
            f.write("-" * 30 + "\n")
            f.write(f"Found {len(duplicate_predictions)} prediction regions mapped to multiple GT tunnels:\n\n")
            for pred_label, gt_matches in duplicate_predictions.items():
                pred_size = np.sum(labeled_prediction == pred_label)
                f.write(f"Prediction {pred_label} (size: {pred_size}) matches:\n")
                for gt_label, overlap_score in gt_matches:
                    gt_size = np.sum(gt_labeled == gt_label)
                    f.write(f"  -> GT {gt_label}: {overlap_score:.2f} overlap (size: {gt_size})\n")
                f.write("\n")
        
        f.write("Regular Matches:\n")
        f.write("-" * 30 + "\n")
        for gt_label, (pred_label, overlap_score) in mapping.items():
            if pred_label not in duplicate_predictions:
                gt_size = np.sum(gt_labeled == gt_label)
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
    if duplicate_predictions:
        print("  - duplicate_mappings_overlay.tif: Duplicate mapping visualization")
    print("  - enhanced_overview_overlay.tif: Complete overview with duplicates highlighted")
    print("  - matching_report.txt: Detailed matching statistics")

def filter_clean_mappings(mapping: Dict[int, Tuple[int, float]], 
                         duplicate_predictions: Dict[int, List[Tuple[int, float]]]) -> Dict[int, Tuple[int, float]]:
    """Filter out mappings that involve duplicate predictions to get clean 1-1 mappings only."""
    clean_mapping = {gt_label: pred_data for gt_label, pred_data in mapping.items() 
                    if pred_data[0] not in duplicate_predictions}
    return clean_mapping

def calculate_matched_metrics(gt_labeled: NDArray[np.uint8], labeled_prediction: NDArray[np.uint8], 
                            mapping: Dict[int, Tuple[int, float]]) -> Dict[str, float]:
    """Calculate Dice and Jaccard scores only for matched tunnels"""
    
    # Create binary masks for only the matched regions
    matched_gt_mask = np.zeros_like(gt_labeled, dtype=bool)
    matched_pred_mask = np.zeros_like(labeled_prediction, dtype=bool)
    
    # Accumulate masks for all  matches
    for gt_label, (pred_label, overlap_score) in mapping.items():
        matched_gt_mask |= (gt_labeled == gt_label)
        matched_pred_mask |= (labeled_prediction == pred_label)
    
    # Calculate confusion matrix for matched regions only
    TP_matched = np.sum(matched_gt_mask & matched_pred_mask)
    FP_matched = np.sum(matched_pred_mask & ~matched_gt_mask)
    FN_matched = np.sum(matched_gt_mask & ~matched_pred_mask)
    
    # Calculate metrics
    dice_matched = tntmetrics.dice_coefficient(TP_matched, FP_matched, FN_matched)
    jaccard_matched = tntmetrics.jaccard_index(TP_matched, FP_matched, FN_matched)
    precision_matched = tntmetrics.precision(TP_matched, FP_matched)
    recall_matched = tntmetrics.recall(TP_matched, FN_matched)
    
    return {
        'matched_dice': dice_matched,
        'matched_jaccard': jaccard_matched,
        'matched_precision': precision_matched,
        'matched_recall': recall_matched,
        'num_clean_matches': len(mapping),
        'TP_matched': TP_matched,
        'FP_matched': FP_matched,
        'FN_matched': FN_matched
    }

def calculate_individual_tunnel_metrics(gt_labeled: NDArray[np.uint8], labeled_prediction: NDArray[np.uint8], 
                                       clean_mapping: Dict[int, Tuple[int, float]]) -> List[Dict]:
    """Calculate metrics for each individual matched tunnel pair."""
    
    individual_metrics = []
    
    for gt_label, (pred_label, overlap_score) in clean_mapping.items():
        # Get masks for this specific tunnel pair
        gt_mask = (gt_labeled == gt_label)
        pred_mask = (labeled_prediction == pred_label)
        
        # Calculate confusion matrix for this pair
        TP = np.sum(gt_mask & pred_mask)
        FP = np.sum(pred_mask & ~gt_mask)
        FN = np.sum(gt_mask & ~pred_mask)
        
        # Calculate metrics
        dice = tntmetrics.dice_coefficient(TP, FP, FN)
        jaccard = tntmetrics.jaccard_index(TP, FP, FN)
        precision = tntmetrics.precision(TP, FP)
        recall = tntmetrics.recall(TP, FN)
        
        gt_size = np.sum(gt_mask)
        pred_size = np.sum(pred_mask)
        
        individual_metrics.append({
            'gt_label': gt_label,
            'pred_label': pred_label,
            'dice': dice,
            'jaccard': jaccard,
            'precision': precision,
            'recall': recall,
            'overlap_score': overlap_score,
            'gt_size': gt_size,
            'pred_size': pred_size,
            'TP': TP,
            'FP': FP,
            'FN': FN
        })
    
    return individual_metrics

def find_unmatched_and_duplicates(gt_labeled, labeled_prediction, mapping, pred_to_gt_mapping):
    duplicate_predictions = {}
    for pred_label, gt_matches in pred_to_gt_mapping.items():
        if len(gt_matches) > 1:
            duplicate_predictions[pred_label] = gt_matches
    
    # Find unmatched labels
    unmatched_labels = []
    for label_id in np.unique(gt_labeled):
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
    return duplicate_predictions,unmatched_labels,unmatched_predictions

def invert_mapping(mapping):
    pred_to_gt_mapping = {}  # prediction_label -> list of (gt_label, overlap_score)
    for gt_label, (pred_label, overlap_score) in mapping.items():
        if pred_label not in pred_to_gt_mapping:
            pred_to_gt_mapping[pred_label] = []
        pred_to_gt_mapping[pred_label].append((gt_label, overlap_score))
    return pred_to_gt_mapping

def map_tunnels_gt_to_pred(labeled_gt: NDArray[np.uint8], labeled_prediction: NDArray[np.uint8], recall_threshold: float = 0.5) -> Dict[int, Tuple[int, int]]:
    mapping = {}
    for label_id_gt in np.unique(labeled_gt):
        if label_id_gt == 0:
            continue  # skip background
        gt_tunnel_mask = labeled_gt == label_id_gt
        for label_id_pred in np.unique(labeled_prediction):
            if label_id_pred == 0:
                continue
            pred_tunnel_mask = labeled_prediction == label_id_pred

            # calculate recall i.e. how much of th GT is inside the prediction
            overlap = np.sum((pred_tunnel_mask == True) & (gt_tunnel_mask == True))
            recall = overlap / np.sum(gt_tunnel_mask)

            if recall > recall_threshold:
                _, best_recall = mapping.get(label_id_gt, (-1, 0))
                if recall > best_recall:
                    mapping[label_id_gt] = (label_id_pred, recall)
    return mapping

def _preproces_raw_image(image):
    img_q3 = np.quantile(img, 0.03)
    img_q97 = np.quantile(img, 0.97)
    image = image.copy()
    image[image <= img_q3] = img_q3
    image[image >= img_q97] = img_q97
    image = (image - image.min()) / (image.max() - image.min())
    image = image *255
    image = image.astype(np.uint8)
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Post-process tunnel predictions')
    parser.add_argument('prediction', type=Path, help='Path to prediction file')
    parser.add_argument('label', type=Path, help='Path to label file')
    parser.add_argument('input', type=Path, help="Path to the original img")
    parser.add_argument('output', type=Path)
    parser.add_argument('--min-size', type=int, default=100, help='Minimum object size in pixels')
    parser.add_argument('--se-size', type=int, default=3, help='Structuring element size')
    parser.add_argument('--recall_threshold', type=float, default=0.6)  # TODO: help missing
    parser.add_argument('--prediction_threshold', type=float, default=0.8)
    args = parser.parse_args()

    output_folder = Path(args.output)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Load the files
    prediction = tifffile.imread(args.prediction)
    label = tifffile.imread(args.label).astype(np.uint8)
    img = tifffile.imread(args.input)

    # Create config
    config = PostprocessConfig(
        minimum_size_px=args.min_size,
        se_type='disk',
        se_size=args.se_size, 
        recall_threshold=args.recall_threshold,
        prediction_threshold=args.prediction_threshold,
    )

    # Run tunnel matching
    processed = detect_tunnels(prediction, label, img, config, output_folder=output_folder / 'ontunnels')
    
    # # Run tunnel matching on CC of the label
    # label_new = label.copy()
    # label_new = label_new != 0
    # label_new = skmorph.label(label_new, background=False)
    # tifffile.imwrite(output_folder/'cctunnels'/'cc_gt.tif', label_new)
    # processed_new = detect_tunnels(prediction, label_new, img, config, output_folder=output_folder / 'cctunnels')