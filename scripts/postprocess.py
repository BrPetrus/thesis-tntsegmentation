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

def detect_tunnels(predictions: NDArray[np.bool], labels: NDArray[np.uint8], config: PostprocessConfig, output_folder: Path) -> TunnelDetectionResult:
    regions_prop, labeled_prediction = post_process_output(predictions, config)
    binary_filtered_pred = labeled_prediction != 0
    binary_gt = labels != 0
    # # Create a mask of all regions
    # filtered_predictions_mask = np.zeros_like(predictions) 
    # for region in regions:
    #     filtered_predictions_mask[region['label'] == labels] = True
    
    # TODO: make the labels also binary and compare those


    TP = np.sum((binary_filtered_pred == True) & (binary_gt == True))
    TN = np.sum((binary_filtered_pred == False) & (binary_gt == False))
    FP = np.sum((binary_filtered_pred == True) & (binary_gt == False))
    FN = np.sum((binary_filtered_pred == False) & (binary_gt == True))

    prec = tntmetrics.precision(TP, FP)
    recall = tntmetrics.recall(TP, FN)
    f1 = tntmetrics.dice_coefficient(TP, FP, FN)

    print(f"prec={prec}, recall={recall}, f1={f1}")

    # Visualise this
    result_vis = np.zeros((predictions.shape[0], predictions.shape[1], predictions.shape[2], 3), np.uint8)
    result_vis[(binary_filtered_pred == True) & (labels != 0)] = (255, 0, 0)
    result_vis[(binary_filtered_pred == True) & (labels == 0)] = (0, 0, 255)
    result_vis[(binary_filtered_pred == False) & (labels != 0)] = (0, 255, 0)
    tifffile.imwrite(str(output_folder / 'visualisation.tif'), result_vis)

    # Now connect the tunnels from my prediction and the labels
    # - how to decide if they are connected?
    # - use labels?
    # 
    # idea: go through each labeled tunnel and see if there is an overlaping segmented region
    #       just 1-1 now ... so the overlap must be at least 80%

    # Matching the tunnels
    mapping = {}
    for label_id in np.unique(labels):
        if label_id == 0:
            continue  # skip background
    
        # labeled_region_mask = labels == label_id
        gt_tunnel_mask = labels == label_id
            
        for segment_label_id in np.unique(labeled_prediction):
            if segment_label_id == 0:
                continue
            pred_tunnel_mask = labeled_prediction == segment_label_id

            # # Calculate Jaccard
            # tp,fp,fn,tn = tntmetrics.calculate_batch_stats(

            # )  # TODO: modify

            # Find how much of the labeled tunnel is inside the segm.
            overlap = np.sum((pred_tunnel_mask == True) & (gt_tunnel_mask == True))
            contains = overlap / np.sum(gt_tunnel_mask)

            if contains > 0.2:  # TODO: think about threshold
                _, best_yet_overlap = mapping.get(label_id, (-1, 0))
                if contains > best_yet_overlap:
                    mapping[label_id] = (segment_label_id, contains)
    
    # Find how many tunnels got matched to something
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
    assert len(np.unique(labels)) - 1 == len(unmatched_labels) + len(mapping)

    # Confusion matrix
    



            # segmentation_label_mask
            # overlap = np.sum((label_id == labels) & (labeled_prediction == segment_label_id))
            # label_total = np.sum(label_id == labels)
            # segment_total = np.sum(labeled_prediction == segment_label_id)
            # overap_perc = overlap / ()
    




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Post-process tunnel predictions')
    parser.add_argument('prediction', type=Path, help='Path to prediction file')
    parser.add_argument('label', type=Path, help='Path to label file')
    parser.add_argument('output', type=Path)
    parser.add_argument('--min-size', type=int, default=100, help='Minimum object size in pixels')
    parser.add_argument('--se-size', type=int, default=3, help='Structuring element size')
    args = parser.parse_args()

    output_folder = Path(args.output)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Load the files
    prediction = tifffile.imread(args.prediction).astype(np.bool)
    label = tifffile.imread(args.label).astype(np.uint8)

    # Create config
    config = PostprocessConfig(
        minimum_size_px=args.min_size,
        se_type='disk',
        se_size=args.se_size, 
    )

    # Run postprocessing
    processed = detect_tunnels(prediction, label, config, output_folder=output_folder)