from dataclasses import dataclass
from numpy.typing import NDArray
from pathlib import Path
from typing import Any, List, Dict, Tuple
import argparse
import numpy as np
import skimage.measure as skmeas
import skimage.morphology as skmorph
import tifffile
import tntseg.utilities.metrics.metrics as tntmetrics


@dataclass(frozen=True)
class PostprocessConfig:
    minimum_size_px: int
    recall_threshold: float
    prediction_threshold: float


@dataclass(frozen=True)
class QualityMetrics:
    tp: int
    fp: int
    tn: int
    fn: int
    jaccard: float
    dice: float
    recall: float
    prec: float
    accuracy: float


@dataclass(frozen=True)
class TunnelMappingResult:
    """Results from tunnel mapping analysis."""

    mapping: Dict[int, Tuple[int, float]]  # gt_label -> (pred_label, overlap_score)
    clean_mapping: Dict[int, Tuple[int, float]]  # Only 1-to-1 mappings
    multimatched_predictions: Dict[
        int, List[Tuple[int, float]]
    ]  # pred_label -> [(gt_label, score), ...]
    unmatched_gt_labels: List[int]
    unmatched_predictions: List[int]


@dataclass(frozen=True)
class TunnelDetectionResult:
    """Complete tunnel detection results."""

    metrics_overall: QualityMetrics
    metrics_one_on_one: QualityMetrics
    metrics_all_matches: QualityMetrics
    metrics_on_tunnels: QualityMetrics
    mapping_result: TunnelMappingResult  # Add this


def post_process_output(
    predictions: NDArray[np.bool], config: PostprocessConfig
) -> List[Dict]:
    if not np.issubdtype(predictions.dtype, np.bool):
        raise ValueError("Expected a boolean array.")
    if predictions.ndim != 3:
        raise ValueError("Expected 3D data")

    # Find connected components
    labelled = skmorph.label(predictions, background=0)
    regions = skmeas.regionprops(labelled)

    # Filter small regions
    big_regions = []
    labeled_regions = np.zeros_like(labelled, dtype=np.uint8)
    for region in regions:
        if region["area"] < config.minimum_size_px:
            continue
        big_regions.append(region)
        labeled_regions[labelled == region["label"]] = region["label"]

    return big_regions, labeled_regions


def create_quality_metrics(tp: int, fp: int, tn: int, fn: int) -> QualityMetrics:
    if min([tp, fp, tn, fn]) < 0:
        raise ValueError("Arguments must be nonnegative!")
    return QualityMetrics(
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        recall=tntmetrics.recall(tp, fn),
        jaccard=tntmetrics.jaccard_index(tp, fp, fn),
        dice=tntmetrics.dice_coefficient(tp, fp, fn),
        prec=tntmetrics.precision(tp, fp),
        accuracy=tntmetrics.accuracy(tp, fp, fn, tn),
    )


def filter_just_1to1_mappings(
    mapping: Dict[int, Tuple[int, float]],
    multimatched_predictions: Dict[int, List[Tuple[int, float]]],
) -> Dict[int, Tuple[int, float]]:
    """
    Filter out mappings involving duplicate predictions to get clean 1-to-1 mappings.

    Parameters
    ----------
    mapping : Dict[int, Tuple[int, float]]
        Full mapping from GT labels to (prediction label, overlap score).
    multimatched_predictions : Dict[int, List[Tuple[int, float]]]
        Dictionary of prediction labels that match multiple GT tunnels.

    Returns
    -------
    Dict[int, Tuple[int, float]]
        Filtered mapping containing only 1-to-1 correspondences where each
        prediction matches exactly one GT tunnel and vice versa.
    """
    clean_mapping = {
        gt_label: pred_data
        for gt_label, pred_data in mapping.items()
        if pred_data[0] not in multimatched_predictions
    }
    return clean_mapping


def calculate_matched_metrics(
    gt_labeled: NDArray[np.uint8],
    labeled_prediction: NDArray[np.uint8],
    mapping: Dict[int, Tuple[int, float]],
) -> QualityMetrics:
    """
    Calculate pixel-level metrics for tunnels specified in the mapping.

    This function computes confusion matrix metrics (TP, FP, FN, TN) and derived
    quality metrics (Dice, Jaccard, precision, recall) considering only the pixels
    within matched tunnel regions.

    Parameters
    ----------
    gt_labeled : NDArray[np.uint8]
        3D array with labeled ground truth instances (0 = background).
    labeled_prediction : NDArray[np.uint8]
        3D array with labeled prediction instances (0 = background).
    mapping : Dict[int, Tuple[int, float]]
        Mapping from GT labels to (prediction label, overlap score) pairs.
        Only tunnels in this mapping are considered for metrics calculation.

    Returns
    -------
    QualityMetrics
        Quality metrics computed only for the matched regions, including:
        - TP: True positive pixels (overlap between matched GT and prediction)
        - FP: False positive pixels (prediction outside matched GT)
        - FN: False negative pixels (GT outside matched prediction)
        - TN: True negative pixels (background correctly identified)
        - Derived metrics: Dice, Jaccard, precision, recall, accuracy
    """
    # Create binary masks for only the matched regions
    matched_gt_mask = np.zeros_like(gt_labeled, dtype=bool)
    matched_pred_mask = np.zeros_like(labeled_prediction, dtype=bool)

    # Accumulate masks for all matches
    for gt_label, (pred_label, overlap_score) in mapping.items():
        matched_gt_mask |= gt_labeled == gt_label
        matched_pred_mask |= labeled_prediction == pred_label

    # Calculate confusion matrix for matched regions only
    TP = np.sum(matched_gt_mask & matched_pred_mask)
    FP = np.sum(matched_pred_mask & ~matched_gt_mask)
    FN = np.sum(matched_gt_mask & ~matched_pred_mask)
    TN = np.sum(~matched_gt_mask & ~matched_pred_mask)

    return create_quality_metrics(TP, FP, TN, FN)


def find_unmatched_and_multi_mapped(
    gt_labeled, labeled_prediction, mapping, pred_to_gt_mapping
):
    """
    Identify predictions matching multiple GT tunnels and unmatched regions.

    Analyzes the bidirectional mapping between ground truth and predictions to find:
    1. Predictions that overlap with multiple GT tunnels (over-segmentation)
    2. GT tunnels with no matching prediction (false negatives)
    3. Predictions with no matching GT tunnel (false positives)

    Parameters
    ----------
    gt_labeled : NDArray[np.uint8]
        3D array with labeled ground truth instances.
    labeled_prediction : NDArray[np.uint8]
        3D array with labeled prediction instances.
    mapping : Dict[int, Tuple[int, float]]
        GT-to-prediction mapping (GT label -> (pred label, overlap score)).
    pred_to_gt_mapping : Dict[int, List[Tuple[int, float]]]
        Prediction-to-GT mapping (pred label -> [(GT label, overlap score), ...]).

    Returns
    -------
    multi_mapped_predictions : Dict[int, List[Tuple[int, float]]]
        Prediction labels mapping to multiple GT tunnels with their overlap scores.
    unmatched_labels : List[int]
        GT labels that have no matching prediction.
    unmatched_predictions : List[int]
        Prediction labels that don't match any GT tunnel.
    """
    multi_mapped_predictions = {}
    for pred_label, gt_matches in pred_to_gt_mapping.items():
        if len(gt_matches) > 1:
            multi_mapped_predictions[pred_label] = gt_matches

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
    return multi_mapped_predictions, unmatched_labels, unmatched_predictions


def invert_mapping(mapping):
    """
    Invert GT-to-prediction mapping to create prediction-to-GT mapping.

    Converts a one-to-one or many-to-one mapping (GT -> prediction) into a
    one-to-many mapping (prediction -> [GT tunnels]) to identify cases where
    a single prediction matches multiple GT tunnels.

    Parameters
    ----------
    mapping : Dict[int, Tuple[int, float]]
        GT-to-prediction mapping (GT label -> (pred label, overlap score)).

    Returns
    -------
    Dict[int, List[Tuple[int, float]]]
        Prediction-to-GT mapping where each prediction label maps to a list
        of (GT label, overlap score) tuples.
    """
    pred_to_gt_mapping = {}  # prediction_label -> list of (gt_label, overlap_score)
    for gt_label, (pred_label, overlap_score) in mapping.items():
        if pred_label not in pred_to_gt_mapping:
            pred_to_gt_mapping[pred_label] = []
        pred_to_gt_mapping[pred_label].append((gt_label, overlap_score))
    return pred_to_gt_mapping


def map_tunnels_gt_to_pred(
    labeled_gt: NDArray[np.uint8],
    labeled_prediction: NDArray[np.uint8],
    recall_threshold: float = 0.5,
) -> Dict[int, Tuple[int, int]]:
    """
    Map ground truth tunnels to predicted tunnels based on overlap.

    For each GT tunnel, finds the prediction with the highest recall (i.e., the
    prediction that covers the most of the GT tunnel). Only predictions with
    recall above the threshold are considered matches.

    Parameters
    ----------
    labeled_gt : NDArray[np.uint8]
        3D array with labeled ground truth instances (0 = background).
    labeled_prediction : NDArray[np.uint8]
        3D array with labeled prediction instances (0 = background).
    recall_threshold : float, default=0.5
        Minimum fraction of GT tunnel that must be covered by a prediction
        to consider it a match. Range: [0, 1].

    Returns
    -------
    Dict[int, Tuple[int, float]]
        Mapping from GT label to (prediction label, recall score) for the
        best matching prediction. Only includes matches above threshold.

    Notes
    -----
    - Recall is defined as: (overlap pixels) / (total GT pixels)
    - Each GT tunnel maps to at most one prediction (the one with highest recall)
    - Multiple GT tunnels may map to the same prediction (many-to-one mapping)
    """
    mapping = {}
    for label_id_gt in np.unique(labeled_gt):
        if label_id_gt == 0:
            continue  # skip background
        gt_tunnel_mask = labeled_gt == label_id_gt
        for label_id_pred in np.unique(labeled_prediction):
            if label_id_pred == 0:
                continue
            pred_tunnel_mask = labeled_prediction == label_id_pred

            # calculate recall i.e. how much of the GT is inside the prediction
            overlap = np.sum((pred_tunnel_mask == True) & (gt_tunnel_mask == True))
            recall = overlap / np.sum(gt_tunnel_mask)

            if recall > recall_threshold:
                _, best_recall = mapping.get(label_id_gt, (-1, 0))
                if recall > best_recall:
                    mapping[label_id_gt] = (label_id_pred, recall)
    return mapping


def percentile_stretch(image, low_quantile=0.03, high_quantile=0.97):
    img_q3 = np.quantile(image, low_quantile)
    img_q97 = np.quantile(image, high_quantile)
    image = image.copy()
    image[image <= img_q3] = img_q3
    image[image >= img_q97] = img_q97
    image = (image - image.min()) / (image.max() - image.min())
    image = image * 255
    image = image.astype(np.uint8)
    return image


def analyze_tunnel_mappings(
    gt_labeled: NDArray[np.uint8],
    labeled_prediction: NDArray[np.uint8],
    recall_threshold: float,
) -> TunnelMappingResult:
    """
    Comprehensive analysis of mappings between ground truth and predicted tunnels.

    Performs bidirectional mapping analysis to identify matched, unmatched, and
    ambiguous tunnel correspondences. This is essential for understanding both
    correct detections and different types of errors.

    Parameters
    ----------
    gt_labeled : NDArray[np.uint8]
        3D array with labeled ground truth instances (0 = background).
    labeled_prediction : NDArray[np.uint8]
        3D array with labeled prediction instances (0 = background).
    recall_threshold : float
        Minimum recall threshold for considering a match. Range: [0, 1].

    Returns
    -------
    TunnelMappingResult
        Complete mapping analysis containing:
        - mapping : Full GT-to-prediction mapping
        - clean_mapping : Only 1-to-1 correspondences
        - multimatched_predictions : Predictions matching multiple GT tunnels
        - unmatched_gt_labels : GT tunnels with no prediction (false negatives)
        - unmatched_predictions : Predictions with no GT match (false positives)

    Notes
    -----
    The analysis reveals three types of mapping issues:
    1. Missed detections: GT tunnels with no matching prediction
    2. False positives: Predictions with no matching GT tunnel
    3. Over-segmentation: Single prediction matching multiple GT tunnels

    Examples
    --------
    >>> result = analyze_tunnel_mappings(gt, pred, recall_threshold=0.6)
    >>> print(f"Found {len(result.mapping)} matches")
    >>> print(f"Clean 1-1 matches: {len(result.clean_mapping)}")
    >>> print(f"Multi-matches: {len(result.multimatched_predictions)}")
    """
    # Create initial mapping
    mapping = map_tunnels_gt_to_pred(gt_labeled, labeled_prediction, recall_threshold)

    # Analyze mapping quality
    pred_to_gt_mapping = invert_mapping(mapping)
    multimatched_pred, unmatched_labels, unmatched_predictions = (
        find_unmatched_and_multi_mapped(
            gt_labeled, labeled_prediction, mapping, pred_to_gt_mapping
        )
    )

    # Create clean 1-to-1 mappings
    clean_mapping = filter_just_1to1_mappings(mapping, multimatched_pred)

    # Optional: Report multi-matches
    if multimatched_pred:
        print(
            f"Found {len(multimatched_pred)} prediction regions mapped to multiple GT tunnels:"
        )
        for pred_label, gt_matches in multimatched_pred.items():
            print(f"  Prediction {pred_label} matches:")
            for gt_label, overlap_score in gt_matches:
                gt_size = np.sum(gt_labeled == gt_label)
                pred_size = np.sum(labeled_prediction == pred_label)
                print(
                    f"    -> GT {gt_label}: {overlap_score:.2f} overlap (GT size: {gt_size}, Pred size: {pred_size})"
                )

    return TunnelMappingResult(
        mapping=mapping,
        clean_mapping=clean_mapping,
        multimatched_predictions=multimatched_pred,
        unmatched_gt_labels=unmatched_labels,
        unmatched_predictions=unmatched_predictions,
    )


def calculate_all_metrics(
    gt_labeled: NDArray[np.uint8],
    labeled_prediction: NDArray[np.uint8],
    binary_gt: NDArray[np.bool_],
    binary_pred: NDArray[np.bool_],
    mapping_result: TunnelMappingResult,
) -> Tuple[QualityMetrics, QualityMetrics, QualityMetrics, QualityMetrics]:
    """
    Calculate comprehensive metrics at different granularities.

    Computes four sets of quality metrics to evaluate tunnel detection performance
    from different perspectives: overall pixel accuracy, matched region quality,
    clean 1-to-1 match quality, and instance-level detection accuracy.

    Parameters
    ----------
    gt_labeled : NDArray[np.uint8]
        3D array with labeled ground truth instances (0 = background).
    labeled_prediction : NDArray[np.uint8]
        3D array with labeled prediction instances (0 = background).
    binary_gt : NDArray[np.bool_]
        Binary ground truth mask (True = tunnel, False = background).
    binary_pred : NDArray[np.bool_]
        Binary prediction mask (True = tunnel, False = background).
    mapping_result : TunnelMappingResult
        Complete mapping analysis from analyze_tunnel_mappings().

    Returns
    -------
    overall_metrics : QualityMetrics
        Pixel-level metrics across the entire volume, treating all pixels equally.
        Reflects overall segmentation quality regardless of instance identity.
    matched_metrics : QualityMetrics
        Pixel-level metrics computed only within all matched tunnel regions.
        Shows how well the model segments tunnels it successfully detects.
    clean_metrics : QualityMetrics
        Pixel-level metrics for only 1-to-1 matched tunnels, excluding ambiguous
        cases. Represents the quality of unambiguous, correctly detected tunnels.
    tunnel_metrics : QualityMetrics
        Instance-level detection metrics treating each tunnel as a single unit.
        TP = detected tunnels, FP = false detections, FN = missed tunnels.
        Reflects detection capability independent of segmentation quality.

    Notes
    -----
    The four metric types serve different purposes:

    1. Overall: Global segmentation performance, useful for pixel-wise comparisons
    2. Matched: Quality of detected regions, excluding missed/false detections
    3. Clean: Best-case quality on unambiguous matches
    4. Tunnel: Detection rate as a classification task (detected vs missed)

    Examples
    --------
    >>> overall, matched, clean, tunnel = calculate_all_metrics(
    ...     gt_labeled, pred_labeled, binary_gt, binary_pred, mapping_result
    ... )
    >>> print(f"Overall Dice: {overall.dice:.3f}")
    >>> print(f"Detection rate: {tunnel.recall:.3f}")
    >>> print(f"Clean match quality: {clean.dice:.3f}")
    """
    # Overall pixel-level metrics
    overall_metrics = create_quality_metrics(
        tp=np.sum(binary_pred & binary_gt),
        tn=np.sum(~binary_pred & ~binary_gt),
        fp=np.sum(binary_pred & ~binary_gt),
        fn=np.sum(~binary_pred & binary_gt),
    )

    # Metrics for all matched tunnels
    matched_metrics = calculate_matched_metrics(
        gt_labeled, labeled_prediction, mapping_result.mapping
    )

    # Metrics for clean 1-to-1 mappings only
    clean_metrics = calculate_matched_metrics(
        gt_labeled, labeled_prediction, mapping_result.clean_mapping
    )

    # Tunnel-level detection metrics
    tunnel_metrics = create_quality_metrics(
        tp=len(mapping_result.mapping),
        fp=len(mapping_result.unmatched_predictions),
        tn=0,  # We are not considering any background component
        fn=len(mapping_result.unmatched_gt_labels),
    )

    return overall_metrics, matched_metrics, clean_metrics, tunnel_metrics


def detect_tunnels(
    prediction: NDArray[np.float32],
    gt_labeled: NDArray[np.uint8] | None,
    image: NDArray,
    config: PostprocessConfig,
    output_folder: Path,
    visualise: bool = False,
) -> TunnelDetectionResult | None:
    """
    Main tunnel detection pipeline with optional ground truth comparison.

    Processes predicted probability maps to detect tunnel-like structures,
    optionally comparing them against ground truth labels to compute accuracy metrics.

    Parameters
    ----------
    prediction : NDArray[np.float32]
        3D array of prediction probabilities in range [0, 1].
    gt_labeled : NDArray[np.uint8] or None
        3D array of labeled ground truth instances. If None, only post-processing
        is performed without evaluation.
    image : NDArray
        3D array of original microscopy image for visualization.
    config : PostprocessConfig
        Configuration containing:
        - prediction_threshold : float
            Threshold for binarizing predictions.
        - minimum_size_px : int
            Minimum object size in pixels to keep after filtering.
        - recall_threshold : float
            Minimum overlap threshold for tunnel matching.
    output_folder : Path
        Directory to save intermediate results and visualizations.
    visualise : bool, default=False
        If True, generate visualization images of the results.

    Returns
    -------
    TunnelDetectionResult or None
        Complete detection results containing all metrics and mapping information
        if gt_labeled is provided, otherwise None. Intermediate post-processed
        results are always saved to disk.

        The result includes:
        - metrics_overall : Overall pixel-level segmentation metrics
        - metrics_all_matches : Metrics for all matched tunnel regions
        - metrics_one_on_one : Metrics for clean 1-to-1 matches only
        - metrics_on_tunnels : Instance-level detection metrics
        - mapping_result : Complete mapping analysis (matches, errors, etc.)

    Notes
    -----
    The pipeline consists of the following steps:

    1. Preprocessing: Stretch image contrast, threshold predictions
    2. Post-processing: Remove small objects, create labeled instances
    3. Evaluation (if GT provided): Map predictions to GT, calculate metrics
    4. Visualization (if enabled): Create overlay images showing matches/errors

    Intermediate files saved to output_folder:
    - processed_prediction.tif : Labeled prediction instances
    - processed_prediction_binary.tif : Binary prediction mask
    - confusion_matrix_overlay.tif : TP/FP/FN visualization (if visualise=True)
    - matched_tunnels_overlay.tif : Color-coded matches (if visualise=True)
    - unmatched_regions_overlay.tif : Missed/false detections (if visualise=True)
    - duplicate_mappings_overlay.tif : Over-segmentation cases (if visualise=True)

    Examples
    --------
    >>> config = PostprocessConfig(
    ...     prediction_threshold=0.5,
    ...     minimum_size_px=100,
    ...     recall_threshold=0.6
    ... )
    >>> result = detect_tunnels(
    ...     prediction=pred_array,
    ...     gt_labeled=gt_array,
    ...     image=img_array,
    ...     config=config,
    ...     output_folder=Path("results/"),
    ...     visualise=True
    ... )
    >>> if result:
    ...     print(f"Overall Dice: {result.metrics_overall.dice:.3f}")
    ...     print(f"Detected: {result.metrics_on_tunnels.tp} tunnels")

    See Also
    --------
    analyze_tunnel_mappings : Performs the GT-to-prediction mapping
    calculate_all_metrics : Computes the four metric sets
    visualise_mapping_results : Creates visualization overlays
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    # Preprocessing
    image = percentile_stretch(image)
    binary_prediction = prediction > config.prediction_threshold
    _, labeled_prediction = post_process_output(binary_prediction, config)

    # Save intermediate results
    tifffile.imwrite(output_folder / "processed_prediction.tif", labeled_prediction)
    tifffile.imwrite(
        output_folder / "processed_prediction_binary.tif", labeled_prediction != 0
    )

    # If no ground truth provided, just return after saving post-processed results
    if gt_labeled is None:
        print(
            "No ground truth provided - post-processing complete, skipping evaluation."
        )
        return None

    binary_filtered_pred = labeled_prediction != 0
    binary_gt = gt_labeled != 0

    # Analyze tunnel mappings
    mapping_result = analyze_tunnel_mappings(
        gt_labeled, labeled_prediction, config.recall_threshold
    )

    # Calculate all metrics
    overall_metrics, matched_metrics, clean_metrics, tunnel_metrics = (
        calculate_all_metrics(
            gt_labeled,
            labeled_prediction,
            binary_gt,
            binary_filtered_pred,
            mapping_result,
        )
    )

    # Print basic metrics
    print("Overall metrics:")
    print(
        f"prec={overall_metrics.prec:.3f}, recall={overall_metrics.recall:.3f}, "
        f"jaccard={overall_metrics.jaccard:.3f}, dice={overall_metrics.dice:.3f}"
    )

    # Visualization
    if visualise:
        visualise_mapping_results(
            gt_labeled,
            image,
            output_folder,
            labeled_prediction,
            binary_filtered_pred,
            binary_gt,
            mapping_result,
        )

    return TunnelDetectionResult(
        metrics_overall=overall_metrics,
        metrics_one_on_one=clean_metrics,
        metrics_all_matches=matched_metrics,
        metrics_on_tunnels=tunnel_metrics,
        mapping_result=mapping_result,
    )


def visualise_matching(
    gt_labeled,
    image,
    output_folder,
    labeled_prediction,
    binary_filtered_pred,
    binary_gt,
    mapping,
    duplicate_predictions,
    unmatched_labels,
    unmatched_predictions,
):
    """Restore the full visualization function that was accidentally removed."""
    if image.dtype == np.float32 or image.dtype == np.float64:
        # Normalize to 0-255
        img_min, img_max = image.min(), image.max()
        image_normalized = ((image - img_min) / (img_max - img_min) * 255).astype(
            np.uint8
        )
    else:
        image_normalized = image.astype(np.uint8)

    # Convert grayscale to RGB if needed
    background_rgb = np.stack(
        [image_normalized, image_normalized, image_normalized], axis=-1
    )

    # 1. Basic confusion matrix visualization with background
    basic_vis = background_rgb.copy()
    alpha = 0.7  # Transparency for overlay

    # Create colored overlays
    tp_color = np.array([0, 255, 0])  # TP: Green
    fp_color = np.array([255, 0, 0])  # FP: Red
    fn_color = np.array([0, 0, 255])  # FN: Blue

    # Apply overlays with alpha blending
    tp_mask = (binary_filtered_pred == True) & (binary_gt == True)
    fp_mask = (binary_filtered_pred == True) & (binary_gt == False)
    fn_mask = (binary_filtered_pred == False) & (binary_gt == True)

    basic_vis[tp_mask] = (alpha * tp_color + (1 - alpha) * basic_vis[tp_mask]).astype(
        np.uint8
    )
    basic_vis[fp_mask] = (alpha * fp_color + (1 - alpha) * basic_vis[fp_mask]).astype(
        np.uint8
    )
    basic_vis[fn_mask] = (alpha * fn_color + (1 - alpha) * basic_vis[fn_mask]).astype(
        np.uint8
    )

    tifffile.imwrite(str(output_folder / "confusion_matrix_overlay.tif"), basic_vis)

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
        matched_vis[gt_mask] = (0.6 * color + 0.4 * matched_vis[gt_mask]).astype(
            np.uint8
        )

        # Prediction regions: blend with different intensity
        pred_color = color * 0.7
        matched_vis[pred_mask] = (
            0.5 * pred_color + 0.5 * matched_vis[pred_mask]
        ).astype(np.uint8)

        # Overlap regions: bright white overlay
        overlap_mask = gt_mask & pred_mask
        overlap_color = np.array([255, 255, 255])
        matched_vis[overlap_mask] = (
            0.8 * overlap_color + 0.2 * matched_vis[overlap_mask]
        ).astype(np.uint8)

    tifffile.imwrite(str(output_folder / "matched_tunnels_overlay.tif"), matched_vis)

    # 3. Unmatched regions visualization with background
    unmatched_vis = background_rgb.copy()

    # Unmatched GT tunnels in bright red
    unmatched_gt_color = np.array([255, 0, 0])
    for gt_label in unmatched_labels:
        gt_mask = gt_labeled == gt_label
        unmatched_vis[gt_mask] = (
            0.7 * unmatched_gt_color + 0.3 * unmatched_vis[gt_mask]
        ).astype(np.uint8)

    # Unmatched predictions in bright blue
    unmatched_pred_color = np.array([0, 0, 255])
    for pred_label in unmatched_predictions:
        pred_mask = labeled_prediction == pred_label
        unmatched_vis[pred_mask] = (
            0.7 * unmatched_pred_color + 0.3 * unmatched_vis[pred_mask]
        ).astype(np.uint8)

    tifffile.imwrite(
        str(output_folder / "unmatched_regions_overlay.tif"), unmatched_vis
    )

    # 4. Multimatch mappings visualization
    if duplicate_predictions:
        duplicate_vis = background_rgb.copy()

        # Use bright, distinctive colors for duplicates
        duplicate_colors = [
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 165, 0),  # Orange
            (255, 20, 147),  # Deep pink
            (50, 205, 50),  # Lime green
            (255, 69, 0),  # Red orange
        ]

        for i, (pred_label, gt_matches) in enumerate(duplicate_predictions.items()):
            color = np.array(duplicate_colors[i % len(duplicate_colors)])

            # Highlight the prediction region in the chosen color
            pred_mask = labeled_prediction == pred_label
            duplicate_vis[pred_mask] = (
                0.8 * color + 0.2 * duplicate_vis[pred_mask]
            ).astype(np.uint8)

            # Highlight each GT region it matches with a slightly different shade
            for j, (gt_label, overlap_score) in enumerate(gt_matches):
                gt_mask = gt_labeled == gt_label
                # Use progressively darker shades for multiple GT matches
                shade_factor = 0.8 - (j * 0.2)  # 0.8, 0.6, 0.4, etc.
                gt_color = color * shade_factor
                duplicate_vis[gt_mask] = (
                    0.6 * gt_color + 0.4 * duplicate_vis[gt_mask]
                ).astype(np.uint8)

        tifffile.imwrite(
            str(output_folder / "duplicate_mappings_overlay.tif"), duplicate_vis
        )

    # Original background for reference
    tifffile.imwrite(str(output_folder / "original_image.tif"), background_rgb)


def visualise_mapping_results(
    gt_labeled,
    image,
    output_folder,
    labeled_prediction,
    binary_filtered_pred,
    binary_gt,
    mapping_result: TunnelMappingResult,
):
    """Updated visualization function that uses the mapping result."""
    visualise_matching(
        gt_labeled,
        image,
        output_folder,
        labeled_prediction,
        binary_filtered_pred,
        binary_gt,
        mapping_result.mapping,
        mapping_result.multimatched_predictions,
        mapping_result.unmatched_gt_labels,
        mapping_result.unmatched_predictions,
    )


def print_detailed_results(result: TunnelDetectionResult):
    """Clean function to print all results."""
    mapping = result.mapping_result.mapping
    clean_mapping = result.mapping_result.clean_mapping
    unmatched_labels = result.mapping_result.unmatched_gt_labels
    unmatched_predictions = result.mapping_result.unmatched_predictions

    print("\n=== OVERALL PIXEL-LEVEL METRICS ===")
    print(f"Dice: {result.metrics_overall.dice:.4f}")
    print(f"Jaccard: {result.metrics_overall.jaccard:.4f}")
    print(f"Precision: {result.metrics_overall.prec:.4f}")
    print(f"Recall: {result.metrics_overall.recall:.4f}")

    print("\n=== METRICS FOR ALL MATCHED TUNNELS ===")
    print(f"Matches: {len(mapping)} total matches")
    print(f"Matched Dice: {result.metrics_all_matches.dice:.4f}")
    print(f"Matched Jaccard: {result.metrics_all_matches.jaccard:.4f}")
    print(f"Matched Precision: {result.metrics_all_matches.prec:.4f}")
    print(f"Matched Recall: {result.metrics_all_matches.recall:.4f}")
    print(
        f"TP: {result.metrics_all_matches.tp}, FP: {result.metrics_all_matches.fp}, FN: {result.metrics_all_matches.fn}"
    )

    print("\n=== METRICS FOR 1-1 MATCHED TUNNELS ONLY ===")
    print(f"Clean 1-1 matches: {len(clean_mapping)}/{len(mapping)} total matches")
    print(f"Matched Dice: {result.metrics_one_on_one.dice:.4f}")
    print(f"Matched Jaccard: {result.metrics_one_on_one.jaccard:.4f}")
    print(f"Matched Precision: {result.metrics_one_on_one.prec:.4f}")
    print(f"Matched Recall: {result.metrics_one_on_one.recall:.4f}")
    print(
        f"TP: {result.metrics_one_on_one.tp}, FP: {result.metrics_one_on_one.fp}, FN: {result.metrics_one_on_one.fn}"
    )

    print("\n=== TUNNEL-LEVEL DETECTION METRICS ===")
    print(f"Detected tunnels: {result.metrics_on_tunnels.tp}")
    print(f"Missed tunnels: {result.metrics_on_tunnels.fn}")
    print(f"False positive tunnels: {result.metrics_on_tunnels.fp}")
    print(f"Tunnel detection precision: {result.metrics_on_tunnels.prec:.4f}")
    print(f"Tunnel detection recall: {result.metrics_on_tunnels.recall:.4f}")

    print(
        f"\nMatched GT {len(mapping)} tunnels ({len(mapping) / (len(mapping) + len(unmatched_labels)) * 100:.1f}%)"
    )
    print(f"Unmatched ground truth tunnels: {len(unmatched_labels)}")
    print(f"Unmatched predictions: {len(unmatched_predictions)}")


# Updated main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process tunnel predictions")
    parser.add_argument("prediction", type=Path, help="Path to prediction file")
    parser.add_argument("label", type=Path, help="Path to label file")
    parser.add_argument("input", type=Path, help="Path to the original img")
    parser.add_argument("output", type=Path, help="Path to output directory")
    parser.add_argument(
        "--min-size", type=int, default=100, help="Minimum object size in pixels"
    )
    parser.add_argument(
        "--recall_threshold",
        type=float,
        default=0.5,
        help="Recall threshold for tunnel matching",
    )
    parser.add_argument(
        "--prediction_threshold",
        type=float,
        default=0.5,
        help="Threshold for binary prediction",
    )
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
        recall_threshold=args.recall_threshold,
        prediction_threshold=args.prediction_threshold,
    )

    # Run tunnel detection
    result = detect_tunnels(
        prediction,
        label,
        img,
        config,
        output_folder=output_folder / "ontunnels",
        visualise=True,
    )

    # Print detailed results
    print_detailed_results(result)
