"""
metrics.py

This module provides functions for calculating various metrics commonly used in binary classification 
and segmentation tasks. These metrics include Jaccard Index, Dice Coefficient, Tversky Index, and 
Focal Tversky Loss. Additionally, it includes a utility function for calculating batch statistics 
such as true positives, false positives, false negatives, and true negatives.
""" 

import numpy as np
from numpy.typing import NDArray
from typing import TypeVar, Tuple

Stats = Tuple[int, int, int, int]
def jaccard_index(tp: int, fp: int, fn: int) -> float:
    """
    Compute the Jaccard Index (Intersection over Union).

    Args:
        tp (int): True positives.
        fp (int): False positives.
        fn (int): False negatives.

    Returns:
        float: Jaccard Index.
    """
    return tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0


def dice_coefficient(tp: int, fp: int, fn: int) -> float:
    """
    Compute the Dice Coefficient (F1 Score).

    Args:
        tp (int): True positives.
        fp (int): False positives.
        fn (int): False negatives.

    Returns:
        float: Dice Coefficient.
    """
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0.0


def tversky_index(tp: int, fp: int, fn: int, alpha: float = 0.5, beta: float = 0.5) -> float:
    """
    Compute the Tversky Index.

    Args:
        tp (int): True positives.
        fp (int): False positives.
        fn (int): False negatives.
        alpha (float): Weight for false positives. Defaults to 0.5.
        beta (float): Weight for false negatives. Defaults to 0.5.

    Returns:
        float: Tversky Index.
    """
    return tp / (tp + alpha * fp + beta * fn) if tp + alpha * fp + beta * fn > 0 else 0.0


def focal_tversky_loss(tp: int, fp: int, fn: int, alpha: float = 0.5, beta: float = 0.5, gamma: float = 1.0) -> float:
    """
    Compute the Focal Tversky Loss.

    Args:
        tp (int): True positives.
        fp (int): False positives.
        fn (int): False negatives.
        alpha (float): Weight for false positives. Defaults to 0.5.
        beta (float): Weight for false negatives. Defaults to 0.5.
        gamma (float): Focusing parameter. Defaults to 1.0.

    Returns:
        float: Focal Tversky Loss.
    """
    tversky = tversky_index(tp, fp, fn, alpha, beta)
    return (1 - tversky) ** gamma

def accuracy(tp: int, fp: int, fn: int, tn: int) -> float:
    return (tp + tn) / (tp+fp+fn+tn)
def precision(tp: int, fp: int) -> float:
    return (tp) / (tp+fp) if tp+fp > 0 else 0.
def recall(tp: int, fn: int) -> float:
    return tp / (tp+fn) if tp+fn > 0 else 0.


def calculate_batch_stats(prediction_batch: NDArray[np.uint8 | np.bool], label_batch: NDArray[np.uint8 | np.bool], negative_val: int = 0, positive_val: int = 255) -> Stats:
    """
    Calculate batch statistics for binary classification metrics.

    This function computes the number of true positives (TP), false positives (FP),
    false negatives (FN), and true negatives (TN) for a batch of predictions and labels.

    Args:
        prediction_batch (NDArray[np.uint8]): A batch of predictions represented as a NumPy array
            with unsigned 8-bit integer values. Each value should correspond to either the
            positive class (`positive_val`) or the negative class (`negative_val`).
        label_batch (NDArray[np.uint8]): A batch of ground truth labels represented as a NumPy array
            with unsigned 8-bit integer values. Each value should correspond to either the
            positive class (`positive_val`) or the negative class (`negative_val`).
        negative_val (int, optional): The value representing the negative class. Defaults to 0.
        positive_val (int, optional): The value representing the positive class. Defaults to 255.

    Returns:
        Stats: A tuple containing the counts of true positives (TP), false positives (FP),
        false negatives (FN), and true negatives (TN).

    Raises:
        ValueError: If the data type of `prediction_batch` or `label_batch` is not unsigned 8-bit integer.

    Notes:
        - The input arrays are flattened before computation.
        - The function assumes that the input arrays are binary and contain only the specified
          `negative_val` and `positive_val` values.
        - An assertion is performed to ensure the sum of TP, FP, FN, and TN equals the total number
          of elements in the input arrays.

    """
    if prediction_batch.dtype != np.uint8 and prediction_batch.dtype != np.bool:
        raise ValueError(f"Expected unsigned 8bit integer type (np.uint8) or boolean, got {prediction_batch.dtype} for the prediction")
    if label_batch.dtype != np.uint8 and label_batch.dtype != np.bool:
        raise ValueError(f"Expected unsigned 8bit integer type or boolean, got {label_batch.dtype} for the label_batch")
    if prediction_batch.dtype != label_batch.dtype:
        raise ValueError(f"Prediction {prediction_batch.dtype} and label {label_batch.dtype} have different types.")

    prediction = prediction_batch.flatten()
    label = label_batch.flatten()

    TP = np.sum((prediction == positive_val) & (label == positive_val))
    FP = np.sum((prediction == positive_val) & (label == negative_val))
    FN = np.sum((prediction == negative_val) & (label == positive_val))
    TN = np.sum((prediction == negative_val) & (label == negative_val))
    total = prediction.size
    assert total == TP+FP+FN+TN, "Sum of TP, FP, FN, TN does not match total elements"

    return TP, FP, FN, TN
