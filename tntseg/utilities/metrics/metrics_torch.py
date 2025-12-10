"""
PyTorch Loss Functions for Segmentation.

This module provides differentiable loss functions commonly used in 3D medical image
segmentation, implemented in PyTorch for GPU acceleration during training.
"""

import torch
from torch import nn
from torch.types import Tensor
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    """
    Tversky Index Loss for Segmentation.

    Computes the Tversky index, a generalization of Dice and Jaccard indices.
    Allows separate weighting of false positives (alpha) and false negatives (beta),
    making it useful for imbalanced segmentation tasks.

    The Tversky index is defined as:
        T = TP / (TP + α·FP + β·FN)

    When α = β = 0.5, this reduces to the Dice coefficient.
    When α = β = 1, this is equivalent to the Jaccard index.

    Parameters
    ----------
    alpha : float
        Weight for false positive term. Controls sensitivity to over-segmentation.
        Higher alpha penalizes false positives more heavily.
    beta : float
        Weight for false negative term. Controls sensitivity to under-segmentation.
        Higher beta penalizes false negatives more heavily.

    Attributes
    ----------
    alpha : float
        FP weight
    beta : float
        FN weight

    Notes
    -----
    - Input predictions should be in [0, 1] range (apply sigmoid beforehand)
    - Returns the Tversky index value (between 0 and 1), not the loss
    - For use as a loss to minimize, consider: loss = 1 - TverskyLoss(...)
    - eta parameter prevents division by zero
    """

    def __init__(self, alpha: float, beta: float, *args, **kwargs) -> None:
        """
        Initialize TverskyLoss.

        Parameters
        ----------
        alpha : float
            FP weight
        beta : float
            FN weight
        """
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred: Tensor, label: Tensor, eta: float = 1e-6) -> Tensor:
        """
        Compute Tversky index.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted segmentation, shape (B, C, ...) with values in [0, 1]
        label : torch.Tensor
            Ground truth labels, shape (B, C, ...) with values in {0, 1}
        eta : float, optional
            Small constant to prevent division by zero. Default is 1e-6.

        Returns
        -------
        torch.Tensor
            Tversky index value (scalar tensor)
        """
        TP = torch.sum(pred * label)
        FP = torch.sum((1 - label) * pred)
        FN = torch.sum(label * (1 - pred))

        return TP / (TP + self.alpha * FP + self.beta * FN + eta)


class FocalTverskyLoss(TverskyLoss):
    """
    Focal Tversky Loss for Segmentation.

    Extends TverskyLoss with a focusing parameter (gamma) to emphasize hard-to-segment
    regions. This is inspired by Focal Loss in object detection.

    The Focal Tversky Loss is defined as:
        FTL = (1 - T)^γ  where T is the Tversky index

    This downweights easy examples and focuses on hard negatives.

    Parameters
    ----------
    alpha : float
        Weight for false positive term in Tversky index
    beta : float
        Weight for false negative term in Tversky index
    gamma : float
        Focusing parameter. When gamma > 1, hard examples get higher weight.

    Attributes
    ----------
    gamma : float
        Focusing parameter

    Notes
    -----
    - gamma=1 reduces to 1 - Tversky index
    - gamma>1 emphasizes hard examples (recommended for imbalanced data)
    - gamma<1 emphasizes easy examples (less common)
    """

    def __init__(self, alpha: float, beta: float, gamma: float, *args, **kwargs):
        """
        Initialize FocalTverskyLoss.

        Parameters
        ----------
        alpha : float
            FP weight
        beta : float
            FN weight
        gamma : float
            Focusing parameter
        """
        super().__init__(alpha, beta, *args, **kwargs)
        self.gamma = gamma

    def forward(self, pred, label, eta=1e-6) -> Tensor:
        """
        Compute Focal Tversky Loss.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted segmentation
        label : torch.Tensor
            Ground truth labels
        eta : float, optional
            Small constant to prevent division by zero. Default is 1e-6.

        Returns
        -------
        torch.Tensor
            Focal Tversky Loss value (scalar tensor)
        """
        return torch.pow(
            1 - super().forward(pred=pred, label=label, eta=eta), self.gamma
        )


class DiceLoss(nn.Module):
    """
    Dice Loss for Segmentation.

    Computes 1 - Dice coefficient, where Dice is a measure of overlap between
    predicted and ground truth segmentations.

    Notes
    -----
    - Input predictions should be in [0, 1] range
    """

    def __init__(self, *args, **kwargs):
        """Initialize DiceLoss."""
        super().__init__(*args, **kwargs)

    def forward(self, pred, label, eta: float = 1e-6) -> Tensor:
        """
        Compute Dice Loss.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted segmentation, shape (...) with values in [0, 1]
        label : torch.Tensor
            Ground truth labels, shape (...) with values in {0, 1}
        eta : float, optional
            Small constant to prevent division by zero. Default is 1e-6.

        Returns
        -------
        torch.Tensor
            Dice Loss value (1 - Dice coefficient), scalar tensor
        """
        numerator = 2 * torch.sum(pred * label)
        denom = torch.sum(torch.pow(pred, 2)) + torch.sum(torch.pow(label, 2)) + eta
        return 1 - numerator / (denom)
