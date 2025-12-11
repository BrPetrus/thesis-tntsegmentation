"""
Training utilities for 3D UNet-based segmentation models.

Provides model creation, loss function setup, and transform visualization
for PyTorch/MONAI workflows. Supports multiple UNet variants and combined loss functions.


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

from torch import nn
from config import BaseConfig, AnisotropicUNetConfig, AnisotropicUNetSEConfig
import torch
import logging
import tifffile
from pathlib import Path
import numpy as np

from torch.types import Tensor
from typing import List

import tntseg.utilities.metrics.metrics_torch as tntloss
from tntseg.nn.models.anisounet3d_basic import AnisotropicUNet3D
from tntseg.nn.models.unet3d_basic import UNet3d
from tntseg.nn.models.anisounet3d_csnet import AnisotropicUNet3DCSAM
from tntseg.nn.models.anisounet3d_seblock import AnisotropicUNet3DSE
from config import ModelType
from tntseg.nn.models.anisounet3d_usenet import AnisotropicUSENet

import monai.transforms as MT

logger = logging.getLogger()


def create_neural_network(
    config: BaseConfig, in_channels: int, out_channels: int
) -> nn.Module:
    """Instantiate a segmentation model from configuration.

    Parameters
    ----------
    config : BaseConfig
        Model configuration dataclass (architecture, hyperparameters).
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.

    Returns
    -------
    nn.Module
        Instantiated segmentation model.

    Raises
    ------
    ValueError
        If config type does not match model type or is unknown.
    """
    match config.model_type:
        case ModelType.AnisotropicUNet:
            if not isinstance(config, AnisotropicUNetConfig):
                raise ValueError(
                    f"Wrong config provided. Expected AnisotropicUNetConfig, got {type(config)}"
                )
            return AnisotropicUNet3D(
                in_channels,
                out_channels,
                depth=config.model_depth,
                base_channels=config.base_channels,
                channel_growth=config.channel_growth,
                horizontal_kernel=config.horizontal_kernel,
                horizontal_padding=config.horizontal_padding,
                upscale_kernel=config.upscale_kernel,
                upscale_stride=config.upscale_stride,
                downscale_kernel=config.downscale_kernel,
                downscale_stride=config.downscale_stride,
            )
        case ModelType.AnisotropicUNetCSAM:
            if not isinstance(config, AnisotropicUNetConfig):
                raise ValueError(
                    f"Wrong config provided. Expected AnisotropicUNetConfig, got {type(config)}"
                )
            return AnisotropicUNet3DCSAM(
                in_channels,
                out_channels,
                depth=config.model_depth,
                base_channels=config.base_channels,
                channel_growth=config.channel_growth,
                horizontal_kernel=config.horizontal_kernel,
                horizontal_padding=config.horizontal_padding,
                upscale_kernel=config.upscale_kernel,
                upscale_stride=config.upscale_stride,
                downscale_kernel=config.downscale_kernel,
                downscale_stride=config.downscale_stride,
            )
        case ModelType.UNet3D:
            if not isinstance(config, BaseConfig):
                raise ValueError(
                    f"Wrong config provided. Expected BaseConfig, got {type(config)}"
                )
            return UNet3d(in_channels, out_channels)
        case ModelType.AnisotropicUNetSE:
            if not isinstance(config, AnisotropicUNetSEConfig):
                raise ValueError(
                    f"Wrong config provided. Expected AnisotropicUNetSEConfig, got {type(config)}"
                )
            return AnisotropicUNet3DSE(
                in_channels,
                out_channels,
                depth=config.model_depth,
                base_channels=config.base_channels,
                channel_growth=config.channel_growth,
                horizontal_kernel=config.horizontal_kernel,
                horizontal_padding=config.horizontal_padding,
                squeeze_factor=config.reduction_factor,
                upscale_kernel=config.upscale_kernel,
                upscale_stride=config.upscale_stride,
                downscale_kernel=config.downscale_kernel,
                downscale_stride=config.downscale_stride,
            )
        case ModelType.AnisotropicUNetUSENet:
            if not isinstance(config, AnisotropicUNetConfig):
                raise ValueError(
                    f"Wrong config provided. Expected AnisotropicUNetConfig, got {type(config)}"
                )
            return AnisotropicUSENet(
                in_channels,
                out_channels,
                depth=config.model_depth,
                base_channels=config.base_channels,
                channel_growth=config.channel_growth,
                horizontal_kernel=config.horizontal_kernel,
                horizontal_padding=config.horizontal_padding,
                squeeze_factor=config.reduction_factor,
                upscale_kernel=config.upscale_kernel,
                upscale_stride=config.upscale_stride,
                downscale_kernel=config.downscale_kernel,
                downscale_stride=config.downscale_stride,
            )
        case _:
            raise ValueError(f"Unknown model type '{config.model_type}'")


def create_loss_criterion(config: BaseConfig) -> nn.Module:
    """Create a combined loss function from configuration.

    Parameters
    ----------
    config : BaseConfig
        Model configuration with loss settings and weights.

    Returns
    -------
    nn.Module
        Combined loss function (BCE, Dice, Focal Tversky, etc.).
    """
    loss_functions = []
    weights = []

    if config.use_cross_entropy:
        if config.ce_use_weights:
            # Use weights
            loss_functions.append(
                nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config.ce_pos_weight))
            )
        else:
            loss_functions.append(nn.BCEWithLogitsLoss())
        weights.append(config.cross_entropy_loss_weight)
    if config.use_dice_loss:
        loss_functions.append(tntloss.DiceLoss())
        weights.append(config.dice_loss_weight)
    if config.use_focal_tversky_loss:
        loss_functions.append(
            tntloss.FocalTverskyLoss(
                alpha=config.train_focal_tversky_alpha,
                beta=config.train_focal_tversky_beta,
                gamma=config.train_focal_tversky_gamma,
            )
        )
        weights.append(config.focal_tversky_loss_weight)
    return CombinedLoss(loss_functions, weights)


class CombinedLoss(nn.Module):
    """Weighted sum of multiple loss functions.

    Parameters
    ----------
    loss_functions : list of nn.Module
        List of loss functions to combine.
    loss_weights : list of float
        Corresponding weights for each loss function.

    Methods
    -------
    forward(pred, mask)
        Compute weighted sum of losses for prediction and mask.
    """

    def __init__(self, loss_functions: List[nn.Module], loss_weights: List[float]):
        super().__init__()
        if len(loss_functions) != len(loss_weights):
            raise ValueError("Number of loss functions must match number of weights")
        self.loss_functions = nn.ModuleList(loss_functions)
        self.loss_weights = loss_weights

    def forward(self, pred: Tensor, mask: Tensor) -> Tensor:
        total_loss = 0.0
        for loss_fun, weight in zip(self.loss_functions, self.loss_weights):
            if isinstance(loss_fun, nn.BCEWithLogitsLoss):
                loss = loss_fun(pred, mask)
            else:
                loss = loss_fun(torch.sigmoid(pred), mask)
            total_loss += weight * loss
            logger.debug(f"{type(loss_fun)} - {loss}")
        return total_loss


def visualize_transform_effects(dataloader, num_samples=3, output_folder=None):
    """Save before/after images to verify dataset transformations.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        DataLoader for the dataset to visualize.
    num_samples : int, optional
        Number of samples to visualize (default: 3).
    output_folder : str or Path, optional
        Directory to save TIFF images (default: None).

    Returns
    -------
    None
        Images and masks are saved to disk if output_folder is provided.
    """
    if output_folder:
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True, parents=True)

    # Get raw data before transformations
    dataset = dataloader.dataset
    transforms_backup = dataset.transforms

    for i in range(min(num_samples, len(dataset))):
        # Temporarily disable transforms
        dataset.transforms = MT.Compose([MT.ToTensord(keys=["volume", "mask3d"])])
        # Get original sample without transforms
        orig_volume, orig_mask = dataset[i]

        # Restore transforms
        dataset.transforms = transforms_backup

        # Get transformed sample
        trans_volume, trans_mask = dataset[i]

        # Compare shapes and values
        print(f"Sample {i}:")
        print(
            f"  Original shape: {orig_volume.shape}, Transformed shape: {trans_volume.shape}"
        )
        print(
            f"  Original range: {orig_volume.min():.3f}-{orig_volume.max():.3f}, "
            f"Transformed range: {trans_volume.min():.3f}-{trans_volume.max():.3f}"
        )
        print(
            f"  Original mean: {orig_volume.mean():.3f}, std: {orig_volume.std():.3f}"
            f"Transformed mean: {trans_volume.mean():.3f}, std{trans_volume.std():.3f}"
        )

        if output_folder:
            # Save original and transformed as TIFF for comparison
            tifffile.imwrite(
                output_path / f"sample_{i}_original.tiff",
                orig_volume.astype(np.float32),
            )
            tifffile.imwrite(
                output_path / f"sample_{i}_transformed.tiff",
                trans_volume.detach().cpu().numpy().astype(np.float32),
            )
            tifffile.imwrite(
                output_path / f"mask_{i}_original.tiff", orig_mask.astype(np.float32)
            )
            tifffile.imwrite(
                output_path / f"mask_{i}_transformed.tiff",
                trans_mask.detach().cpu().numpy().astype(np.float32),
            )

    # Restore transforms
    dataset.transforms = transforms_backup
