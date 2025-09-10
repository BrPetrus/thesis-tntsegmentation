from torch import nn
from config import Config
import torch
import logging
import tifffile
from pathlib import Path
import numpy as np

from torch.types import Tensor
from typing import List, Tuple

import tntseg.utilities.metrics.metrics_torch as tntloss
from tntseg.nn.models.anisounet3d_basic import AnisotropicUNet3D
from tntseg.nn.models.unet3d_basic import UNet3d

logger = logging.getLogger()

def create_neural_network(config: Config, in_channels: int, out_channels: int) -> nn.Module:
    match config.neural_network:
        case 'AnisotropicUNetV0':
            return AnisotropicUNet3D(
                in_channels, 
                out_channels, 
                depth=config.model_depth, 
                base_channels=config.base_channels, 
                channel_growth=config.channel_growth,
                horizontal_kernel=config.horizontal_kernel,
                horizontal_padding=config.horizontal_padding
            )
        case 'BasicUNetV1':
            return UNet3d(in_channels, out_channels)
        case '2DCSNet':
            raise NotImplementedError
        case 'AttentionUNet':
            raise NotImplementedError
        case 'SEBlockUNet':
            raise NotImplementedError
        case _:
            raise ValueError(f"Unknown model type '{config.neural_network}'")

def create_loss_criterion(config: Config) -> nn.Module:
    loss_functions = []
    weights = []

    if config.use_cross_entropy:
        if config.ce_use_weights:
            # Use weights
            loss_functions.append(nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config.ce_pos_weight)))
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
                gamma=config.train_focal_tversky_gamma
            ))
        weights.append(config.focal_tversky_loss_weight)
    return CombinedLoss(loss_functions, weights)


class CombinedLoss(nn.Module):
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
            total_loss += weight*loss
            logger.debug(f"{type(loss_fun)} - {loss}")
        return total_loss


def visualize_transform_effects(dataloader, num_samples=3, output_folder=None):
    """Save before/after images to verify transformations are working"""
    if output_folder:
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True, parents=True)
    
    # Get raw data before transformations
    dataset = dataloader.dataset
    transforms_backup = dataset.transforms
    
    
    for i in range(min(num_samples, len(dataset))):
        # Temporarily disable transforms
        dataset.transforms = MT.Compose([MT.ToTensord(keys=['volume', 'mask3d'])])
        # Get original sample without transforms
        orig_volume, orig_mask = dataset[i]
        
        # Restore transforms
        dataset.transforms = transforms_backup
        
        # Get transformed sample
        trans_volume, trans_mask = dataset[i]
        
        # Compare shapes and values
        print(f"Sample {i}:")
        print(f"  Original shape: {orig_volume.shape}, Transformed shape: {trans_volume.shape}")
        print(f"  Original range: {orig_volume.min():.3f}-{orig_volume.max():.3f}, "
              f"Transformed range: {trans_volume.min():.3f}-{trans_volume.max():.3f}")
        print(f"  Original mean: {orig_volume.mean():.3f}, std: {orig_volume.std():.3f}"
              f"Transformed mean: {trans_volume.mean():.3f}, std{trans_volume.std():.3f}")
              
        
        if output_folder:
            # Save original and transformed as TIFF for comparison
            tifffile.imwrite(output_path / f"sample_{i}_original.tiff", 
                           orig_volume.astype(np.float32))
            tifffile.imwrite(output_path / f"sample_{i}_transformed.tiff", 
                           trans_volume.detach().cpu().numpy().astype(np.float32))
    
    # Restore transforms
    dataset.transforms = transforms_backup