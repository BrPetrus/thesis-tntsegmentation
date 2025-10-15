from dataclasses import dataclass
from enum import StrEnum, auto

from logging import basicConfig
from typing import Tuple

class ModelType(StrEnum):
    UNet3D = "unet3d"
    AnisotropicUNet = "anisotropicunet"
    AnisotropicUNetSE = "anisotropicunet_se"
    AnisotropicUNetCSAM = "anisotropicunet_csam"
    AnisotropicUNetUSENet = "anisotropicunet_usenet"

@dataclass
class BaseConfig:
    epochs: int
    lr: float
    batch_size: int
    num_workers: int
    shuffle: bool
    device: str
    input_folder: str
    seed: int = 42
    dataset_std: float = 0.07579
    dataset_mean: float = 0.05988
    test_size: float = 1/3
    notimprovement_tolerance: int = 50
    eval_tversky_alpha: float = 0.8
    eval_tversky_beta: float = 0.2
    eval_tversky_gamma: float = 2
    use_cross_entropy: bool = True
    cross_entropy_loss_weight: float = 0.5
    ce_use_weights: bool = True
    ce_pos_weight: float  = ((3602171+67845) / 67845.) / 1. # Negative/positive ratio to penalize
    use_dice_loss: bool = True
    dice_loss_weight: float = 0.5
    use_focal_tversky_loss: bool = False
    focal_tversky_loss_weight: float = 1.0
    train_focal_tversky_alpha: float = 0.8
    train_focal_tversky_beta: float = 0.2
    train_focal_tversky_gamma: float = 2
    weight_decay: float = 0.0001
    model_type: ModelType = ModelType.Unkwnown
    crop_size: Tuple[int, int, int] = (7, 64, 64)

@dataclass
class AnisotropicUNetConfig(BaseConfig):
    model_depth: int = 3 
    base_channels: int = 64
    channel_growth: int = 2
    horizontal_kernel: Tuple[int, int, int] = (1, 3, 3) 
    horizontal_padding: Tuple[int, int, int] = (0, 1, 1)

@dataclass
class AnisotropicUNetSEConfig(AnisotropicUNetConfig):
    reduction_factor: int = 16