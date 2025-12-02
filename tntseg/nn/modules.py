import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, List, Optional
from torchvision.ops import SqueezeExcitation

from tntseg.nn.squeeze_excitation import SqueezeExcitation3D


class HorizontalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int | List[int] = 3,
        padding: int | List[int] = 1,
        stride: int | List[int] = 1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            # conv + batch + relu  * 2
            nn.Conv3d(in_channels, out_channels, kernel, stride, padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Dropout3d(p=0.2),
            nn.Conv3d(out_channels, out_channels, kernel, stride, padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Dropout3d(p=0.2)
        )
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        return self.block(x)


class DownscaleBlock(nn.Module):
    def __init__(
        self, kernel: int | List[int] = 2, stride: Optional[int | List[int]] = None
    ):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.MaxPool3d(kernel, stride),
        )

    def forward(self, x):
        return self.downsample(x)


class UpscaleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int | List[int] = 2,
        stride: int | List[int] = 2,
    ):
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=kernel, stride=stride
        )
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Fix problem when x1 is not even. The dims are (B,C,D,H,W)
        # NOTE: We are considering just the D dimension to be not divisble by 2
        if x1.shape[2] != x2.shape[2]:
            x1 = F.pad(
                x1, (0, 0, 0, 0, 0, 1), "constant", 0
            )  # Pad z-dim by 1 at the end

        # Concatonation
        x = torch.concat([x1, x2], dim=1)
        return x


class UpscaleBlockSE(UpscaleBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int | List[int] = 2,
        stride: int | List[int] = 2,
    ):
        super().__init__(in_channels, out_channels, kernel, stride)
        self.se_block = SqueezeExcitation3D(self.out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.se_block(x1)

        # Fix problem when x1 is not even
        # See method forward() of UpscaleBlock class
        if x1.shape[2] != x2.shape[2]:
            x1 = F.pad(
                x1, (0, 0, 0, 0, 0, 1), "constant", 0
            )  # Pad z-dim by 1 at the end

        # Concatonation
        x = torch.concat([x1, x2], dim=1)
        return x
