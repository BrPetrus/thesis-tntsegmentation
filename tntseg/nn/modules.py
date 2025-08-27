
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, List, Optional

class HorizontalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int | List[int] = 3, padding: int | List[int] = 1, stride: int | List[int] = 1):
        super().__init__()
        self.block = nn.Sequential(
            # conv + batch + relu  * 2
            nn.Conv3d(in_channels, out_channels, kernel, stride, padding, padding_mode="reflect"),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(),

            nn.Conv3d(out_channels, out_channels, kernel, stride, padding, padding_mode="reflect"),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d()
        )
    
    def forward(self, x):
        return self.block(x)

class DownscaleBlock(nn.Module):
    def __init__(self, kernel: int | List[int] = 2, stride: Optional[int | List[int]] = None):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.MaxPool3d(kernel, stride),
        )

    def forward(self, x):
        return self.downsample(x)

class UpscaleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int | List[int] = 2, stride: int | List[int] = 2):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel, stride=stride)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Fix problem when x1 is not even
        if x1.shape[2] != x2.shape[2]:
            x1 = F.pad(x1, (0, 0, 0, 0, 0, 1), 'constant', 0)  # Pad z-dim by 1 at the end
            # TODO: make this generic

        # Concatonation
        x = torch.concat([x1, x2], dim=1)
        return x
