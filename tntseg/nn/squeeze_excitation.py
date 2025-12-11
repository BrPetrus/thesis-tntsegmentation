"""
This module implements the popular Squeeze and Excitation operator for 3D volumes.

The code is a modification of the implementation of the squeeze-and-excitation module by PyTorch team.
"""

import torch
import torch.nn as nn
from torch.types import Tensor


class SqueezeExcitation3D(nn.Module):
    def __init__(self, input_channels: int, reduction_factor: int = 16):
        super().__init__()
        squeezed_channels = input_channels // reduction_factor
        self.glob_pool = nn.AdaptiveAvgPool3d(1)  # Reduce (B,C,Z,Y,X) ~> (B,C,1,1,1)
        self.fc1 = nn.Conv3d(
            input_channels, squeezed_channels, 1
        )  # NOTE: Using conv instead of linear layers as torchvision uses
        self.fc2 = nn.Conv3d(squeezed_channels, input_channels, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def _scale_operation(self, x: Tensor) -> Tensor:
        """Calculates the scale as described in the paper"""
        x = self.glob_pool(x)  # (B, C, Z, Y, C) ~> (B, C, 1, 1, 1)
        x = self.fc1(x)  # (B, C, ...) ~> (B, C/r)
        x = self.relu(x)
        x = self.fc2(x)  # (B, C/r) ~> (B, C)
        x = self.sigmoid(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        scale = self._scale_operation(x)
        return x * scale


if __name__ == "__main__":
    arr = torch.randn(512, 7, 8, 8)
    out = SqueezeExcitation3D(512, 16)(arr)
    assert arr.shape == out.shape
