"""
Core Neural Network Modules for 3D Segmentation.

This module provides fundamental building blocks for constructing 3D UNet architectures:

- HorizontalBlock: Double convolution block with batch norm and ReLU
- DownscaleBlock: Max pooling downsampling
- UpscaleBlock: Transposed convolution upsampling with skip connection concatenation
- UpscaleBlockSE: UpscaleBlock with Squeeze-Excitation attention

These modules are designed to be flexible with respect to kernel sizes and strides,
allowing both isotropic and anisotropic architectures.

Key Design Decisions
--------------------
- HorizontalBlock applies the same stride to both convolutions (for consistency with
  batch norm statistics)
- DownscaleBlock uses MaxPool3d for downsampling (preserves features)
- UpscaleBlock performs transposed convolution followed by concatenation with skip
  connections
- Handles dimension mismatches in z-dimension (depth) with mirror padding

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

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, List, Optional
from torchvision.ops import SqueezeExcitation

from tntseg.nn.squeeze_excitation import SqueezeExcitation3D


class HorizontalBlock(nn.Module):
    """
    Double Convolution Block with Batch Normalization and ReLU.

    Applies two consecutive 3D convolutions, each followed by batch normalization
    and ReLU activation. This is the fundamental building block for UNet.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel : int or tuple of int, optional
        Kernel size for convolutions. Default is 3.
    padding : int or tuple of int, optional
        Padding for convolutions. Default is 1.
    stride : int or tuple of int, optional
        Stride for convolutions. Default is 1.

    Attributes
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels

    Notes
    -----
    Both convolutions use the same kernel, padding, and stride. This ensures
    consistent receptive field and output dimensions.
    """
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
        """
        Forward pass through the double convolution block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, in_channels, D, H, W)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, out_channels, D, H, W)
        """
        return self.block(x)


class DownscaleBlock(nn.Module):
    """
    Max Pooling Downsampling Block.

    Performs 3D max pooling for spatial downsampling. Supports both isotropic
    (cubic) and anisotropic (non-cubic) kernels and strides.

    Parameters
    ----------
    kernel : int or tuple of int, optional
        Kernel size for max pooling. Default is 2.
    stride : int, tuple of int, or None, optional
        Stride for max pooling. If None, defaults to kernel size. Default is None.
    """

    def __init__(
        self, kernel: int | List[int] = 2, stride: Optional[int | List[int]] = None
    ):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.MaxPool3d(kernel, stride),
        )

    def forward(self, x):
        """
        Forward pass through max pooling.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, D, H, W)

        Returns
        -------
        torch.Tensor
            Downsampled tensor with spatial dimensions reduced
        """
        return self.downsample(x)


class UpscaleBlock(nn.Module):
    """
    Transposed Convolution Upsampling with Skip Connection Concatenation.

    Performs spatial upsampling via transposed convolution, then concatenates with
    skip connection features from the encoder. Handles dimension mismatches in the
    z-dimension (depth) with mirror padding.

    Parameters
    ----------
    in_channels : int
        Number of input channels (from bottleneck)
    out_channels : int
        Number of output channels after transposed convolution
    kernel : int or tuple of int, optional
        Kernel size for transposed convolution. Default is 2.
    stride : int or tuple of int, optional
        Stride for transposed convolution. Default is 2.

    Attributes
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels (after transposed conv, before concatenation)

    Examples
    --------
    >>> # Isotropic upsampling
    >>> up = UpscaleBlock(in_channels=512, out_channels=256, kernel=(2, 2, 2), stride=(2, 2, 2))
    >>> x = torch.randn(2, 512, 8, 16, 16)
    >>> skip = torch.randn(2, 256, 16, 32, 32)
    >>> output = up(x, skip)
    >>> output.shape  # Concatenated channels
    torch.Size([2, 512, 16, 32, 32])  # 256 + 256

    >>> # Anisotropic upsampling
    >>> up_aniso = UpscaleBlock(in_channels=512, out_channels=256, kernel=(1, 2, 2), stride=(1, 2, 2))
    >>> x = torch.randn(2, 512, 7, 16, 16)
    >>> skip = torch.randn(2, 256, 7, 32, 32)
    >>> output = up_aniso(x, skip)
    >>> output.shape
    torch.Size([2, 512, 7, 32, 32])  # z-dimension unchanged

    Notes
    -----
    The output channels equal the concatenation of upscaled and skip features:
    output_channels = out_channels + skip.channels
    """

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
        """
        Upsample and concatenate with skip connection.

        Parameters
        ----------
        x1 : torch.Tensor
            Tensor to upsample, shape (B, in_channels, D_small, H_small, W_small)
        x2 : torch.Tensor
            Skip connection tensor, shape (B, skip_channels, D_large, H_large, W_large)

        Returns
        -------
        torch.Tensor
            Concatenated tensor of shape (B, out_channels + skip_channels, D_large, H_large, W_large)
        """
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
    """
    Transposed Convolution Upsampling with Squeeze-Excitation Attention.

    Extends UpscaleBlock with a Squeeze-Excitation block after upsampling to
    adaptively recalibrate channel responses based on global information.

    Parameters
    ----------
    in_channels : int
        Number of input channels (from bottleneck)
    out_channels : int
        Number of output channels after transposed convolution
    kernel : int or tuple of int, optional
        Kernel size for transposed convolution. Default is 2.
    stride : int or tuple of int, optional
        Stride for transposed convolution. Default is 2.

    Attributes
    ----------
    se_block : SqueezeExcitation3D
        Squeeze-Excitation module applied to upsampled features
    """

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
        """
        Upsample with SE attention and concatenate with skip connection.

        Parameters
        ----------
        x1 : torch.Tensor
            Tensor to upsample, shape (B, in_channels, D_small, H_small, W_small)
        x2 : torch.Tensor
            Skip connection tensor, shape (B, skip_channels, D_large, H_large, W_large)

        Returns
        -------
        torch.Tensor
            Concatenated tensor of shape (B, out_channels + skip_channels, D_large, H_large, W_large)
        """
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
