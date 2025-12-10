"""
Anisotropic 3D UNet with Squeeze-Excitation Block at Bottleneck.

This module extends the basic AnisotropicUNet3D architecture by adding a
Squeeze-Excitation block at the bottleneck (neck) layer. The SE block adaptively
recalibrates channel responses based on global information, potentially improving
feature representation at the bottleneck.

Architecture:
- Encoder: Same as AnisotropicUNet3D
- Neck: HorizontalBlock + SqueezeExcitation3D
- Decoder: Same as AnisotropicUNet3D

The SE block operates at the bottleneck feature space, allowing the network to
learn which features are most important for the specific segmentation task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from tntseg.nn.models.anisounet3d_basic import AnisotropicUNet3D
from tntseg.nn.squeeze_excitation import SqueezeExcitation3D


class AnisotropicUNet3DSE(AnisotropicUNet3D):
    """
    Anisotropic 3D UNet with Squeeze-Excitation Block.

    Adds a Squeeze-Excitation block at the bottleneck to adaptively recalibrate
    channel responses. This helps the network focus on the most informative features
    at the bottleneck layer.

    Parameters
    ----------
    n_channels_in : int, optional
        Number of input channels. Default is 1.
    n_classes_out : int, optional
        Number of output classes. Default is 1.
    depth : int, optional
        Network depth (number of encoder/decoder levels). Default is 3.
    base_channels : int, optional
        Number of channels in first encoder layer. Default is 64.
    channel_growth : int or float, optional
        Factor to multiply channels by at each depth. Default is 2.
    horizontal_kernel : tuple of int, optional
        Kernel size for horizontal convolutions. Default is (1, 3, 3).
    horizontal_padding : tuple of int, optional
        Padding for horizontal convolutions. Default is (0, 1, 1).
    horizontal_stride : tuple of int, optional
        Stride for horizontal convolutions. Default is (1, 1, 1).
    downscale_kernel : tuple of int, optional
        Kernel size for downsampling. Default is (1, 2, 2).
    downscale_stride : tuple of int, optional
        Stride for downsampling. Default is (1, 2, 2).
    upscale_kernel : tuple of int, optional
        Kernel size for upsampling. Default is (1, 2, 2).
    upscale_stride : tuple of int, optional
        Stride for upsampling. Default is (1, 2, 2).
    squeeze_factor : int, optional
        Reduction factor for SE block. Reduces channel count by this factor
        in the bottleneck of the SE module. Default is 16.

    Attributes
    ----------
    squeeze_factor : int
        SE block reduction factor
    se_module : SqueezeExcitation3D
        Squeeze-Excitation module applied at bottleneck

    Notes
    -----
    The SE block is applied after the neck HorizontalBlock, before decoder upsampling.
    """

    def __init__(
        self,
        n_channels_in=1,
        n_classes_out=1,
        depth=3,
        base_channels=64,
        channel_growth=2,
        horizontal_kernel=(1, 3, 3),
        horizontal_padding=(0, 1, 1),
        horizontal_stride=(1, 1, 1),
        downscale_kernel=(1, 2, 2),
        downscale_stride=(1, 2, 2),
        upscale_kernel=(1, 2, 2),
        upscale_stride=(1, 2, 2),
        squeeze_factor=16,
    ) -> None:
        """Initialize AnisotropicUNet3DSE network."""
        super().__init__(
            n_channels_in,
            n_classes_out,
            depth,
            base_channels,
            channel_growth,
            horizontal_kernel,
            horizontal_padding,
            horizontal_stride,
            downscale_kernel,
            downscale_stride,
            upscale_kernel,
            upscale_stride,
        )

        self.squeeze_factor = squeeze_factor
        self.se_module = SqueezeExcitation3D(
            self.neck.out_channels, reduction_factor=squeeze_factor
        )

    def forward(self, x):
        """
        Forward pass through the SE-UNet.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, n_channels_in, D, H, W)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, n_classes_out, D, H, W)
        """
        x_values = []

        # Contracting path
        for i in range(len(self.downsampling_layers)):
            hor_block, downsample_block = self.downsampling_layers[i]
            x = hor_block(x)
            x_values.append(x)
            x = downsample_block(x)

        # SE block
        x = self.neck(x)
        x = self.se_module(x)

        # Expansive path
        for i in range(len(self.upsampling_layers)):
            upsample_block, hor_block = self.upsampling_layers[i]
            x = upsample_block(x, x_values[-i - 1])
            x = hor_block(x)

        return self.final_conv(x)

    def get_signature(self):
        """
        Get model architecture signature.

        Returns
        -------
        str
            Model identifier including parent signature and squeeze factor
        """
        return f"{super().get_signature()}_sf{self.squeeze_factor}"


if __name__ == "__main__":
    from torchsummary import summary

    # Test the original network configuration
    net = AnisotropicUNet3DSE(1, 1, depth=5)
    print("Original Network:")
    summary(net, (1, 7, 64, 64))
