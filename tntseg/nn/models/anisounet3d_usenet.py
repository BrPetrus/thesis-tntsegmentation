"""
Anisotropic 3D UNet with SE Blocks in Encoder and Decoder (USENet).

This module extends AnisotropicUNet3D with Squeeze-Excitation blocks integrated
throughout both encoder and decoder paths. This creates a more comprehensive
attention mechanism compared to having SE only at the bottleneck.

Architecture:
- Encoder: HorizontalBlock + Downscale + SE block on skip connections
- Neck: HorizontalBlock
- Decoder: UpscaleBlockSE (with built-in SE) + HorizontalBlock

This design allows the network to learn channel-wise attention at multiple
scales, which can improve feature representation throughout the network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from tntseg.nn.models.anisounet3d_basic import AnisotropicUNet3D
from tntseg.nn.squeeze_excitation import SqueezeExcitation3D
from tntseg.nn.modules import HorizontalBlock, UpscaleBlockSE


class AnisotropicUSENet(AnisotropicUNet3D):
    """
    Anisotropic 3D UNet with SE Blocks in Encoder and Decoder (USENet).

    Integrates Squeeze-Excitation blocks throughout the network:
    - SE blocks on encoder skip connections
    - SE blocks embedded in UpscaleBlocks
    - This comprehensive attention helps the network learn importance weighting
      at multiple scales

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
        Reduction factor for all SE blocks. Default is 16.

    Attributes
    ----------
    squeeze_factor : int
        SE block reduction factor
    se_encoder : nn.ModuleList
        SE blocks applied to encoder skip connections
    se_decoder : nn.ModuleList
        SE blocks for decoder layers (note: UpscaleBlockSE has embedded SE)
    upsampling_layers : nn.ModuleList
        Decoder layers with UpscaleBlockSE instead of regular UpscaleBlock

    Notes
    -----
    Channel flow with SE blocks:
    - Encoder: Conv → Save → SE(saved) → use in decoder
    - Decoder: Upscale → SE(upscaled) → concat with SE'd skip → Conv
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
        """Initialize AnisotropicUSENet network."""
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
        # self.se_module = SqueezeExcitation3D(self.neck.out_channels, reduction_factor=squeeze_factor)

        # SE after encoder
        self.se_encoder = nn.ModuleList()
        for i in range(len(self.downsampling_layers)):
            self.se_encoder.append(
                SqueezeExcitation3D(self.downsampling_layers[i][0].out_channels)
            )

        # SE after transposed conv
        self.se_decoder = nn.ModuleList()
        for i in range(len(self.upsampling_layers)):
            self.se_decoder.append(
                SqueezeExcitation3D(self.upsampling_layers[i][0].out_channels)
            )

        # Upscale blocks with SE
        self.upsampling_layers = nn.ModuleList()
        for in_chann, out_chann in self.up_channels:
            self.upsampling_layers.append(
                nn.ModuleList(
                    [
                        UpscaleBlockSE(
                            in_channels=in_chann,
                            out_channels=out_chann,
                            kernel=upscale_kernel,
                            stride=upscale_stride,
                        ),
                        HorizontalBlock(
                            in_channels=in_chann,
                            out_channels=out_chann,
                            kernel=horizontal_kernel,
                            stride=horizontal_stride,
                            padding=horizontal_padding,
                        ),
                    ]
                )
            )

    def forward(self, x):
        """
        Forward pass through the USENet.

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
            x_skip = self.se_encoder[i](x)
            x_values.append(x_skip)
            x = downsample_block(x)

        x = self.neck(x)

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
            Model identifier including parent signature, USENet variant, and squeeze factor
        """
        return f"{super().get_signature()}_usenet_sf{self.squeeze_factor}"


if __name__ == "__main__":
    from torchsummary import summary

    # Test the original network configuration
    net = AnisotropicUSENet(1, 1, depth=3)
    print("Original Network:")
    summary(net, (1, 7, 64, 64))
