"""
Isotropic 3D UNet Implementation.

This module provides a standard 3D UNet architecture with symmetric (isotropic) kernels
and strides. The UNet follows the classic encoder-decoder design with skip connections.

The network uses:
- Cubic convolution kernels (3x3x3) with uniform stride (1, 1, 1)
- Isotropic downsampling with stride (2, 2, 2) in all dimensions
- Isotropic upsampling with stride (2, 2, 2) in all dimensions
- Skip connections from encoder to decoder

This is suitable for uniformly-sampled 3D data where all spatial dimensions have
similar resolution and importance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tntseg.nn.modules import HorizontalBlock, DownscaleBlock, UpscaleBlock


class UNet3d(nn.Module):
    """
    Isotropic 3D UNet with fixed depth of 2.

    This is a standard UNet architecture with fixed depth (2 encoding and 2 decoding
    levels) designed for 3D medical image segmentation. The network is fully
    isotropic with cubic kernels and symmetric stride values.

    Parameters
    ----------
    n_channels_in : int, optional
        Number of input channels. Default is 1.
    n_classes_out : int, optional
        Number of output classes (segmentation targets). Default is 1.
    model_depth : int, optional
        Model depth parameter (currently unused). Default is 3.

    Attributes
    ----------
    n_channels_in : int
        Number of input channels
    n_classes_out : int
        Number of output classes
    model_depth : int
        Model depth parameter

    Examples
    --------
    >>> model = UNet3d(n_channels_in=1, n_classes_out=1)
    >>> x = torch.randn(2, 1, 32, 128, 128)  # batch_size, channels, depth, height, width
    >>> output = model(x)
    >>> output.shape
    torch.Size([2, 1, 32, 128, 128])

    Notes
    -----
    - Input shape: (B, 1, D, H, W) where B is batch size, D is depth, H is height, W is width
    - Output shape: (B, n_classes_out, D, H, W)
    - All spatial dimensions are processed symmetrically (isotropic)
    """

    def __init__(self, n_channels_in=1, n_classes_out=1, model_depth=3):
        super().__init__()
        self.n_channels_in = n_channels_in
        self.n_classes_out = n_classes_out
        self.model_depth = model_depth

        # Fixed isotropic parameters
        horizontal_kernel = (3, 3, 3)
        horizontal_padding = (1, 1, 1)
        downscale_kernel = (2, 2, 2)
        downscale_stride = (2, 2, 2)
        upscale_kernel = (2, 2, 2)
        upscale_stride = (2, 2, 2)

        # Encoder path
        self.in_conv = HorizontalBlock(
            n_channels_in, 64, kernel=horizontal_kernel, padding=horizontal_padding
        )

        self.down1 = DownscaleBlock(kernel=downscale_kernel, stride=downscale_stride)
        self.conv1 = HorizontalBlock(
            64, 128, kernel=horizontal_kernel, padding=horizontal_padding
        )

        self.down2 = DownscaleBlock(kernel=downscale_kernel, stride=downscale_stride)
        self.conv2 = HorizontalBlock(
            128, 256, kernel=horizontal_kernel, padding=horizontal_padding
        )

        # Decoder path
        self.up1 = UpscaleBlock(256, 128, kernel=upscale_kernel, stride=upscale_stride)
        self.conv_up1 = HorizontalBlock(
            256,
            128,  # 128 from upscale + 128 from skip = 256
            kernel=horizontal_kernel,
            padding=horizontal_padding,
        )

        self.up2 = UpscaleBlock(128, 64, kernel=upscale_kernel, stride=upscale_stride)
        self.conv_up2 = HorizontalBlock(
            128,
            64,  # 64 from upscale + 64 from skip = 128
            kernel=horizontal_kernel,
            padding=horizontal_padding,
        )

        # Output
        self.out = nn.Conv3d(64, n_classes_out, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through the UNet.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 1, D, H, W) where B is batch size, D is depth,
            H is height, and W is width.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, n_classes_out, D, H, W)
        """
        # Encoder with skip connections
        x_in = self.in_conv(x)

        x = self.down1(x_in)
        x_d1 = self.conv1(x)

        x = self.down2(x_d1)
        x_d2 = self.conv2(x)

        # Decoder
        x = self.up1(x_d2, x_d1)
        x = self.conv_up1(x)

        x = self.up2(x, x_in)
        x = self.conv_up2(x)

        return self.out(x)

    def get_signature(self) -> str:
        """
        Get a string identifier for the model architecture.

        Returns
        -------
        str
            Model signature: "basicunet"
        """
        return "basicunet"


if __name__ == "__main__":
    from torchsummary import summary

    net = UNet3d(n_channels_in=1, n_classes_out=1)
    summary(net, (1, 32, 128, 128))
