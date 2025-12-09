"""
Anisotropic 3D UNet with Context and Spatial Attention Module (CSAM).

This module extends AnisotropicUNet3D with a Context and Spatial Attention Module
at the bottleneck. CSAM provides spatial attention that is aware of context,
helping the network focus on task-relevant spatial regions.

Architecture:
- Encoder: Same as AnisotropicUNet3D
- Neck: HorizontalBlock + AffinityAttention3d (CSAM)
- Decoder: Same as AnisotropicUNet3D

The CSAM block operates by computing affinity/attention maps that weight spatial
locations based on their contextual relationships, particularly useful for structured
data like tunnel segmentation.

References
----------
Inspired by the CSNet approach for semantic segmentation with context and spatial awareness.
Note: CSAM originally expects dimension permutation; this implementation handles it automatically.
"""

import torch
from torchvision.ops import SqueezeExcitation
import torch.nn as nn
import torch.nn.functional as F

from tntseg.nn.models.anisounet3d_basic import AnisotropicUNet3D
from tntseg.nn.csnet_affinity_modules import AffinityAttention3d


class AnisotropicUNet3DCSAM(AnisotropicUNet3D):
    """
    Anisotropic 3D UNet with Context and Spatial Attention Module.

    Adds a spatial attention mechanism at the bottleneck that learns to focus on
    important spatial regions by computing affinity relationships.

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

    Attributes
    ----------
    affinity_attention : AffinityAttention3d
        Context and Spatial Attention Module applied at bottleneck

    Examples
    --------
    >>> model = AnisotropicUNet3DCSAM(depth=3)
    >>> x = torch.randn(2, 1, 7, 64, 64)
    >>> output = model(x)
    >>> output.shape
    torch.Size([2, 1, 7, 64, 64])

    Notes
    -----
    The CSAM block is applied after the neck HorizontalBlock.
    Due to dimension preferences of the CSAM module, input is permuted from
    (B,C,D,H,W) → (B,C,H,W,D), CSAM applied, then permuted back.
    This permutation should have minimal impact due to CSAM's symmetric design,
    but is kept for compatibility with the original CSAM implementation.
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
    ) -> None:
        """Initialize AnisotropicUNet3DCSAM network."""
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
        self.affinity_attention = AffinityAttention3d(self.neck.out_channels)

    def forward(self, x):
        """
        Forward pass through the CSAM-UNet.

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

        x = self.neck(x)

        # Apply the Spatial Attention from CSNet paper
        # Note: that the authors of CSAM expected a different perumation of the dimensions
        #       it should not really make a difference due to the symmetric nature of the CSAM
        #       block, but just in case we are leaving this here.
        x_perm = x.permute(0, 1, 3, 4, 2).contiguous()  # B,C,D,H,W ~> B,C,H,W,D
        attention = self.affinity_attention(x_perm)
        attention = attention.permute(
            0, 1, 4, 2, 3
        ).contiguous()  # B,C,H,W,D ~> B,C,D,H,W
        x = x + attention

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
            Model identifier including parent signature and CSAM variant
        """
        return f"{super().get_signature()}-CSAM"


if __name__ == "__main__":
    from torchsummary import summary

    # Test the original network configuration
    net = AnisotropicUNet3DCSAM(1, 1)
    print("Original Network:")
    summary(net, (1, 7, 64, 64))

    deeper_net = AnisotropicUNet3DCSAM(1, 1, depth=5)
    print("\nDeeper Network:")
    summary(deeper_net, (1, 7, 64, 64))
