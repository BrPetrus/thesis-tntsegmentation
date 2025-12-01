import torch
import torch.nn as nn
import torch.nn.functional as F

from tntseg.nn.models.anisounet3d_basic import AnisotropicUNet3D
from tntseg.nn.squeeze_excitation import SqueezeExcitation3D
from tntseg.nn.modules import HorizontalBlock, UpscaleBlockSE

class AnisotropicUSENet(AnisotropicUNet3D):
    def __init__(self, n_channels_in=1, n_classes_out=1, 
                 depth=3, base_channels=64, channel_growth=2,
                 horizontal_kernel=(1, 3, 3), 
                 horizontal_padding=(0, 1, 1),
                 horizontal_stride=(1, 1, 1),
                 downscale_kernel=(1, 2, 2),
                 downscale_stride=(1, 2, 2),
                 upscale_kernel=(1, 2, 2),
                 upscale_stride=(1, 2, 2),
                 squeeze_factor=16) -> None:
        """
        Anisotropic 3D UNet with configurable depth and SE Block.
        
        Args:
            n_channels_in: Number of input channels
            n_classes_out: Number of output classes
            depth: Number of downsampling/upsampling blocks
            base_channels: Number of channels in first layer
            channel_growth: Factor to multiply channels by at each depth
            horizontal_kernel: Kernel size for horizontal convolutions
            horizontal_padding: Padding for horizontal convolutions
            horizontal_stride: Stride for horizontal convolutions
            downscale_kernel: Kernel size for downscaling
            downscale_stride: Stride for downscaling
            upscale_kernel: Kernel size for upscaling
            upscale_stride: Stride for upscaling
            squeeze_factor: How much to reduce the number of channels inside SE block 
        """
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
            upscale_stride
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
                nn.ModuleList([
                    UpscaleBlockSE(
                        in_channels=in_chann,
                        out_channels=out_chann,
                        kernel=upscale_kernel,
                        stride=upscale_stride
                    ),
                    HorizontalBlock(
                        in_channels=in_chann,
                        out_channels=out_chann,
                        kernel=horizontal_kernel,
                        stride=horizontal_stride,
                        padding=horizontal_padding
                    )
                ])
            )
        



    def forward(self, x):
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
            x = upsample_block(x, x_values[-i-1])
            x = hor_block(x)
            
        return self.final_conv(x)
    
    def get_signature(self):
        return f"{super().get_signature()}_usenet_sf{self.squeeze_factor}"

 
if __name__ == "__main__":
    from torchsummary import summary
    
    # Test the original network configuration
    net = AnisotropicUSENet(1, 1, depth=3)
    print("Original Network:")
    summary(net, (1, 7, 64, 64))
    