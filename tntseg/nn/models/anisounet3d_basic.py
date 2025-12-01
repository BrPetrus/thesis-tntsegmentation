import torch
import torch.nn as nn
import torch.nn.functional as F

from tntseg.nn.modules import DownscaleBlock, HorizontalBlock, UpscaleBlock

class AnisotropicUNet3D(nn.Module):
    def __init__(self, n_channels_in=1, n_classes_out=1, 
                 depth=3, base_channels=64, channel_growth=2,
                 horizontal_kernel=(1, 3, 3), 
                 horizontal_padding=(0, 1, 1),
                 horizontal_stride=(1, 1, 1),
                 downscale_kernel=(1, 2, 2),
                 downscale_stride=(1, 2, 2),
                 upscale_kernel=(1, 2, 2),
                 upscale_stride=(1, 2, 2)) -> None:
        """
        Anisotropic 3D UNet with configurable depth.
        
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
        """
        super().__init__()
        self.n_channels_in = n_channels_in
        self.n_classes_out = n_classes_out
        self.depth = depth
        self.horizontal_kernel = horizontal_kernel
        self.downsampling_kernel = downscale_kernel
        print(f"Using depth={self.depth}")

        # Generate channel configurations
        down_channels = []
        in_ch = n_channels_in
        for i in range(depth):
            out_ch = base_channels * (channel_growth ** i)
            down_channels.append((in_ch, out_ch))
            in_ch = out_ch
            
        # Calculate channels for neck
        neck_in_channels = down_channels[-1][1]
        neck_out_channels = neck_in_channels * channel_growth
        
        # Generate upsampling channel configuration
        self.up_channels = []
        in_ch = neck_out_channels
        for i in range(depth):
            out_ch = down_channels[depth-i-1][1]
            self.up_channels.append((in_ch, out_ch))
            in_ch = out_ch

        # Contracting path
        self.downsampling_layers = nn.ModuleList()
        for in_chan, out_chan in down_channels:
            self.downsampling_layers.append(
                nn.ModuleList([
                    HorizontalBlock(
                        in_channels=in_chan, out_channels=out_chan,
                        kernel=horizontal_kernel, 
                        padding=horizontal_padding,
                        stride=horizontal_stride
                    ),
                    DownscaleBlock(
                        kernel=downscale_kernel,
                        stride=downscale_stride
                    )
                ])
            )

        # Neck
        self.neck = HorizontalBlock(
            in_channels=neck_in_channels, 
            out_channels=neck_out_channels,
            kernel=horizontal_kernel,
            stride=horizontal_stride,
            padding=horizontal_padding
        )

        # Expansive path
        self.upsampling_layers = nn.ModuleList()
        for in_chann, out_chann in self.up_channels:
            self.upsampling_layers.append(
                nn.ModuleList([
                    UpscaleBlock(
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
        
        # Out
        self.final_conv = nn.Conv3d(self.up_channels[-1][1], self.n_classes_out, 1)

    def forward(self, x):
        x_values = []        

        # Contracting path
        for i in range(len(self.downsampling_layers)):
            hor_block, downsample_block = self.downsampling_layers[i]
            x = hor_block(x)
            x_values.append(x)
            x = downsample_block(x)
        
        # "Neck"
        x = self.neck(x)

        # Expansive path
        for i in range(len(self.upsampling_layers)):
            upsample_block, hor_block = self.upsampling_layers[i]
            x = upsample_block(x, x_values[-i-1])
            x = hor_block(x)
            
        return self.final_conv(x)
    
    def get_signature(self) -> str:
        return f"AnisotropicUNet3D-d{self.depth}-hk{self.horizontal_kernel}-dk{self.downsampling_kernel}".replace(',','_')


def create_anisotropic_unet3d(config):
    """
    Factory function to create AnisotropicUNet3D models with different configurations.
    
    Args:
        config: Configuration object with the following attributes:
            - depth: Number of downsampling/upsampling blocks
            - base_channels: Number of channels in first layer
            - channel_growth: Factor to multiply channels by at each depth
            
    Returns:
        Configured AnisotropicUNet3D model
    """
    return AnisotropicUNet3D(
        n_channels_in=config.get('n_channels_in', 1),
        n_classes_out=config.get('n_classes_out', 1),
        depth=config.get('depth', 3),
        base_channels=config.get('base_channels', 64),
        channel_growth=config.get('channel_growth', 2),
        horizontal_kernel=config.get('horizontal_kernel', (3, 3, 3)),
        horizontal_padding=config.get('horizontal_padding', (1, 1, 1)),
        horizontal_stride=config.get('horizontal_stride', (1, 1, 1)),
        downscale_kernel=config.get('downscale_kernel', (1, 2, 2)),
        downscale_stride=config.get('downscale_stride', (1, 2, 2)),
        upscale_kernel=config.get('upscale_kernel', (1, 2, 2)),
        upscale_stride=config.get('upscale_stride', (1, 2, 2))
    )


if __name__ == "__main__":
    from torchsummary import summary
    
    # Test the original network configuration
    net = AnisotropicUNet3D(1, 1)
    print("Original Network:")
    summary(net, (1, 7, 64, 64))
    
    # Test with different depths
    config = {
        'n_channels_in': 1,
        'n_classes_out': 1,
        'depth': 4,  # Deeper network
        'base_channels': 64,
        'channel_growth': 2
    }
    
    deeper_net = create_anisotropic_unet3d(config)
    print("\nDeeper Network:")
    summary(deeper_net, (1, 7, 64, 64))
    
    # Test with a shallower network
    config['depth'] = 2
    shallow_net = create_anisotropic_unet3d(config)
    print("\nShallower Network:")
    summary(shallow_net, (1, 7, 64, 64))

    # Truly 3D version
    config['horizontal_kernel'] = (3, 3, 3)
    config['horizontal_padding'] = (1, 1, 1)
    config['horizontal_stride'] = (1, 1, 1)
    config['depth'] = 4
    deep_3d_net = create_anisotropic_unet3d(config)
    print("\nDeep truly 3D network")
    summary(deep_3d_net, (1, 7, 64, 64))
    print(deep_3d_net.get_signature())