import torch
import torch.nn as nn
import torch.nn.functional as F

from tntseg.nn.modules import DownscaleBlock, HorizontalBlock, UpscaleBlock

class AnisotropicUNet3D(nn.Module):
    def __init__(self, n_channels_in = 1, n_classes_out = 1) -> None:
        super().__init__()
        self.n_channels_in = n_channels_in
        self.n_classes_out = n_classes_out

        # Contracting path
        horizontal_kernel = (1, 3, 3)
        horizontal_padding = (0, 1, 1)
        horizontal_stride = (1, 1, 1)
        downscale_kernel = (1, 2, 2)
        downscale_stride = (1, 2, 2)
        channels = [
            (1, 64),
            (64, 128),
            (128, 256)
        ]
        self.downsampling_layers = nn.ModuleList()
        for in_chan, out_chan in channels:
            self.downsampling_layers.append(
                nn.ModuleList([
                    HorizontalBlock(
                        in_channels=in_chan, out_channels=out_chan,
                        kernel=horizontal_kernel, 
                        padding=horizontal_padding,
                        stride=horizontal_stride
                    ),
                    DownscaleBlock(
                        in_channels=64,
                        out_channels=64,
                        kernel=downscale_kernel,
                        stride=downscale_stride
                    )
                ])
            )

        # Neck
        self.neck = HorizontalBlock(
            in_channels=channels[-1][1], 
            out_channels=channels[-1][1]*2,
            kernel=horizontal_kernel,
            stride=horizontal_stride,
            padding=horizontal_padding
        )

        # Expansive path
        self.upsampling_layers = nn.ModuleList()
        upscale_kernel = (1, 2, 2)
        upscale_stride = (1, 2, 2)
        channels = [
            (512, 256),
            (256, 128),
            (128, 64)
        ]
        for in_chann, out_chann in channels:
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
        self.final_conv = nn.Conv3d(64, self.n_classes_out, 1)

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
    

if __name__ == "__main__":
    from torchsummary import summary
    net = AnisotropicUNet3D(1, 1)
    summary(net, (1, 7, 64, 64))