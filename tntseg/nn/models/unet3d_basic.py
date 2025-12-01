import torch
import torch.nn as nn
import torch.nn.functional as F
from tntseg.nn.modules import HorizontalBlock, DownscaleBlock, UpscaleBlock

class UNet3d(nn.Module):
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
            n_channels_in, 
            64,
            kernel=horizontal_kernel,
            padding=horizontal_padding
        )
        
        self.down1 = DownscaleBlock(
            kernel=downscale_kernel,
            stride=downscale_stride
        )
        self.conv1 = HorizontalBlock(
            64, 128,
            kernel=horizontal_kernel,
            padding=horizontal_padding
        )
        
        self.down2 = DownscaleBlock(
            kernel=downscale_kernel,
            stride=downscale_stride
        )
        self.conv2 = HorizontalBlock(
            128, 256,
            kernel=horizontal_kernel,
            padding=horizontal_padding
        )
        
        # Decoder path
        self.up1 = UpscaleBlock(
            256, 128,
            kernel=upscale_kernel,
            stride=upscale_stride
        )
        self.conv_up1 = HorizontalBlock(
            256, 128,  # 128 from upscale + 128 from skip = 256
            kernel=horizontal_kernel,
            padding=horizontal_padding
        )
        
        self.up2 = UpscaleBlock(
            128, 64,
            kernel=upscale_kernel,
            stride=upscale_stride
        )
        self.conv_up2 = HorizontalBlock(
            128, 64,  # 64 from upscale + 64 from skip = 128
            kernel=horizontal_kernel,
            padding=horizontal_padding
        )
        
        # Output
        self.out = nn.Conv3d(64, n_classes_out, kernel_size=1)
    
    def forward(self, x):
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
        return "basicunet"


if __name__ == "__main__":
    from torchsummary import summary
    
    net = UNet3d(n_channels_in=1, n_classes_out=1)
    summary(net, (1, 32, 128, 128))