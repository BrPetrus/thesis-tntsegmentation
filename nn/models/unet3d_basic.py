import torch
import torch.nn as nn
import torch.nn.functional as F

class HorizontalBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            # conv + batch + relu  * 2
            nn.Conv3d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)

class DownscaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.MaxPool3d(2),
            HorizontalBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.downsample(x)

class UpscaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = HorizontalBlock(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Fix problem when x1 is not even
        if x1.shape[2] != x2.shape[2]:
            x1 = F.pad(x1, (0, 0, 0, 0, 0, 1), 'constant', 0)  # Pad z-dim by 1 at the end
            # TODO: make this generic

        # Concatonation
        x = torch.concat([x1, x2], dim=1)
        return self.conv(x)

class UNet3d(nn.Module):
    def __init__(self, n_channels_in = 1, n_classes_out=2):
        super().__init__()
        self.n_channels_in = n_channels_in
        self.n_classes_out = n_classes_out
    
        # Encoded part
        self.in_conv = HorizontalBlock(n_channels_in, 32)
        self.down1 = DownscaleBlock(32, 64)
        self.down2 = DownscaleBlock(64, 128)

        # Decoder part
        self.up1 = UpscaleBlock(128, 64)
        self.up2 = UpscaleBlock(64, 32)

        # Out
        self.out = nn.Conv3d(32, n_classes_out, 1)  # Dim. reduction
    
    def forward(self, x):
        # in: [7, 32, 32]
        x_in = self.in_conv(x)  # [32, 7, 32, 32]
        x_d1 = self.down1(x_in)  # [64, 3, 64, 64]
        x_d2 = self.down2(x_d1)  # [128, 1, 8, 8]
        x = self.up1(x_d2, x_d1)
        x = self.up2(x, x_in)
        return self.out(x)

if __name__ == "__main__":
    from torchsummary import summary
    # Init
    net = UNet3d(1, 1)

    summary(net, (1, 7, 32, 32))