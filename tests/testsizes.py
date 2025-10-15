import torch
from torchsummary import summary
from tntseg.nn.models.unet3d_basic import UNet3d
from tntseg.nn.models.anisounet3d_basic import AnisotropicUNet3D

if __name__ == "__main__":
    from torchsummary import summary
    
    print("=== BasicUNet (default) ===")
    net_basic = UNet3d(1, 1)
    summary(net_basic, (1, 7, 64, 64))
    
    print("\n=== AnisotropicUNet (matching BasicUNet exactly) ===")
    net_aniso = AnisotropicUNet3D(
        n_channels_in=1, 
        n_classes_out=1, 
        depth=2,  # Same depth as BasicUNet
        base_channels=32,  # Same as BasicUNet
        channel_growth=2,  # 32→64→128 same as BasicUNet
        horizontal_kernel=(3,3,3),  # Same as BasicUNet HorizontalBlock
        horizontal_padding=(1,1,1),  # Same as BasicUNet HorizontalBlock
        downscale_kernel=(2,2,2),   # Same as BasicUNet MaxPool3d(2)
        downscale_stride=(2,2,2),   # Same as BasicUNet MaxPool3d(2)
        upscale_kernel=(2,2,2),     # Same as BasicUNet ConvTranspose3d
        upscale_stride=(2,2,2)      # Same as BasicUNet ConvTranspose3d
    )
    summary(net_aniso, (1, 7, 64, 64))
    
    print("\n=== Manual Parameter Count Verification ===")
    basic_params = sum(p.numel() for p in net_basic.parameters())
    aniso_params = sum(p.numel() for p in net_aniso.parameters())
    
    print(f"BasicUNet parameters: {basic_params:,}")
    print(f"AnisotropicUNet parameters: {aniso_params:,}")
    print(f"Difference: {abs(basic_params - aniso_params):,}")
    
    # Test forward pass shapes
    test_input = torch.randn(1, 1, 7, 64, 64)
    with torch.no_grad():
        basic_out = net_basic(test_input)
        aniso_out = net_aniso(test_input)
        print(f"\nOutput shapes - Basic: {basic_out.shape}, Aniso: {aniso_out.shape}")