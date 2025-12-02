import torch
import torch.nn as nn
from torchinfo import summary
from torchviz import make_dot
import os
from pathlib import Path
import matplotlib.pyplot as plt
import sys

# Add the project root to the path (if running outside of package context)
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from tntseg.nn.models.anisounet3d_basic import AnisotropicUNet3D
from tntseg.nn.models.anisounet3d_seblock import AnisotropicUNet3DSE
from tntseg.nn.models.anisounet3d_csnet import AnisotropicUNet3DCSAM
from tntseg.nn.models.anisounet3d_usenet import AnisotropicUSENet
from tntseg.nn.models.unet3d_basic import UNet3d


def create_model_configs():
    """Create different model configurations to test."""
    configs = {
        "basic": {
            "depths": [2, 3, 4],
            "base_channels": 64,
            "channel_growth": 2,
            "horizontal_kernel": (1, 3, 3),
            "horizontal_padding": (0, 1, 1),
            "downscale_kernel": (1, 2, 2),
            "downscale_stride": (1, 2, 2),
            "upscale_kernel": (1, 2, 2),
            "upscale_stride": (1, 2, 2),
        },
        "3d": {
            "depths": [2, 3, 4],
            "base_channels": 64,
            "channel_growth": 2,
            "horizontal_kernel": (3, 3, 3),
            "horizontal_padding": (1, 1, 1),
            "downscale_kernel": (1, 2, 2),
            "downscale_stride": (1, 2, 2),
            "upscale_kernel": (1, 2, 2),
            "upscale_stride": (1, 2, 2),
        },
    }
    return configs


def create_model(model_type, config=None, depth=None):
    """Create a model instance based on type and configuration."""

    if model_type == "basic_unet":
        # Basic UNet uses fixed architecture - check the constructor signature
        return UNet3d(n_channels_in=1, n_classes_out=1)

    # For anisotropic models, we need the config
    if config is None:
        raise ValueError(f"Config required for {model_type}")

    common_args = {
        "n_channels_in": 1,
        "n_classes_out": 1,
        "depth": depth,
        "base_channels": config["base_channels"],
        "channel_growth": config["channel_growth"],
        "horizontal_kernel": config["horizontal_kernel"],
        "horizontal_padding": config["horizontal_padding"],
        "downscale_kernel": config["downscale_kernel"],
        "downscale_stride": config["downscale_stride"],
        "upscale_kernel": config["upscale_kernel"],
        "upscale_stride": config["upscale_stride"],
    }

    if model_type == "anisotropic_basic":
        return AnisotropicUNet3D(**common_args)
    elif model_type == "anisotropic_se":
        return AnisotropicUNet3DSE(**common_args)
    elif model_type == "anisotropic_csam":
        return AnisotropicUNet3DCSAM(**common_args)
    elif model_type == "anisotropic_usenet":
        return AnisotropicUSENet(**common_args)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def generate_torchinfo_summary(model, input_size, output_path):
    """Generate and save torchinfo summary as text and try to create a visual."""
    try:
        # Generate summary
        model_summary = summary(
            model,
            input_size=input_size,
            col_names=[
                "input_size",
                "output_size",
                "num_params",
                "kernel_size",
                "mult_adds",
            ],
            verbose=0,
        )

        # Save as text
        txt_path = output_path.with_suffix(".txt")
        with open(txt_path, "w") as f:
            f.write(str(model_summary))

        # Create a simple matplotlib visualization of the summary
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.text(
            0.05,
            0.95,
            str(model_summary),
            transform=ax.transAxes,
            fontfamily="monospace",
            fontsize=6,
            verticalalignment="top",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(f"Model Summary: {model.__class__.__name__}", fontsize=14)

        png_path = output_path.with_suffix(".png")
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✓ Summary saved: {txt_path} and {png_path}")
        return True

    except Exception as e:
        print(f"✗ Failed to generate summary: {e}")
        return False


def generate_torchviz_graph(model, input_tensor, output_path):
    """Generate and save torchviz computational graph."""
    try:
        model.eval()

        # Forward pass with gradient tracking
        input_tensor.requires_grad_(True)
        output = model(input_tensor)

        # Create a dummy loss for backprop visualization
        loss = output.mean()

        # Generate graph
        dot = make_dot(
            loss,
            params=dict(model.named_parameters()),
            show_attrs=True,
            show_saved=True,
        )

        # Save as PNG
        png_path = str(output_path.with_suffix(""))
        dot.render(png_path, format="png", cleanup=True)

        print(f"✓ Computation graph saved: {png_path}.png")
        return True

    except Exception as e:
        print(f"✗ Failed to generate computation graph: {e}")
        return False


def main():
    """Main function to generate all visualizations."""
    # Create output directory
    output_dir = Path("./output/model_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model types to visualize
    model_types = [
        "anisotropic_basic",
        "anisotropic_se",
        "anisotropic_csam",
        "anisotropic_usenet",
        "basic_unet",
    ]

    # Configuration types
    configs = create_model_configs()

    # Input sizes to test for basic_unet
    basic_unet_input_sizes = [
        (1, 1, 7, 64, 64),  # Small
    ]

    print("=== MODEL VISUALIZATION GENERATOR ===")
    print(f"Output directory: {output_dir}")
    print(f"Model types: {model_types}")
    print("=====================================\n")

    total_visualizations = 0
    successful_visualizations = 0

    for model_type in model_types:
        print(f"\n{'=' * 50}")
        print(f"Processing: {model_type}")
        print(f"{'=' * 50}")

        if model_type == "basic_unet":
            # Basic UNet has fixed architecture
            try:
                model = create_model(model_type)
                print(f"Created {model_type}: {model.__class__.__name__}")

                for i, input_size in enumerate(basic_unet_input_sizes):
                    print(f"\n--- Input size: {input_size} ---")

                    # Create input tensor
                    input_tensor = torch.randn(*input_size)

                    # Generate summary
                    summary_path = output_dir / f"{model_type}_input{i + 1}_summary"
                    total_visualizations += 1
                    if generate_torchinfo_summary(model, input_size, summary_path):
                        successful_visualizations += 1

                    # Generate computation graph
                    graph_path = output_dir / f"{model_type}_input{i + 1}_graph"
                    total_visualizations += 1
                    if generate_torchviz_graph(model, input_tensor, graph_path):
                        successful_visualizations += 1

            except Exception as e:
                print(f"✗ Failed to process {model_type}: {e}")
                import traceback

                traceback.print_exc()
                continue
        else:
            # Anisotropic models with different configurations and depths
            for config_name, config in configs.items():
                print(f"\n--- Configuration: {config_name} ---")

                for depth in config["depths"]:
                    print(f"\n--- Depth: {depth} ---")

                    try:
                        model = create_model(model_type, config, depth)
                        print(
                            f"Created {model_type}: {model.__class__.__name__} (depth={depth})"
                        )

                        input_size = (1, 1, 7, 64, 64)  # Anisotropic

                        input_tensor = torch.randn(*input_size)

                        # Generate summary
                        summary_path = (
                            output_dir / f"{model_type}_{config_name}_d{depth}_summary"
                        )
                        total_visualizations += 1
                        if generate_torchinfo_summary(model, input_size, summary_path):
                            successful_visualizations += 1

                        # Generate computation graph
                        graph_path = (
                            output_dir / f"{model_type}_{config_name}_d{depth}_graph"
                        )
                        total_visualizations += 1
                        if generate_torchviz_graph(model, input_tensor, graph_path):
                            successful_visualizations += 1

                    except Exception as e:
                        print(
                            f"✗ Failed to process {model_type} {config_name} depth {depth}: {e}"
                        )
                        import traceback

                        traceback.print_exc()
                        continue

    # Summary
    print(f"\n{'=' * 50}")
    print("VISUALIZATION GENERATION COMPLETE")
    print(f"{'=' * 50}")
    print(f"Total attempts: {total_visualizations}")
    print(f"Successful: {successful_visualizations}")
    if total_visualizations > 0:
        print(
            f"Success rate: {successful_visualizations / total_visualizations * 100:.1f}%"
        )
    print(f"Output directory: {output_dir}")

    # List generated files
    print(f"\nGenerated files:")
    for file in sorted(output_dir.glob("*")):
        print(f"  {file.name}")


if __name__ == "__main__":
    main()
