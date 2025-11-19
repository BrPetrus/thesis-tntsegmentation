"""
Simple inference script for running model predictions on new images.
No evaluation metrics are calculated - just tiles, stitches, and saves predictions.
"""

import json
import torch
import numpy as np
import tifffile
from pathlib import Path
import argparse
import tqdm
from torch.utils.data import DataLoader
import monai.transforms as MT

from tntseg.nn.models.anisounet3d_basic import AnisotropicUNet3D
from tntseg.nn.models.anisounet3d_seblock import AnisotropicUNet3DSE
from tntseg.nn.models.anisounet3d_csnet import AnisotropicUNet3DCSAM
from tntseg.nn.models.unet3d_basic import UNet3d
from tntseg.nn.models.anisounet3d_usenet import AnisotropicUSENet

from tilingutilities import AggregationMethod, stitch_volume, tile_volume
from postprocess import PostprocessConfig, detect_tunnels, print_detailed_results


class TiledDataset:
    """Dataset that preserves tile positions with data."""
    def __init__(self, tiles_data, tiles_positions):
        self.tiles_data = tiles_data
        self.tiles_positions = tiles_positions

    def __len__(self):
        return len(self.tiles_data)

    def __getitem__(self, idx):
        return self.tiles_data[idx], self.tiles_positions[idx]


def load_training_config(model_path: str) -> dict:
    """Load training configuration to get dataset statistics and model parameters."""
    model_dir = Path(model_path).parent
    config_path = model_dir / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"No config found at {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    print(f"Loaded configuration from {config_path}")
    print(f"Dataset mean: {config.get('dataset_mean', 'N/A')}, Std: {config.get('dataset_std', 'N/A')}")
    print(f"Model type: {config.get('model_type', 'N/A')}")

    return config


def create_model_from_config(config: dict) -> torch.nn.Module:
    """Create a model based on loaded configuration."""
    model_type = config["model_type"]

    if model_type == "anisotropicunet":
        return AnisotropicUNet3D(
            n_channels_in=1, n_classes_out=1,
            depth=config["model_depth"],
            base_channels=config["base_channels"],
            channel_growth=config["channel_growth"],
            horizontal_kernel=tuple(config["horizontal_kernel"]),
            horizontal_padding=tuple(config["horizontal_padding"]),
            upscale_kernel=tuple(config["upscale_kernel"]),
            upscale_stride=tuple(config["upscale_stride"]),
            downscale_kernel=tuple(config["downscale_kernel"]),
            downscale_stride=tuple(config["downscale_stride"]),
        )
    elif model_type == "anisotropicunet_se":
        return AnisotropicUNet3DSE(
            n_channels_in=1, n_classes_out=1,
            depth=config["model_depth"],
            base_channels=config["base_channels"],
            channel_growth=config["channel_growth"],
            horizontal_kernel=tuple(config["horizontal_kernel"]),
            horizontal_padding=tuple(config["horizontal_padding"]),
            squeeze_factor=config["reduction_factor"],
            upscale_kernel=tuple(config["upscale_kernel"]),
            upscale_stride=tuple(config["upscale_stride"]),
            downscale_kernel=tuple(config["downscale_kernel"]),
            downscale_stride=tuple(config["downscale_stride"]),
        )
    elif model_type == "anisotropicunet_csam":
        return AnisotropicUNet3DCSAM(
            n_channels_in=1, n_classes_out=1,
            depth=config["model_depth"],
            base_channels=config["base_channels"],
            channel_growth=config["channel_growth"],
            horizontal_kernel=tuple(config["horizontal_kernel"]),
            horizontal_padding=tuple(config["horizontal_padding"]),
            upscale_kernel=tuple(config["upscale_kernel"]),
            upscale_stride=tuple(config["upscale_stride"]),
            downscale_kernel=tuple(config["downscale_kernel"]),
            downscale_stride=tuple(config["downscale_stride"]),
        )
    elif model_type == "anisotropicunet_usenet":
        return AnisotropicUSENet(
            n_channels_in=1, n_classes_out=1,
            depth=config["model_depth"],
            base_channels=config["base_channels"],
            channel_growth=config["channel_growth"],
            horizontal_kernel=tuple(config["horizontal_kernel"]),
            horizontal_padding=tuple(config["horizontal_padding"]),
            upscale_kernel=tuple(config["upscale_kernel"]),
            upscale_stride=tuple(config["upscale_stride"]),
            downscale_kernel=tuple(config["downscale_kernel"]),
            downscale_stride=tuple(config["downscale_stride"]),
            squeeze_factor=config["reduction_factor"],
        )
    elif model_type == "unet3d":
        return UNet3d(n_channels_in=1, n_classes_out=1)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_inference(
    model: torch.nn.Module,
    image_path: Path,
    output_dir: Path,
    config: dict,
    crop_size: tuple,
    tile_overlap: int,
    batch_size: int,
    device: str,
    apply_postprocessing: bool = False,
    postprocess_config: PostprocessConfig = None,
    save_probability: bool = True,
    save_binary: bool = True,
) -> None:
    """Run inference on a single image."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading image from {image_path}")
    image = tifffile.imread(image_path).astype(np.float32)
    
    # Normalize image
    mean = config.get("dataset_mean", 0.0)
    std = config.get("dataset_std", 1.0)
    image_normalized = (image - mean) / std
    
    print(f"Image shape: {image.shape}")
    print(f"Tiling with overlap: {tile_overlap}px, crop size: {crop_size}")
    
    # Tile the volume
    tiles_data, tiles_positions = tile_volume(
        image_normalized, crop_size, overlap=tile_overlap
    )
    
    tiles_dataset = TiledDataset(tiles_data, tiles_positions)
    tiles_dataloader = DataLoader(
        tiles_dataset, batch_size=batch_size, shuffle=False
    )
    
    # Run inference
    all_predictions = []
    all_positions = []
    
    model.eval()
    with torch.no_grad():
        for batch_data, batch_positions in tqdm.tqdm(tiles_dataloader, desc="Running inference"):
            batch_data = batch_data[:, None, ...]  # Add channel dimension
            batch_data = batch_data.to(device)
            predictions = model(batch_data).detach().cpu()
            
            all_predictions.extend(predictions)
            depths = batch_positions[0]
            rows = batch_positions[1]
            cols = batch_positions[2]
            all_positions.extend(
                [(depths[i], rows[i], cols[i]) for i in range(len(depths))]
            )
    
    # Stitch predictions
    print("Stitching tiles...")
    all_predictions_tensor = torch.stack(all_predictions)
    reconstructed_volume = stitch_volume(
        all_predictions_tensor,
        all_positions,
        original_shape=image.shape,
        aggregation_method=AggregationMethod.Mean,
        visualise_lines=False,
    )
    
    # Apply sigmoid to get probabilities
    probabilities = torch.sigmoid(reconstructed_volume).numpy()
    
    # Save probability map
    if save_probability:
        prob_path = output_dir / f"{image_path.stem}_probability.tif"
        tifffile.imwrite(prob_path, probabilities.astype(np.float32))
        print(f"Saved probability map to {prob_path}")
    
    # Threshold for binary predictions
    if save_binary:
        binary = (probabilities > 0.5).astype(np.uint8) * 255
        binary_path = output_dir / f"{image_path.stem}_binary.tif"
        tifffile.imwrite(binary_path, binary)
        print(f"Saved binary prediction to {binary_path}")
    
    # Run postprocessing if requested
    if apply_postprocessing and postprocess_config is not None:
        print("\nRunning postprocessing...")
        postproc_dir = output_dir / f"{image_path.stem}_postprocessed"
        
        try:
            # Note: Without ground truth, we pass None for GT mask
            tunnel_result = detect_tunnels(
                probabilities,
                gt_instance_mask=None,  # No ground truth available
                original_image=image,
                config=postprocess_config,
                output_folder=postproc_dir,
                visualise=True
            )
            
            print("\nPostprocessing completed")
            print(f"Results saved to {postproc_dir}")
            
        except Exception as e:
            print(f"Error in postprocessing: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on images using a trained model"
    )
    
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to model checkpoint (.pth file)"
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to input image (TIFF format)"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory for predictions"
    )
    
    # Optional arguments
    parser.add_argument(
        "--tile_overlap",
        type=int,
        default=20,
        help="Tile overlap in pixels (default: 20)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference (default: 8)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu). Auto-detects if not specified"
    )
    parser.add_argument(
        "--no_probability",
        action="store_true",
        help="Don't save probability map"
    )
    parser.add_argument(
        "--no_binary",
        action="store_true",
        help="Don't save binary prediction"
    )
    
    # Postprocessing arguments
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="Apply postprocessing to clean up predictions"
    )
    parser.add_argument(
        "--prediction_threshold",
        type=float,
        default=0.5,
        help="Threshold for binarization (default: 0.5)"
    )
    parser.add_argument(
        "--minimum_size",
        type=int,
        default=100,
        help="Minimum component size in pixels (default: 100)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading model configuration...")
    config = load_training_config(args.model_path)
    
    # Determine device
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    print("Creating model...")
    model = create_model_from_config(config)
    model = model.to(device)
    
    # Load weights
    print(f"Loading model weights from {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # Get crop size from config
    crop_size = tuple(config.get("crop_size", [7, 64, 64]))
    
    # Create postprocess config if needed
    postprocess_config = None
    if args.postprocess:
        postprocess_config = PostprocessConfig(
            prediction_threshold=args.prediction_threshold,
            minimum_size_px=args.minimum_size,
            recall_threshold=0.5,  # Not used without ground truth
        )
    
    # Run inference
    run_inference(
        model=model,
        image_path=Path(args.image_path),
        output_dir=Path(args.output_dir),
        config=config,
        crop_size=crop_size,
        tile_overlap=args.tile_overlap,
        batch_size=args.batch_size,
        device=device,
        apply_postprocessing=args.postprocess,
        postprocess_config=postprocess_config,
        save_probability=not args.no_probability,
        save_binary=not args.no_binary,
    )
    
    print("\nInference complete!")


if __name__ == "__main__":
    main()