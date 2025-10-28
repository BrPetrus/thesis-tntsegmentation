from dataclasses import dataclass
import json
import torch
import numpy as np
import tifffile
from pathlib import Path
import argparse
import pandas as pd
from typing import Tuple, Dict, List, Optional
import sys
import os
import tqdm

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import monai.transforms as MT

from tntseg.nn.models.anisounet3d_basic import AnisotropicUNet3D
from tntseg.nn.models.anisounet3d_seblock import AnisotropicUNet3DSE
from tntseg.nn.models.anisounet3d_csnet import AnisotropicUNet3DCSAM
from tntseg.nn.models.unet3d_basic import UNet3d
from tntseg.nn.models.anisounet3d_usenet import AnisotropicUSENet
from tntseg.utilities.dataset.datasets import MaskType, TNTDataset, load_dataset_metadata
import tntseg.utilities.metrics.metrics as tntmetrics

from tilingutilities import AggregationMethod, stitch_volume, tile_volume
from postprocess import PostprocessConfig, QualityMetrics, TunnelDetectionResult, TunnelMappingResult, create_quality_metrics, detect_tunnels, print_detailed_results

@dataclass
class EvaluationConfig:
    device: str = "cpu"
    crop_size: Tuple[int, int, int] = (7, 64, 64)
    batch_size: int = 2
    dataset_std: float = 0
    dataset_mean: float = 0
    shuffle: bool = False


# Create a config instance
config = EvaluationConfig()

class TiledDataset(Dataset):
    """Dataset that preserves tile positions with data."""

    def __init__(self, tiles_data, tiles_positions):
        self.tiles_data = tiles_data
        self.tiles_positions = tiles_positions

    def __len__(self):
        return len(self.tiles_data)

    def __getitem__(self, idx):
        return self.tiles_data[idx], self.tiles_positions[idx]


# TODO: unify this with the training.py
def create_model_from_args(
    model_type: str,
    model_depth: int,
    base_channels: int,
    channel_growth: int,
    horizontal_kernel: Tuple[int, int, int],
    horizontal_padding: Tuple[int, int, int],
    downscale_kernel,
    downscale_stride,
    upscale_kernel,
    upscale_stride,
    reduction_factor: int = 16,
) -> torch.nn.Module:
    """Create a model based on command line arguments."""

    if model_type == "anisotropicunet":
        return AnisotropicUNet3D(
            n_channels_in=1,
            n_classes_out=1,
            depth=model_depth,
            base_channels=base_channels,
            channel_growth=channel_growth,
            horizontal_kernel=horizontal_kernel,
            horizontal_padding=horizontal_padding,
            upscale_kernel=upscale_kernel,
            upscale_stride=upscale_stride,
            downscale_kernel=downscale_kernel,
            downscale_stride=downscale_stride,
        )
    elif model_type == "anisotropicunet_se":
        return AnisotropicUNet3DSE(
            n_channels_in=1,
            n_classes_out=1,
            depth=model_depth,
            base_channels=base_channels,
            channel_growth=channel_growth,
            horizontal_kernel=horizontal_kernel,
            horizontal_padding=horizontal_padding,
            squeeze_factor=reduction_factor,
            upscale_kernel=upscale_kernel,
            upscale_stride=upscale_stride,
            downscale_kernel=downscale_kernel,
            downscale_stride=downscale_stride,
        )
    elif model_type == "anisotropicunet_csam":
        return AnisotropicUNet3DCSAM(
            n_channels_in=1,
            n_classes_out=1,
            depth=model_depth,
            base_channels=base_channels,
            channel_growth=channel_growth,
            horizontal_kernel=horizontal_kernel,
            horizontal_padding=horizontal_padding,
            upscale_kernel=upscale_kernel,
            upscale_stride=upscale_stride,
            downscale_kernel=downscale_kernel,
            downscale_stride=downscale_stride,
        )
    elif model_type == "anisotropicunet_usenet":
        return AnisotropicUSENet(
            n_channels_in=1,
            n_classes_out=1,
            depth=model_depth,
            base_channels=base_channels,
            channel_growth=channel_growth,
            horizontal_kernel=horizontal_kernel,
            horizontal_padding=horizontal_padding,
            upscale_kernel=upscale_kernel,
            upscale_stride=upscale_stride,
            downscale_kernel=downscale_kernel,
            downscale_stride=downscale_stride,
            squeeze_factor=reduction_factor,
        )
    elif model_type == "basicunet":
        return UNet3d(n_channels_in=1, n_classes_out=1)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def evaluate_predictions(
    predictions: np.ndarray, ground_truth: np.ndarray, threshold: float = 0.5
) -> dict:
    """
    Evaluate predictions against ground truth using tntseg metrics.

    Args:
        predictions: Raw model predictions (probabilities)
        ground_truth: Ground truth masks
        threshold: Threshold for binarizing predictions

    Returns:
        Dictionary containing all calculated metrics
    """
    # Threshold predictions to binary
    binary_predictions = (predictions > threshold).astype(np.uint8) * 255
    binary_ground_truth = (ground_truth > 0.5).astype(np.uint8) * 255

    # Calculate batch statistics using tntseg metrics
    TP, FP, FN, TN = tntmetrics.calculate_batch_stats(
        binary_predictions, binary_ground_truth, negative_val=0, positive_val=255
    )

    # TODO: automate
    # Calculate all metrics
    metrics = {
        "TP": int(TP),
        "FP": int(FP),
        "FN": int(FN),
        "TN": int(TN),
        "jaccard": tntmetrics.jaccard_index(TP, FP, FN),
        "dice": tntmetrics.dice_coefficient(TP, FP, FN),
        "accuracy": tntmetrics.accuracy(TP, FP, FN, TN),
        "precision": tntmetrics.precision(TP, FP),
        "recall": tntmetrics.recall(TP, FN),
        "tversky": tntmetrics.tversky_index(TP, FP, FN, alpha=0.5, beta=0.5),
        "focal_tversky": tntmetrics.focal_tversky_loss(
            TP, FP, FN, alpha=0.5, beta=0.5, gamma=1.0
        ),
    }

    return metrics


def main(
    model: torch.nn.Module,
    database_path: str,
    output_dir: str | Path,
    store_predictions: bool,
    tile_overlap: int,
    visualise_tiling_lines: bool = False,
    run_postprocessing: bool = False,
    postprocess_config: Optional[PostprocessConfig] = None,
    training_config: Dict[str, str] = dict()
) -> None:
    
    if not isinstance(output_dir, str) and not isinstance(output_dir, Path):
        raise ValueError("Invalid output_dir. Expected string or Path type")
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = create_dataset(database_path)
    print(f"Found {len(dataset)} images.")

    all_metrics = []  # Store metrics for each volume
    # volume_results = []  # Store detailed volume results
    postprocess_results = []  # Store postprocessing results

    # Split the data
    for i in range(len(dataset)):
        data, mask = dataset[i]
        print(f"\nProcessing volume {i + 1}/{len(dataset)}")
        
        # # Get volume info for postprocessing
        # volume_info = dataset.dataframe.iloc[i]

        # Split the volume into tiles
        tiles_data, tiles_positions = tile_volume(
            data[0], config.crop_size, overlap=tile_overlap
        )

        tiles_dataset = TiledDataset(tiles_data, tiles_positions)
        tiles_dataloader = DataLoader(
            tiles_dataset, batch_size=config.batch_size, shuffle=False
        )

        # Inference
        all_predictions = []
        all_positions = []
        for batch_data, batch_positions in tqdm.tqdm(
            tiles_dataloader, desc="Running inference..."
        ):
            batch_data = batch_data[:, torch.newaxis, ...]
            batch_data_dev = batch_data.to(config.device)
            predictions = model(batch_data_dev).detach().to("cpu")

            # Store predictions and their positions
            all_predictions.extend(predictions)
            depths = batch_positions[0]
            rows = batch_positions[1]
            cols = batch_positions[2]
            all_positions.extend(
                [(depths[i], rows[i], cols[i]) for i in range(len(batch_positions[0]))]
            )

        # Convert to tensor for stitching
        all_predictions_tensor = torch.stack(all_predictions)

        # Stitch the predictions back together using positions
        out = stitch_volume(
            all_predictions_tensor,
            all_positions,
            original_shape=data.shape[1:],
            aggregation_method=AggregationMethod.Mean,
            visualise_lines=visualise_tiling_lines,
        )
        
        if visualise_tiling_lines:
            reconstructed_volume, stitch_lines_vis = out
            tifffile.imwrite(
                output_dir / "visualised_tiling_lines.tif", stitch_lines_vis
            )
        else:
            reconstructed_volume = out

        # Apply sigmoid to get probabilities
        probabilities = torch.sigmoid(reconstructed_volume).numpy()

        # Threshold for binary predictions
        thresholded = (probabilities > 0.5).astype(np.uint8) * 255

        if store_predictions:
            tifffile.imwrite(output_dir / f"prediction_{i}.tif", probabilities)
            tifffile.imwrite(output_dir / f"threshold_{i}.tif", thresholded)
        
        # Evaluate
        print(f"Evaluating volume {i + 1}...")
        volume_metrics = evaluate_predictions(probabilities, mask[0].numpy())
        volume_metrics["volume_id"] = i
        all_metrics.append(volume_metrics)

        # Print metrics for this volume
        print(f"Volume {i + 1} Metrics:")
        print(f"  Dice: {volume_metrics['dice']:.4f}")
        print(f"  Jaccard: {volume_metrics['jaccard']:.4f}")
        print(f"  Accuracy: {volume_metrics['accuracy']:.4f}")
        print(f"  Precision: {volume_metrics['precision']:.4f}")
        print(f"  Recall: {volume_metrics['recall']:.4f}")
        print(f"  Tversky: {volume_metrics['tversky']:.4f}")

        # Run postprocessing if requested
        if run_postprocessing and postprocess_config is not None:
            print(f"\nRunning postprocessing for volume {i + 1}...")
            
            # Get original image for visualization
            original_image = data[0].numpy()
            
            # Get instance mask (ground truth)
            gt_instance_mask = mask[0].numpy().astype(np.uint8)
            
            # Create volume-specific output directory
            volume_output_dir = output_dir / f"postprocess_volume_{i}"
            
            try:
                # Run tunnel detection
                tunnel_result = detect_tunnels(
                    probabilities,
                    gt_instance_mask, 
                    original_image,
                    postprocess_config,
                    output_folder=volume_output_dir,
                    visualise=True
                )
                
                # Print detailed results for this volume
                print(f"\nPostprocessing Results for Volume {i + 1}:")
                print_detailed_results(tunnel_result)  # TODO fix
                
                # Store results
                postprocess_results.append({
                    'volume_id': i,
                    'tunnel_result': tunnel_result
                })
                
            except Exception as e:
                print(f"Error in postprocessing volume {i + 1}: {e}")
                continue

    # Calculate and display aggregate metrics
    print("\n" + "=" * 50)
    print("AGGREGATE METRICS ACROSS ALL VOLUMES")
    print("=" * 50)

    mean_metrics = {}
    metric_names = [
        "dice", "jaccard", "accuracy", "precision", "recall", "tversky", "focal_tversky",
    ]

    for metric in metric_names:
        values = [m[metric] for m in all_metrics]
        mean_metrics[f"mean_{metric}"] = np.mean(values)
        mean_metrics[f"std_{metric}"] = np.std(values)

    # Save metrics to CSV
    csv_file_path = output_dir / "evaluation_metrics.csv"
    if not csv_file_path.exists():
        write_header = True
    else:
        write_header = False

    with open(csv_file_path, "a") as csv_file:
        if write_header:
            # Define header sections
            basic_info = "database_path;run_name;run_id;model_signature;train_dice;train_jaccard"
            eval_metrics = "eval_dice_mean;eval_dice_std;eval_jaccard_mean;eval_jaccard_std;eval_accuracy_mean;eval_accuracy_std;eval_precision_mean;eval_precision_std;eval_recall_mean;eval_recall_std;eval_tversky_mean;eval_tversky_std;eval_focal_tversky_mean;eval_focal_tversky_std"
            postproc_metrics = "postprocess_overall_dice;postprocess_overall_jaccard;postprocess_overall_precision;postprocess_overall_recall;postprocess_matched_dice;postprocess_matched_jaccard;postprocess_matched_precision;postprocess_matched_recall;postprocess_clean_matched_dice;postprocess_clean_matched_jaccard;postprocess_clean_matched_precision;postprocess_clean_matched_recall"
            tunnel_metrics = "tunnel_tp;tunnel_fp;tunnel_fn;tunnel_precision;tunnel_recall;tunnel_f1;unmatched_predictions;unmatched_labels"
            
            # Combine all sections
            header = f"{basic_info};{eval_metrics};{postproc_metrics};{tunnel_metrics}\n"
            csv_file.write(header)
        

        # Get model info from training config
        run_name = training_config.get("mlflow_run_name")
        run_id = training_config.get("mlflow_run_id")
        model_signature = training_config.get("model_signature")
        train_dice = training_config.get("best_dice", "N/A")
        train_jaccard = training_config.get("best_jaccard", "N/A")
        
        # Calculate aggregate postprocessing metrics
        postprocess_overall_dice = "N/A"
        postprocess_overall_jaccard = "N/A"
        postprocess_overall_precision = "N/A"
        postprocess_overall_recall = "N/A"
        postprocess_matched_dice = "N/A"
        postprocess_matched_jaccard = "N/A"
        postprocess_matched_precision = "N/A"
        postprocess_matched_recall = "N/A"
        postprocess_clean_matched_dice = "N/A"
        postprocess_clean_matched_jaccard = "N/A"
        postprocess_clean_matched_precision = "N/A"
        postprocess_clean_matched_recall = "N/A"
        tunnel_tp = "N/A"
        tunnel_fp = "N/A"
        tunnel_fn = "N/A"
        tunnel_precision = "N/A"
        tunnel_recall = "N/A"
        tunnel_f1 = "N/A"
        unmatched_predictions = "N/A"
        unmatched_labels = "N/A"
        
        if postprocess_results:
            # Calculate mean postprocessing metrics
            overall_metrics = [r['tunnel_result'].metrics_overall for r in postprocess_results]
            matched_metrics = [r['tunnel_result'].metrics_all_matches for r in postprocess_results]
            clean_matched_metrics = [r['tunnel_result'].metrics_one_on_one for r in postprocess_results]
            tunnel_metrics = [r['tunnel_result'].metrics_on_tunnels for r in postprocess_results]
            tunnel_mappings = [r['tunnel_result'].mapping_result for r in postprocess_results]
            
            postprocess_overall_dice = np.mean([m.dice for m in overall_metrics])
            postprocess_overall_jaccard = np.mean([m.jaccard for m in overall_metrics])
            postprocess_overall_precision = np.mean([m.prec for m in overall_metrics])
            postprocess_overall_recall = np.mean([m.recall for m in overall_metrics])
            
            postprocess_matched_dice = np.mean([m.dice for m in matched_metrics])
            postprocess_matched_jaccard = np.mean([m.jaccard for m in matched_metrics])
            postprocess_matched_precision = np.mean([m.prec for m in matched_metrics])
            postprocess_matched_recall = np.mean([m.recall for m in matched_metrics])

            postprocess_clean_matched_dice = np.mean([m.dice for m in clean_matched_metrics])
            postprocess_clean_matched_jaccard = np.mean([m.jaccard for m in clean_matched_metrics])
            postprocess_clean_matched_precision = np.mean([m.prec for m in clean_matched_metrics])
            postprocess_clean_matched_recall = np.mean([m.recall for m in clean_matched_metrics])
            
            tunnel_tp = np.sum([m.tp for m in tunnel_metrics])
            tunnel_fp = np.sum([m.fp for m in tunnel_metrics])
            tunnel_fn = np.sum([m.fn for m in tunnel_metrics])
            tunnel_precision = np.mean([m.prec for m in tunnel_metrics])
            tunnel_recall = np.mean([m.recall for m in tunnel_metrics])
            tunnel_f1 = np.mean([m.f1 for m in tunnel_metrics])

            unmatched_predictions = np.sum([m.unmatched_predictions for m in tunnel_mappings])
            unmatched_labels = np.sum([m.unmatched_labels for m in tunnel_mappings])
        
        # Write the data row
        csv_file.write(
            f"{database_path};{run_name}'{run_id};{model_signature};{train_dice};{train_jaccard};"
            f"{mean_metrics['mean_dice']};{mean_metrics['std_dice']};"
            f"{mean_metrics['mean_jaccard']};{mean_metrics['std_jaccard']};"
            f"{mean_metrics['mean_accuracy']};{mean_metrics['std_accuracy']};"
            f"{mean_metrics['mean_precision']};{mean_metrics['std_precision']};"
            f"{mean_metrics['mean_recall']};{mean_metrics['std_recall']};"
            f"{mean_metrics['mean_tversky']};{mean_metrics['std_tversky']};"
            f"{mean_metrics['mean_focal_tversky']};{mean_metrics['std_focal_tversky']};"
            f"{postprocess_overall_dice};{postprocess_overall_jaccard};{postprocess_overall_precision};{postprocess_overall_recall};"
            f"{postprocess_matched_dice};{postprocess_matched_jaccard};{postprocess_matched_precision};{postprocess_matched_recall};"
            f"{postprocess_clean_matched_dice};{postprocess_clean_matched_jaccard};{postprocess_clean_matched_precision};{postprocess_clean_matched_recall};"
            f"{tunnel_tp};{tunnel_fp};{tunnel_fn};{tunnel_precision};{tunnel_recall};{tunnel_f1};{unmatched_predictions};{unmatched_labels}\n"
        )

    
    # Print results
    print(
        f"Mean Dice: {mean_metrics['mean_dice']:.4f} ± {mean_metrics['std_dice']:.4f}"
    )
    print(
        f"Mean Jaccard: {mean_metrics['mean_jaccard']:.4f} ± {mean_metrics['std_jaccard']:.4f}"
    )
    print(
        f"Mean Accuracy: {mean_metrics['mean_accuracy']:.4f} ± {mean_metrics['std_accuracy']:.4f}"
    )
    print(
        f"Mean Precision: {mean_metrics['mean_precision']:.4f} ± {mean_metrics['std_precision']:.4f}"
    )
    print(
        f"Mean Recall: {mean_metrics['mean_recall']:.4f} ± {mean_metrics['std_recall']:.4f}"
    )
    print(
        f"Mean Tversky: {mean_metrics['mean_tversky']:.4f} ± {mean_metrics['std_tversky']:.4f}"
    )

    # Print aggregate postprocessing results if available
    if postprocess_results:
        print("\n" + "=" * 50)
        print("AGGREGATE POSTPROCESSING METRICS")
        print("=" * 50)
        
        # Calculate mean postprocessing metrics
        tunnel_metrics = [r['tunnel_result'].metrics_on_tunnels for r in postprocess_results]
        overall_metrics = [r['tunnel_result'].metrics_overall for r in postprocess_results]
        
        print(f"Tunnel Detection - Mean Precision: {np.mean([m.prec for m in tunnel_metrics]):.4f}")
        print(f"Tunnel Detection - Mean Recall: {np.mean([m.recall for m in tunnel_metrics]):.4f}")
        print(f"Overall Postprocessed - Mean Dice: {np.mean([m.dice for m in overall_metrics]):.4f}")
        print(f"Overall Postprocessed - Mean Jaccard: {np.mean([m.jaccard for m in overall_metrics]):.4f}")


def load_training_config(model_path: str) -> dict:
    """Load training configuration to get dataset statistics and model parameters."""
    model_dir = Path(model_path).parent
    config_path = model_dir / "config.json"

    if not config_path.exists():
        print(f"Warning: no config found at {config_path}")
        return {}

    with open(config_path, "r") as f:
        config = json.load(f)

    print(f"Loaded configuration from {config_path}")
    print(
        f"Dataset statistics - Mean: {config.get('dataset_mean', 'N/A')}, Std: {config.get('dataset_std', 'N/A')}"
    )
    print(f"Model type: {config.get('model_type', 'N/A')}")

    return config


def create_dataset(database_path: str | Path) -> Dataset:
    if isinstance(database_path, str):
        database_path = Path(database_path)

    # Load the dataset
    df = load_dataset_metadata(
        database_path / "IMG", database_path / "GT"
    )

    # Create the transforms
    transforms = MT.Compose(
        [
            MT.NormalizeIntensityd(
                keys=["volume"],
                subtrahend=config.dataset_mean,
                divisor=config.dataset_std,
            ),
            # Convert to Tensor
            MT.ToTensord(keys=["volume", "mask3d"]),
        ]
    )

    # Create the dataset
    dataset = TNTDataset(df, load_masks=True, transforms=transforms, mask_type=MaskType.instance)
    return dataset


def parse_tuple_arg(arg_string: str, arg_name: str) -> tuple:
    """Parse comma-separated string into tuple of integers."""
    try:
        values = [int(x.strip()) for x in arg_string.split(",")]
        if len(values) != 3:
            raise ValueError(f"{arg_name} must have exactly 3 values")
        return tuple(values)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid {arg_name}: {arg_string}. {str(e)}")


# TODO: reuse code from training utils
def create_model_from_config(config: dict) -> torch.nn.Module:
    """Create a model based on loaded configuration."""
    try:
        model_type = config["model_type"]

        if model_type == "anisotropicunet":
            return AnisotropicUNet3D(
                n_channels_in=1,
                n_classes_out=1,
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
                n_channels_in=1,
                n_classes_out=1,
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
                n_channels_in=1,
                n_classes_out=1,
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
                n_channels_in=1,
                n_classes_out=1,
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
        elif model_type == "unet3d":  # TODO: is this still the right name?
            return UNet3d(n_channels_in=1, n_classes_out=1)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    except Exception as e:
        print(f"Error creating model from config: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate deep learning models")

    # Essential arguments
    parser.add_argument(
        "local_model",
        type=str,
        help="Path to a local model file (config.json should be in same directory)",
    )
    parser.add_argument(
        "database", type=str, help="Path to database folder containing images"
    )
    parser.add_argument("output_dir", type=str, help="Output directory path")

    # Runtime arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for inference (overrides config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run PyTorch on (overrides config)",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        default=False,
        help="Save prediction outputs",
    )
    parser.add_argument(
        "--tile_overlap",
        type=int,
        default=0,
        help="How much (px) of overlap in tiling and stitching",
    )
    parser.add_argument(
        "--visualise_tiling",
        action="store_true",
        default=False,
        help="Visualise tiling and stitching lines",
    )

    # Optional overrides (only needed if you want to override config values)
    parser.add_argument(
        "--override_model_type",
        type=str,
        choices=[
            "anisotropicunet",
            "anisotropicunet_se",
            "anisotropicunet_csam",
            "anisotropicunet_usenet",
            "basicunet",
        ],
        help="Override model type from config",
    )
    parser.add_argument(
        "--override_dataset_mean", type=float, help="Override dataset mean from config"
    )
    parser.add_argument(
        "--override_dataset_std", type=float, help="Override dataset std from config"
    )

    # Add new arguments for postprocessing
    parser.add_argument(
        "--run_postprocessing",
        action="store_true",
        default=False,
        help="Run tunnel-level postprocessing analysis"
    )
    parser.add_argument(
        "--prediction_threshold",
        type=float,
        default=0.5,
        help="Threshold for binarizing predictions in postprocessing"
    )
    parser.add_argument(
        "--recall_threshold",
        type=float,
        default=0.5,
        help="Recall threshold for tunnel matching"
    )
    parser.add_argument(
        "--minimum_size",
        type=int,
        default=100,
        help="Minimum size of a connected component to be considered a real prediction"
    )
    
    args = parser.parse_args()

    # Load training configuration
    training_config = load_training_config(args.local_model)

    if not training_config:
        print("Error: Could not load training configuration!")
        exit(1)

    # Update global config with loaded values
    try:
        config.dataset_mean = (
            args.override_dataset_mean
            if args.override_dataset_mean is not None
            else training_config["dataset_mean"]
        )
        config.dataset_std = (
            args.override_dataset_std
            if args.override_dataset_std is not None
            else training_config["dataset_std"]
        )
        config.device = (
            args.device
            if args.device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        config.batch_size = (
            args.batch_size
            if args.batch_size is not None
            else training_config.get("batch_size", 32)
        )
    except Exception as e:
        print(f"Error loading config values: {e}")
        exit(1)

    print("Using configuration:")
    print(f"  Dataset mean: {config.dataset_mean:.6f}")
    print(f"  Dataset std: {config.dataset_std:.6f}")
    print(f"  Device: {config.device}")
    print(f"  Batch size: {config.batch_size}")

    # Create model from config
    model_config = training_config.copy()
    if args.override_model_type:
        model_config["model_type"] = args.override_model_type

    model = create_model_from_config(model_config)
    if model is None:
        print("Failed to create model from configuration.")
        exit(1)

    model = model.to(config.device)

    # Load model weights
    try:
        model.load_state_dict(torch.load(args.local_model, map_location=config.device))
        model.eval()
        print(
            f"Model ({model_config.get('model_type', 'unknown')}) loaded successfully"
        )
    except Exception as e:
        print(f"Error loading model weights: {e}")
        exit(1)

    # Create postprocess config if needed
    postprocess_config = None
    if args.run_postprocessing:
        postprocess_config = PostprocessConfig(
            prediction_threshold=args.prediction_threshold,
            recall_threshold=args.recall_threshold,
            minimum_size_px=args.minimum_size,
        )

    # Run evaluation
    with torch.no_grad():
        result = main(
            model,
            args.database,
            args.output_dir,
            args.save_predictions,
            args.tile_overlap,
            args.visualise_tiling,
            run_postprocessing=args.run_postprocessing,
            postprocess_config=postprocess_config,
            training_config=training_config
        )
