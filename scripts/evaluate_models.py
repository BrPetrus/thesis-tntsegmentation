from dataclasses import dataclass
import mlflow.artifacts
import mlflow.client
import mlflow.pytorch
import mlflow.pytorch
from monai.transforms.croppad import batch
import torch
import numpy as np
import tifffile
from pathlib import Path
import os
import argparse
import matplotlib.pyplot as plt
import mlflow
import tempfile
import csv

from typing import Tuple
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset

from tntseg.nn.models.anisounet3d_basic import AnisotropicUNet3D

from config import ModelType
from tilingutilities import AggregationMethod, stitch_volume, tile_volume
from training_utils import create_neural_network
from config import BaseConfig, ModelType

from torch.utils.data import DataLoader

from tntseg.utilities.dataset.datasets import TNTDataset, load_dataset_metadata
import tqdm

import monai.transforms as MT
import tntseg.utilities.metrics.metrics as tntmetrics

@dataclass
class EvaluationConfig:
    device: str = 'cpu'
    crop_size: Tuple[int, int, int] = (7, 64, 64)
    batch_size: int = 2
    dataset_std: float = 0.07579
    dataset_mean: float = 0.05988
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

def evaluate_model(nn: nn.Module, data: Dataset) -> None:
    pass

def evaluate_predictions(predictions: np.ndarray, ground_truth: np.ndarray, threshold: float = 0.5) -> dict:
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
        binary_predictions, 
        binary_ground_truth, 
        negative_val=0, 
        positive_val=255
    )
    
    # Calculate all metrics
    metrics = {
        'TP': int(TP),
        'FP': int(FP), 
        'FN': int(FN),
        'TN': int(TN),
        'jaccard': tntmetrics.jaccard_index(TP, FP, FN),
        'dice': tntmetrics.dice_coefficient(TP, FP, FN),
        'accuracy': tntmetrics.accuracy(TP, FP, FN, TN),
        'precision': tntmetrics.precision(TP, FP),
        'recall': tntmetrics.recall(TP, FN),
        'tversky': tntmetrics.tversky_index(TP, FP, FN, alpha=0.5, beta=0.5),
        'focal_tversky': tntmetrics.focal_tversky_loss(TP, FP, FN, alpha=0.5, beta=0.5, gamma=1.0)
    }
    
    return metrics

def main(model: nn.Module, database_path: str, output_dir: str | Path, store_predictions: bool) -> None:
    if not isinstance(output_dir, str) and not isinstance(output_dir, Path):
        raise ValueError(f"Invalid output_dir. Expected string or Path type")
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = create_dataset(database_path)
    print(f"Found {len(dataset)} images.")

    all_metrics = []  # Store metrics for each volume
    
    # Split the data
    for i in range(len(dataset)):
        data, mask = dataset[i]
        print(f"\nProcessing volume {i+1}/{len(dataset)}")
        
        # Split the volume into tiles
        tiles_data, tiles_positions = tile_volume(data[0], config.crop_size, overlap=0)

        tiles_dataset = TiledDataset(tiles_data, tiles_positions)
        tiles_dataloader = DataLoader(
            tiles_dataset,
            batch_size=config.batch_size,
            shuffle=False
        )

        # Inference
        all_predictions = []
        all_positions = []
        for batch_data, batch_positions in tqdm.tqdm(tiles_dataloader, desc="Running inference..."):
            # Add fake colour channel
            batch_data = batch_data[:, torch.newaxis, ...]

            # Move data to device
            batch_data_dev = batch_data.to(config.device)

            # # Permute the z-axis (shuffle slices)
            # z_indices = torch.randperm(batch_data_dev.shape[2])
            # batch_data_shuffled = batch_data_dev[:, :, z_indices, :, :].to(config.device)

            # # Run inference on shuffled data
            # shuffled_predictions = model(batch_data_shuffled).detach()

            # # Unshuffle the predictions
            # inverse_indices = torch.zeros(len(z_indices), dtype=torch.long)
            # for i, idx in enumerate(z_indices):
            #     inverse_indices[idx] = i

            # # Apply inverse permutation and move to CPU
            # predictions = shuffled_predictions[:, :, inverse_indices, :, :].to('cpu')
            
            # Run model
            predictions = model(batch_data_dev).detach().to('cpu')
            
            # Store predictions and their positions
            all_predictions.extend(predictions)
            depths = batch_positions[0]
            rows = batch_positions[1]
            cols = batch_positions[2]
            all_positions.extend([(depths[i], rows[i], cols[i]) for i in range(len(batch_positions[0]))])
            
        # Convert to tensor for stitching
        all_predictions_tensor = torch.stack(all_predictions)
        
        # Stitch the predictions back together using positions
        reconstructed_volume = stitch_volume(
            all_predictions_tensor, 
            all_positions,
            original_shape=data.shape[1:],
            aggregation_method=AggregationMethod.Mean
        )

        # Apply sigmoid to get probabilities
        probabilities = torch.sigmoid(reconstructed_volume).numpy()
        
        # Threshold for binary predictions
        thresholded = (probabilities > 0.5).astype(np.uint8) * 255

        if store_predictions:
            tifffile.imwrite(output_dir / f'prediction_{i}.tif', probabilities)
            tifffile.imwrite(output_dir / f'threshold_{i}.tif', thresholded)

        # Evaluate
        print(f"Evaluating volume {i+1}...")
        volume_metrics = evaluate_predictions(probabilities, mask[0].numpy())
        volume_metrics['volume_id'] = i
        all_metrics.append(volume_metrics)
        
        # Print metrics for this volume
        print(f"Volume {i+1} Metrics:")
        print(f"  Dice: {volume_metrics['dice']:.4f}")
        print(f"  Jaccard: {volume_metrics['jaccard']:.4f}")
        print(f"  Accuracy: {volume_metrics['accuracy']:.4f}")
        print(f"  Precision: {volume_metrics['precision']:.4f}")
        print(f"  Recall: {volume_metrics['recall']:.4f}")
        print(f"  Tversky: {volume_metrics['tversky']:.4f}")

    # Calculate and display aggregate metrics
    print("\n" + "="*50)
    print("AGGREGATE METRICS ACROSS ALL VOLUMES")
    print("="*50)
    
    # Calculate mean metrics
    mean_metrics = {}
    metric_names = ['dice', 'jaccard', 'accuracy', 'precision', 'recall', 'tversky', 'focal_tversky']
    
    for metric in metric_names:
        values = [m[metric] for m in all_metrics]
        mean_metrics[f'mean_{metric}'] = np.mean(values)
        mean_metrics[f'std_{metric}'] = np.std(values)
    
    # Save metrics to CSV
    csv_file_path = output_dir / 'evaluation_metrics.csv'
    if not csv_file_path.exists():
        write_header=True
    else:
        write_header=False
    with open(csv_file_path, 'a') as csv_file:
        if write_header:
            csv_file.write('database_path;dice_mean;dice_std;jaccard_mean;jaccard_std;accuracy_mean;accuracy_std;precision_mean;precision_std;recall_mean;recall_std;tversky_mean;tversky_std;focal_tversky_mean;focal_tversky_std\n')
        csv_file.write(
            f'{database_path};'
            f'{mean_metrics['mean_dice']};{mean_metrics['std_dice']};'
            f'{mean_metrics['mean_jaccard']};{mean_metrics['std_jaccard']};'
            f'{mean_metrics['mean_accuracy']};{mean_metrics['std_accuracy']};'
            f'{mean_metrics['mean_precision']};{mean_metrics['std_precision']};'
            f'{mean_metrics['mean_recall']};{mean_metrics['std_recall']};'
            f'{mean_metrics['mean_tversky']};{mean_metrics['std_tversky']};'
            f'{mean_metrics['mean_focal_tversky']};{mean_metrics['std_focal_tversky']};\n'
        )
    # Print results
    print(f"Mean Dice: {mean_metrics['mean_dice']:.4f} ± {mean_metrics['std_dice']:.4f}")
    print(f"Mean Jaccard: {mean_metrics['mean_jaccard']:.4f} ± {mean_metrics['std_jaccard']:.4f}")
    print(f"Mean Accuracy: {mean_metrics['mean_accuracy']:.4f} ± {mean_metrics['std_accuracy']:.4f}")
    print(f"Mean Precision: {mean_metrics['mean_precision']:.4f} ± {mean_metrics['std_precision']:.4f}")
    print(f"Mean Recall: {mean_metrics['mean_recall']:.4f} ± {mean_metrics['std_recall']:.4f}")
    print(f"Mean Tversky: {mean_metrics['mean_tversky']:.4f} ± {mean_metrics['std_tversky']:.4f}")
    

def create_dataset(database_path: str | Path) -> Dataset: 
    if isinstance(database_path, str):
        database_path = Path(database_path)

    # Load the dataset
    df = load_dataset_metadata(database_path / "IMG", database_path / "GT_MERGED_LABELS")    

    # Create the transforms
    transforms = MT.Compose([
        MT.NormalizeIntensityd(
            keys=["volume"],
            subtrahend=config.dataset_mean,
            divisor=config.dataset_std
        ),
        # Convert to Tensor
        MT.ToTensord(keys=['volume', 'mask3d']),
    ])

    # Create the dataset
    dataset = TNTDataset(df, load_masks=True, transforms=transforms)
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate deep learning models')
    parser.add_argument('local_model', type=str, default=None,
                        help="Path to a local model file (bypass MLflow)")
    parser.add_argument('database', type=str,
                        help='Path to database folder containing images')
    parser.add_argument('output_dir', type=str,
                        help='Output directory path')
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument('--device', type=str, default='cpu',
                        help="Device to run PyTorch on")
    parser.add_argument('--save_predictions', action="store_true", default=False)
    # Add a parameter for local model path
    args = parser.parse_args()

    config.device = args.device
    config.batch_size = args.batch_size

    model = AnisotropicUNet3D(
        1, 1, depth=5,
        horizontal_kernel=(1,3,3),
        horizontal_padding=(0,1,1)
    ).to(config.device)
    model.load_state_dict(torch.load(args.local_model, map_location=args.device))
    model.eval()
    print('Model loaded successfully')
    
    # Run evaluation
    with torch.no_grad():
        main(model, args.database, args.output_dir, args.save_predictions)






