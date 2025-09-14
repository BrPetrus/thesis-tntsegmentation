from dataclasses import dataclass
import mlflow.artifacts
import mlflow.client
import mlflow.pytorch
import mlflow.pytorch
import torch
import numpy as np
import tifffile
from pathlib import Path
import os
import argparse
import matplotlib.pyplot as plt
import mlflow
import tempfile

from typing import Tuple
import torch.nn as nn
from torch.utils.data import Dataset

from config import ModelType
from tilingutilities import AggregationMethod, stitch_volume, tile_volume
from training_utils import create_neural_network
from config import Config, ModelType

def evaluate_model(nn: nn.Module, data: Dataset) -> None:
    pass

def main(model: nn.Module, database_path: str, quadrant_idx: int, batch_size: int, device: str = 'cpu') -> None:
    pass
    
def fetch_model_mlflow(client: mlflow.MlflowClient, experiment_name: str, run_name: str, model_path: str = 'model', device: str = 'cpu') -> nn.Module:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f'Experiment with name "{experiment_name}" not found.')
    
    # Find the run
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
    )
    if not runs:
        raise ValueError(f'No runs in experiment \'{experiment_name}\' with run name \'{run_name}\'')
    assert len(runs) == 1
    run = runs[0]

    # Now find the model ID
    model_id = run.outputs.model_outputs[0].model_id

    # Construct the URI
    logged_model = client.get_logged_model(model_id=model_id)
    
    # Download weights from MLFlow model registry
    model_uri = f'{logged_model.model_uri}/artifacts/data/model.pth'
    print(f"Using {model_uri} path")
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Saving to temporary folder '{temp_dir}'")
        mlflow.artifacts.download_artifacts(artifact_uri=logged_model.model_uri, dst_path=temp_dir)

        # Load weights
        model = torch.load(os.path.join(temp_dir, 'data', 'model.pth'), map_location=device, weights_only=False)

    model.eval()
    return model

def fetch_model_local(model_path: str, device: str = 'cpu') -> nn.Module:
    """Load a model from a local file path.
    
    Args:
        model_path: Path to the model .pth file
        device: Device to load the model on
        
    Returns:
        Loaded PyTorch model
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at '{model_path}'")
    
    print(f"Loading model from '{model_path}'")
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate deep learning models')
    parser.add_argument('run_name', type=str,
                        help='MLflow run name')
    parser.add_argument('database', type=str,
                        help='Path to database folder containing images')
    parser.add_argument('quadrant', type=int,
                        choices=[1, 2, 3, 4],
                        help='Quadrant number to evaluate (1-4)')
    parser.add_argument('--experiment_name', type=str, default='Default',
                        help="Name of the MLFlow experiment")
    parser.add_argument('--mlflow_uri', type=str, default="http://127.0.0.1:8000",
                        help="MLFlow URI")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument('--device', type=str, default='cpu',
                        help="Device to run PyTorch on")
    # Add a parameter for local model path
    parser.add_argument('--local_model', type=str, default=None,
                        help="Path to a local model file (bypass MLflow)")
    args = parser.parse_args()
    
    # Fetch the model (either local or from MLflow)
    if args.local_model:
        model = fetch_model_local(args.local_model, device=args.device)
    else:
        # Set up MLFlow
        mlflow.set_tracking_uri(args.mlflow_uri)
        client = mlflow.MlflowClient(args.mlflow_uri)
        model = fetch_model_mlflow(client, args.experiment_name, args.run_name, device=args.device)
    
    print('Model loaded successfully')
    
    # Run evaluation
    main(model, args.database, args.quadrant, args.batch_size, args.device)






