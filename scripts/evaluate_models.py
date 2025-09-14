
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
    
def fetch_model(client: mlflow.MlflowClient, experiment_name: str, run_name: str, model_path: str = 'model', device: str = 'cpu') -> Tuple[nn.Module, Config]:
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

    # Initialise the neural network
    params = run.data.params
    if "neural_network" not in params.keys():
        raise ValueError("Run does not contain metadata about model type")
    neural_network_str = params['neural_network']
    try:
        neural_network = ModelType(neural_network_str)
    except ValueError:
        raise ValueError(f"Unknown model type with str '{neural_network_str}'")
    config = Config(
        **params
    )
    model = create_neural_network(config, 1, 1).to(device)
    
    # Download weights from MLFlow model registry
    model_uri = f'{logged_model.model_uri}/artifacts/data/model.pth'
    print(f"Using {model_uri} path")
    temp_dir = tempfile.mkdtemp()
    print(f"Saving to temporary folder '{temp_dir}'")
    mlflow.artifacts.download_artifacts(artifact_uri=logged_model.model_uri, dst_path=temp_dir)

    # Load weights
    state_dict = torch.load(os.path.join(neural_network_str, 'data', 'model.pth'), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate deep learning models')
    parser.add_argument('run_name', type=str,
                        help='MLflow run name')
    # parser.add_argument('model_type', type=str, required=True,
    #                     choices=[str(model) for model in ModelType],  # TODO
    #                     help='Type of model architecture')
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
    args = parser.parse_args()
        
    # Set up MLFlow
    mlflow.set_tracking_uri(args.mlflow_uri)
    client = mlflow.MlflowClient(args.mlflow_uri)

    # Fetch the model
    model = fetch_model(client, args.experiment_name, args.run_name)

    # Run evaluation
    main(nn, args.database, args.quadrant, args.batch_size, args.device)





    
