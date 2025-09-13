
from dataclasses import dataclass
import mlflow.client
import mlflow.pytorch
import torch
import numpy as np
import tifffile
from pathlib import Path
import os
import argparse
import matplotlib.pyplot as plt
import mlflow

import torch.nn as nn
from torch.utils.data import Dataset

from config import ModelType
from tilingutilities import AggregationMethod, stitch_volume, tile_volume

def evaluate_model(nn: nn.Module, data: Dataset) -> None:
    pass

def main(model: nn.Module, database_path: str, quadrant_idx: int, batch_size: int, device: str = 'cpu') -> None:
    pass
    
def fetch_model(client: mlflow.MlflowClient, experiment_name: str, run_name: str, model_path: str = 'model') -> nn.Module:
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

    # Construct the URI
    run_id = runs[0].info.run_id
    model_uri = f'runs:/{run_id}/{model_path}'
    print(f"using model uri: '{model_uri}'")

    # Fetch and load
    model = mlflow.pytorch.load_model(model_uri)
    return model


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





    
