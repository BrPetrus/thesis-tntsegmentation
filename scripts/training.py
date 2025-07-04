from tntseg.utilities.dataset.datasets import TNTDataset, load_dataset_metadata
from tntseg.nn.models.unet3d_basic import UNet3d
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import logging
import os
from pathlib import Path
from sklearn.model_selection import train_test_split


config = {
    'epochs': 10,
    'lr': 0.001,
    'batchsize': 4,
}

def main(input_folder: Path, mask_folder: Path, output_folder: Path, logger: logging.Logger, seed: int) -> None:
    # Load data
    df = load_dataset_metadata(input_folder, mask_folder)

    # Split
    train_x, test_x = train_test_split(df, test_size=1/3., random_state=seed)


    # Create net
    # nn = UNet3d(1, 1)
    
    

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Training script for neural network.")
    parser.add_argument("--input_folder", type=str, required=True, 
                        help="Path to the input folder containing images.")
    parser.add_argument("--mask_folder", type=str, required=True,
                        help="Path to the folder containing masks.")
    parser.add_argument("--output_folder", type=str, required=True, 
                        help="Path to the output folder where checkpoint and other files will be stored.")
    parser.add_argument("--log_folder", type=str, required=True,
                        help="Path to the out folder where log files are stored.")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Seed value for the train/test split.")
    

    # Parse arguments & Setup
    args = parser.parse_args()
    output_folder = Path(args.output_folder)
    input_folder = Path(args.input_folder)
    log_folder = Path(args.log_folder)
    mask_folder = Path(args.mask_folder)
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='training.log', encoding='utf-8', level=logging.DEBUG, format='%(asctime)s %(message)s')
    logger.debug(f"Input folder: {args.input_folder}")
    logger.debug(f"Output folder: {args.output_folder}")
    logger.debug(f"Log folder: {args.log_folder}")
    if not input_folder.exists():
        raise RuntimeError("Specified input folder does not exist!")
    if not mask_folder.exists():
        raise RuntimeError("Specified mask folder does not exist!")
    output_folder.mkdir(parents=True, exist_ok=True)
    log_folder.mkdir(parents=True, exist_ok=True)

    # Run training
    main(input_folder, mask_folder, output_folder, logger, args.random_state)
    
    
    
