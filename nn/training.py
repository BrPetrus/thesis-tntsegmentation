from utilities.dataset.datasets import TNTDataset
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import logging
import os
from pathlib import Path

config = {
    'epochs': 10,
    'lr': 0.001,
    'batchsize': 4,
}

def main(input_folder: Path, output_folder: Path, logger: logging.Logger) -> None:
    pass

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Training script for neural network.")
    parser.add_argument("--input_folder", type=str, required=True, 
                        help="Path to the input folder containing GT and img.")
    parser.add_argument("--output_folder", type=str, required=True, 
                        help="Path to the output folder where checkpoint and other files will be stored.")
    parser.add_argument("--log_folder", type=str, required=True,
                        help="Path to the out folder where log files are stored.")
    

    # Parse arguments & Setup
    args = parser.parse_args()
    output_folder = Path(args.output_folder)
    input_folder = Path(args.input_folder)
    log_folder = Path(args.log_folder)
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='training.log', encoding='utf-8', level=logging.DEBUG, format='%(asctime)s %(message)s')
    logger.debug(f"Input folder: {args.input_folder}")
    logger.debug(f"Output folder: {args.output_folder}")
    logger.debug(f"Log folder: {args.log_folder}")
    if not input_folder.exists():
        raise RuntimeError("Specified input folder does not exist!")
    output_folder.mkdir(parents=True, exist_ok=True)
    log_folder.mkdir(parents=True, exist_ok=True)

    # Run training
    main(input_folder, output_folder, logger)
    
    
    
