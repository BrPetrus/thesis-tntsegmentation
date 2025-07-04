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
from dataclasses import dataclass

@dataclass
class Config:
    epochs: int
    lr: float
    batch_size: int
    device: str
    num_workers: int
    shuffle: bool
    device: str
    test_size: float = 1/3


def main(input_folder: Path, mask_folder: Path, output_folder: Path, logger: logging.Logger, seed: int, config: Config) -> None:
    # Load metadata
    df = load_dataset_metadata(input_folder, mask_folder)

    # Train/Test Split
    train_x, test_x = train_test_split(df, test_size=1/3., random_state=seed)

    # Create datasets
    train_dataset = TNTDataset(train_x)
    test_dataset = TNTDataset(test_x)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
    )

    # Create net
    nn = UNet3d(1, 1).to(config.device)

    # Training loop
    optimizer = torch.optim.Adam(nn.parameters(), lr=config.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(config.epochs):
        nn.train()
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")):
            inputs, masks = batch
            inputs, masks = inputs.to(config.device), masks.to(config.device)

            optimizer.zero_grad()
            outputs = nn(inputs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Save predictions for the first batch of each epoch
            if batch_idx == 0:
                predictions = torch.sigmoid(outputs).cpu().detach().numpy()
                for i, prediction in enumerate(predictions):
                    prediction_path = output_folder / f"epoch_{epoch+1}_batch_{batch_idx+1}_sample_{i+1}.tiff"
                    plt.imsave(prediction_path, prediction[0], cmap='gray')
                    logger.info(f"Saved prediction for epoch {epoch+1}, batch {batch_idx+1}, sample {i+1} at {prediction_path}")

        logger.info(f"Epoch {epoch+1}/{config.epochs}, Loss: {epoch_loss:.4f}")

        # Save model checkpoint
        checkpoint_path = output_folder / f"model_epoch_{epoch+1}.pth"
        torch.save(nn.state_dict(), checkpoint_path)
        logger.info(f"Model checkpoint saved at {checkpoint_path}")


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
    logging.basicConfig(filename=log_folder / 'training.log', encoding='utf-8', level=logging.DEBUG, format='%(asctime)s %(message)s')
    logger.debug(f"Input folder: {args.input_folder}")
    logger.debug(f"Output folder: {args.output_folder}")
    logger.debug(f"Log folder: {args.log_folder}")
    if not input_folder.exists():
        raise RuntimeError("Specified input folder does not exist!")
    if not mask_folder.exists():
        raise RuntimeError("Specified mask folder does not exist!")
    output_folder.mkdir(parents=True, exist_ok=True)
    log_folder.mkdir(parents=True, exist_ok=True)

    config = Config(
        epochs=10,
        lr=1.0e-3,
        device='cpu',
        num_workers=2,
        shuffle=False,
        batch_size=4,
    )

    # Run training
    main(input_folder, mask_folder, output_folder, logger, args.random_state, config)
    
    
    
