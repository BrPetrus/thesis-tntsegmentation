import argparse
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import tifffile
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from tntseg.nn.models.unet3d_basic import UNet3d
from tntseg.utilities.dataset.datasets import TNTDataset, load_dataset_metadata

# --- Configuration ---
# These values should match the training configuration for normalization
DATASET_MEAN = 0.05988
DATASET_STD = 0.07579
# The model expects a specific input size, ensure this matches your training
CROP_SIZE = (7, 64, 64) 

def setup_logging(output_folder: Path):
    """Set up logging to file and console."""
    log_file = output_folder / 'inference.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main(checkpoint_path: Path, input_folder: Path, output_folder: Path, device: str, batch_size: int):
    """
    Run inference on a dataset using a trained model checkpoint.

    Args:
        checkpoint_path: Path to the model .pth checkpoint file.
        input_folder: Path to the folder containing images for inference.
        output_folder: Path where predicted masks will be saved.
        device: The device to run inference on ('cuda' or 'cpu').
        batch_size: Number of samples to process at once.
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_folder)

    logger.info(f"Starting inference...")
    logger.info(f"  - Checkpoint: {checkpoint_path}")
    logger.info(f"  - Input folder: {input_folder}")
    logger.info(f"  - Output folder: {output_folder}")
    logger.info(f"  - Device: {device}")
    logger.info(f"  - Batch size: {batch_size}")

    # --- 1. Load Model ---
    logger.info("Loading model...")
    model = UNet3d(in_channels=1, out_channels=1)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Model state loaded successfully from checkpoint.")
    except FileNotFoundError:
        logger.error(f"Checkpoint file not found at {checkpoint_path}")
        return
    except KeyError:
        logger.error("Checkpoint is missing 'model_state_dict'. Please ensure it was saved correctly.")
        return
        
    model.to(device)
    model.eval()

    # --- 2. Prepare Dataset ---
    logger.info("Preparing dataset...")
    try:
        # Create a DataFrame of image paths, no masks needed
        inference_df = load_dataset_metadata(img_folder=str(input_folder))
    except ValueError as e:
        logger.error(f"Error loading dataset: {e}")
        return

    # Define transformations (must match validation/test transforms from training)
    transforms_inference = A.Compose([
        A.Normalize(mean=DATASET_MEAN, std=DATASET_STD, max_pixel_value=1.0, p=1.0),
        A.CenterCrop(height=CROP_SIZE[1], width=CROP_SIZE[2]), # Assuming 2D crop for now, adjust if 3D
        ToTensorV2()
    ])

    # The dataset should not load masks and should return file paths
    inference_dataset = TNTDataset(
        dataframe=inference_df, 
        load_masks=False, 
        transforms=transforms_inference,
        return_filepath=True # Important: ensure your dataset can return file paths
    )
    
    inference_dataloader = DataLoader(
        inference_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    logger.info(f"Found {len(inference_dataset)} images to process.")

    # --- 3. Run Inference ---
    logger.info("Running inference and saving predictions...")
    with torch.no_grad():
        for batch in tqdm(inference_dataloader, desc="Inference"):
            inputs, filepaths = batch
            inputs = inputs.to(device)

            # Get model predictions (logits)
            outputs = model(inputs)

            # Convert logits to probabilities
            probs = torch.sigmoid(outputs)

            # Apply threshold to get binary mask
            preds = (probs > 0.5).cpu().numpy()

            # Save each prediction in the batch
            for i in range(preds.shape[0]):
                prediction_mask = preds[i].squeeze() # Remove channel dimension
                
                # Convert float mask (0.0, 1.0) to uint8 (0, 255) for saving
                prediction_mask_uint8 = (prediction_mask * 255).astype(np.uint8)

                original_path = Path(filepaths[i])
                output_path = output_folder / original_path.name
                
                tifffile.imwrite(output_path, prediction_mask_uint8)

    logger.info(f"Inference complete. Predictions saved to {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference for 3D U-Net segmentation.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the trained model checkpoint (.pth).")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the folder with images for inference.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the folder where predicted masks will be saved.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use ('cuda' or 'cpu').")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference.")
    
    args = parser.parse_args()

    main(
        checkpoint_path=Path(args.checkpoint_path),
        input_folder=Path(args.input_folder),
        output_folder=Path(args.output_folder),
        device=args.device,
        batch_size=args.batch_size
    )