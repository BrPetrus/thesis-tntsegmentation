import logging
import torch
from pathlib import Path
import argparse
import tifffile
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
import os

from tntseg.nn.models.unet3d_basic import UNet3d
from tntseg.utilities.dataset.datasets import TNTDataset, load_dataset_metadata

DATASET_MEAN = 0.05988
DATASET_STD = 0.07578
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

def main(checkpoint_path: Path, input_folder: Path, output_folder: Path, batch_size: int = 32, 
         quad_mode: bool = False, quad_idx: int = None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_folder.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_folder)
    logger.info(f"Starting inference...")
    logger.info(f"  - Checkpoint: {checkpoint_path}")
    logger.info(f"  - Input folder: {input_folder}")
    logger.info(f"  - Output folder: {output_folder}")
    logger.info(f"  - Device: {device}")
    logger.info(f"  - Batch size: {batch_size}")
    if quad_mode:
        logger.info(f"  - Quad mode: {quad_mode}, quad index: {quad_idx}")

    # Prepare the model
    logger.info("Loading model...")
    model = UNet3d(n_channels_in=1, n_classes_out=1)
    try:
        model = torch.load(checkpoint_path, map_location=device, weights_only=False)
        # model.load_state_dict(checkpoint)
        logger.info("Model state loaded")
    except Exception as e:
        logger.error(f"Could not load checkpoint: {e}")
        return
    model.to(device)
    model.eval()

    # Prepare transforms
    transforms = A.Compose([
        A.Normalize(mean=DATASET_MEAN, std=DATASET_STD, max_pixel_value=1.0, p=1.0),
        A.ToTensorV2()
    ])

    # Prepare the dataset with tiling (and optional quad extraction)
    logger.info("Preparing dataset...")
    df = load_dataset_metadata(img_folder=input_folder)
    dataset = TNTDataset(
        dataframe=df,
        load_masks=False,
        transforms=transforms,
        tile=True,
        tile_size=CROP_SIZE,
        overlap=0,
        quad_mode=quad_mode,
        quad_idx=quad_idx
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    logger.info(f"Found {len(dataset)} tiles to process")

    # Run inference
    logger.info("Running inference...")
    predictions = []
    metadata = []
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Inference"):
            batch, meta = batch_data  # Unpack (volume, metadata)
            batch = batch.to(device)[:, np.newaxis, ...]  # Add channel
            
            outputs = model(batch)
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            
            # Store predictions and metadata for stitching
            for i in range(probs.shape[0]):
                predictions.append(probs[i, 0])  # Remove channel dimension
                # Extract the i-th value for each key in meta and store as a flat dict
                metadata.append({k: v[i] for k, v in meta.items()})

    logger.info("Inference complete")
    
    # Save individual tiles if needed
    tiles_output_path = output_folder / "tiles"
    tiles_output_path.mkdir(exist_ok=True, parents=True)
    
    logger.info("Saving individual tiles...")
    for i, (pred, meta) in enumerate(zip(predictions, metadata)):
        # Generate a meaningful filename using metadata
        if quad_mode:
            tile_name = f"img{meta['image_idx']}_quad{meta['quad_idx']}"
            if 'row' in meta:
                tile_name += f"_r{meta['row']}_c{meta['col']}"
        else:
            tile_name = f"img{meta['image_idx']}"
            if 'row' in meta:
                tile_name += f"_r{meta['row']}_c{meta['col']}"
        
        # Save probability map
        prob_path = tiles_output_path / f"{tile_name}_prob.tif"
        tifffile.imwrite(str(prob_path), pred.astype(np.float32))
        
        # Save binary prediction
        binary_pred = (pred > 0.5).astype(np.uint8) * 255
        binary_path = tiles_output_path / f"{tile_name}_binary.tif"
        tifffile.imwrite(str(binary_path), binary_pred)
    
    # Stitch the predictions
    logger.info("Stitching predictions...")
    stitched_output_path = output_folder / "stitched"
    stitched_output_path.mkdir(exist_ok=True, parents=True)
    
    # Use TNTDataset's stitch_predictions method
    binary_preds = [(pred > 0.5).astype(np.uint8) * 255 for pred in predictions]
    
    # Stitch probability maps
    prob_stitched = TNTDataset.stitch_predictions(
        predictions, metadata, output_path=str(stitched_output_path / "prob")
    )
    
    # Stitch binary predictions
    binary_stitched = TNTDataset.stitch_predictions(
        binary_preds, metadata, output_path=str(stitched_output_path / "binary")
    )
    
    logger.info("Stitching complete!")
    logger.info(f"Results saved to {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on 3D images')
    parser.add_argument('checkpoint_path', type=Path, help='Path to model checkpoint')
    parser.add_argument('input_folder', type=Path, help='Folder containing input images')
    parser.add_argument('output_folder', type=Path, help='Folder for output predictions')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--quad_mode', action='store_true', help='Enable quadrant mode')
    parser.add_argument('--quad_idx', type=int, choices=[0, 1, 2, 3], 
                       help='Quadrant index (0: top-left, 1: top-right, 2: bottom-left, 3: bottom-right)')
    
    args = parser.parse_args()
    
    if args.quad_mode and args.quad_idx is None:
        parser.error("--quad_mode requires --quad_idx")
    
    main(args.checkpoint_path, args.input_folder, args.output_folder, 
         args.batch_size, args.quad_mode, args.quad_idx)
