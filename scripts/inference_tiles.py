import logging
import torch
from pathlib import Path
import argparse
import tifffile
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

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

def main(checkpoint_path: Path, input_folder: Path, output_folder: Path, batch_size: int = 32):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_folder.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_folder)
    logger.info(f"Starting inference...")
    logger.info(f"  - Checkpoint: {checkpoint_path}")
    logger.info(f"  - Input folder: {input_folder}")
    logger.info(f"  - Output folder: {output_folder}")
    logger.info(f"  - Device: {device}")
    logger.info(f"  - Batch size: {batch_size}")

    # Prepare the model
    logger.info("Loading model...")
    model = UNet3d(n_channels_in=1, n_classes_out=1)
    try:
        model = torch.load(checkpoint_path, weights_only=False, map_location=torch.device(device))
        # model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Model state loaded")
    except FileNotFoundError:
        logger.error(f"Path {checkpoint_path} does not exists")
        return
    model.to(device)
    model.eval()

    # Prepare the dataset
    logger.info("Preparing dataset...")
    df = load_dataset_metadata(img_folder=input_folder)

    # Load the images, split them into CROP_SIZE chunks and export them
    for img_path in df['img_path']:
        # img_path = row['img_path']
        try:
            img = tifffile.imread(img_path)
        except:
            logger.error(f"Could not load img at {img_path}")
            continue

        # Tile the img
        i = 0
        number_of_rows = img.shape[1] // CROP_SIZE[1]
        number_of_cols = img.shape[2] // CROP_SIZE[2] 
        tile_output_path = output_folder / "tiles"
        tile_output_path.mkdir(parents=True, exist_ok=True)
        for r in range(img.shape[1] // CROP_SIZE[1]):
            for c in range(img.shape[2] // CROP_SIZE[2]):
                tile = img[:, r * CROP_SIZE[1]:(r+1) * CROP_SIZE[1], c*CROP_SIZE[2]:(c+1)*CROP_SIZE[2]]
                tifffile.imwrite(tile_output_path / f"{i:10}_r-{r}_c-{c}.tif", tile)
                i += 1

    # Create the TNTDataset object
    transforms = A.Compose([
        A.Normalize(mean=DATASET_MEAN, std=DATASET_STD, max_pixel_value=1.0, p=1.0),
        A.CenterCrop(height=CROP_SIZE[1], width=CROP_SIZE[2]),
        ToTensorV2()
    ])
    df_tiles = load_dataset_metadata(img_folder=str(tile_output_path))
    dataset = TNTDataset(
        dataframe = df_tiles,
        load_masks=False,
        transforms=transforms,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    logger.info(f"Found {len(dataset)} tiles to process")

    # Run the inference on tiles
    logger.info("Running inference and saving the predictions...")
    output_inference_path = output_folder / "predictions"
    output_inference_path.mkdir(exist_ok=True, parents=True)
    img_idx = 0
    all_tiles = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            batch = batch.to(device)[:, np.newaxis, ...]  # Add channel
            outputs = model(batch)
            prob = torch.sigmoid(outputs)
            thresh = (prob > 0.5).detach().cpu().numpy()

            # Save
            for i in range(thresh.shape[0]):
                prediction_mask = thresh[i].squeeze()  # Remove channel
                prob_float = prob[i].detach().cpu().numpy().squeeze().astype(np.float32)
                prediction_mask_uint8 = (prediction_mask * 255).astype(np.uint8)
                tifffile.imwrite(output_inference_path / f"{img_idx:10}.tif", prediction_mask_uint8)
                tifffile.imwrite(output_inference_path / f"{img_idx:10}-sigmoid.tif", prob_float)
                img_idx += 1
                all_tiles.append(prediction_mask_uint8)
    logger.info("Inference complete") 

    # Stiching it together
    logger.info("Stitching tiles together...")
    tiles_per_image = number_of_rows * number_of_cols
    num_images = len(all_tiles) // tiles_per_image
    stitched_output_folder = output_folder / "stitched"
    stitched_output_folder.mkdir(parents=True, exist_ok=True)
    for img_idx in range(num_images):
        full_height = CROP_SIZE[1]*number_of_rows
        full_width = CROP_SIZE[2]*number_of_cols
        relevant_tiles = np.array(all_tiles[img_idx*tiles_per_image:(img_idx+1)*tiles_per_image])
        # img = np.array(img)
        # assert CROP_SIZE[0] == 7
        # img = img.reshape(7, full_height, full_width)
        stitched = np.zeros((7, full_height, full_width)).astype(np.uint8)

        for r in range(number_of_rows):
            for c in range(number_of_cols):
                stitched[:, r*CROP_SIZE[1]:(r+1)*CROP_SIZE[1], c*CROP_SIZE[2]:(c+1)*CROP_SIZE[2]] = relevant_tiles[r*number_of_rows+c]

        

        tifffile.imwrite(stitched_output_folder / f"img_idx.tif", stitched)
    logger.info("Done stitching")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on 3D images')
    parser.add_argument('checkpoint_path', type=Path, help='Path to model checkpoint')
    parser.add_argument('input_folder', type=Path, help='Folder containing input images')
    parser.add_argument('output_folder', type=Path, help='Folder for output predictions')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    args = parser.parse_args()

    main(args.checkpoint_path, args.input_folder, args.output_folder, args.batch_size)
