from tntseg.utilities.dataset.datasets import TNTDataset, load_dataset_metadata
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import logging
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
import tifffile
import mlflow
from sklearn import metrics as skmetrics
import tntseg.utilities.metrics.metrics as tntmetrics
from typing import Dict, List, Tuple
import monai.transforms as MT
from monai.utils import set_determinism
import json
from dataclasses import asdict

from scripts.training_utils import (
    create_neural_network,
    create_loss_criterion,
    visualize_transform_effects,
)
from config import AnisotropicUNetConfig, AnisotropicUNetSEConfig, BaseConfig, ModelType


def set_all_seeds(seed: int):
    """Set seeds for all random number generators"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    set_determinism(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def worker_init_fn(worker_id):
    """Initialise worker with different seed"""
    worker_seed = torch.initial_seed() % (2**32) + worker_id

    # Set seeds for all relevant libraries
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    set_determinism(seed=worker_seed)


def _analyze_dataset_stats(input_folder: Path) -> Tuple[float, float]:
    """Analyze dataset to calculate mean and std"""
    logging.info("Analyzing dataset")

    # Load only training data
    input_folder_train = input_folder / "train"
    if not input_folder_train.exists():
        raise ValueError(f"Missing train subfolder at '{input_folder_train}'")
    df_train = load_dataset_metadata(input_folder_train / "IMG")
    logging.info(f"Loading {len(df_train)} training samples for analysis...")
    all_images = []
    for idx, row in tqdm(
        df_train.iterrows(), total=len(df_train), desc="Loading training images"
    ):
        # NOTE: no normalisation is used
        img = tifffile.imread(row["img_path"]).astype(np.float32)
        all_images.append(img.flatten())

    # Estimate statisti
    all_images = np.concatenate(all_images)
    dataset_mean = float(np.mean(all_images))
    dataset_std = float(np.std(all_images))
    logging.info(
        f"Training datatset ({len(df_train)} samples) statistics: \n"
        f"   Mean: {dataset_mean}\n"
        f"   std: {dataset_std}"
    )
    return dataset_mean, dataset_std


def _prepare_datasets(
    input_folder: Path, seed: int, config: BaseConfig, validation_ratio=1 / 3.0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Load metadata
    input_folder_train = input_folder / "train"
    if not input_folder_train.exists():
        raise RuntimeError(f"Missing train subfolder at '{input_folder_train}'")
    input_folder_test = input_folder / "test"
    if not input_folder_test.exists():
        raise RuntimeError(f"Missing testing subfolder at '{input_folder_test}'")

    # Analyse for mean and std
    calculated_mean, calculated_std = _analyze_dataset_stats(input_folder)
    config.dataset_mean = calculated_mean
    config.dataset_std = calculated_std

    # Load datset properly
    df_train = load_dataset_metadata(
        input_folder_train / "IMG", input_folder_train / "GT_MERGED_LABELS"
    )
    test_x = load_dataset_metadata(
        input_folder_test / "IMG", input_folder_test / "GT_MERGED_LABELS"
    )

    # Train/Test Split
    train_x, valid_x = train_test_split(
        df_train, test_size=validation_ratio, random_state=seed
    )

    logger.info(f"Train {len(train_x)} Test {len(test_x)} Validation {len(valid_x)}")
    total = len(train_x) + len(test_x) + len(valid_x)
    logger.info(
        f"Train {len(train_x) / total * 100}% Test {len(test_x) / total * 100}% Validation {len(valid_x) / total * 100}%"
    )

    # Define transforms
    transforms_train = MT.Compose(
        [
            MT.ToMetaTensord(keys=["volume", "mask3d"]),
            # Normalise
            MT.NormalizeIntensityd(
                keys=["volume"],
                subtrahend=config.dataset_mean,
                divisor=config.dataset_std,
            ),
            # Random noise
            MT.RandGaussianNoised(keys=["volume"], prob=0.5, mean=0, std=0.01),
            # Random flips
            MT.RandFlipd(keys=["volume", "mask3d"], prob=0.5, spatial_axis=0),
            MT.RandFlipd(keys=["volume", "mask3d"], prob=0.5, spatial_axis=1),
            MT.RandFlipd(keys=["volume", "mask3d"], prob=0.5, spatial_axis=2),
            MT.RandSpatialCropd(keys=["volume", "mask3d"], roi_size=config.crop_size),
            # Convert to Tensor
            MT.ToTensord(keys=["volume", "mask3d"]),
        ]
    )

    transform_test = MT.Compose(
        [
            MT.NormalizeIntensityd(
                keys=["volume"],
                subtrahend=config.dataset_mean,
                divisor=config.dataset_std,
            ),
            MT.CenterSpatialCropd(keys=["volume", "mask3d"], roi_size=config.crop_size),
            # Convert to Tensor
            MT.ToTensord(keys=["volume", "mask3d"]),
        ]
    )

    # Create datasets
    train_dataset = TNTDataset(train_x, transforms=transforms_train)
    test_dataset = TNTDataset(test_x, transforms=transform_test)
    valid_dataset = TNTDataset(valid_x, transforms=transform_test)

    # Create dataloaders
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        worker_init_fn=worker_init_fn,
        generator=generator,  # For reproducibility
        persistent_workers=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        persistent_workers=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        persistent_workers=True,
    )
    return train_dataloader, test_dataloader, valid_dataloader


# TODO: think abou treplacing with batchstats from utlities metrics
def _calculate_metrics(
    nn: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    epoch: int,
    prefix: str,
    output_folder: Path,
    save_results: bool = False,
) -> Tuple[int, int, int, int, float]:
    # Run evaluation on validation set
    with torch.no_grad():
        nn.eval()
        loss = 0.0
        total = 0
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation")):
            inputs, masks = batch
            inputs, masks = inputs.to(config.device), masks.to(config.device)

            outputs = nn(inputs)
            batch_loss = criterion(outputs, masks)
            loss += batch_loss

            # Evaluate metrics
            total += np.prod(inputs.shape)
            thresholded = torch.sigmoid(outputs) > 0.5
            TP += ((thresholded == True) & (masks == True)).float().sum()
            TN += ((thresholded == False) & (masks == False)).float().sum()
            FP += ((thresholded == True) & (masks == False)).float().sum()
            FN += ((thresholded == False) & (masks == True)).float().sum()
            assert total == sum([TP, TN, FP, FN])

            # Save first batch
            if batch_idx == 0 and save_results:
                predictions = torch.sigmoid(outputs).cpu().detach().numpy()
                inputs_np = inputs.cpu().detach().numpy()
                masks_np = masks.cpu().detach().numpy()

                for i, (prediction, input_img, mask) in enumerate(
                    zip(predictions, inputs_np, masks_np)
                ):
                    prediction_path = (
                        output_folder
                        / f"epoch_{prefix}_{epoch + 1}_batch_{batch_idx + 1}_sample_{i + 1}_prediction.tiff"
                    )
                    input_path = (
                        output_folder
                        / f"epoch_{prefix}_{epoch + 1}_batch_{batch_idx + 1}_sample_{i + 1}_input.tiff"
                    )
                    mask_path = (
                        output_folder
                        / f"epoch_{prefix}_{epoch + 1}_batch_{batch_idx + 1}_sample_{i + 1}_mask.tiff"
                    )
                    # Save prediction
                    tifffile.imwrite(
                        prediction_path, prediction[0, ...].astype(np.float32)
                    )
                    logger.debug(
                        f"Saved prediction for epoch {epoch + 1}, batch {batch_idx + 1}, sample {i + 1} at {prediction_path}"
                    )

                    # Save input image
                    tifffile.imwrite(input_path, input_img[0, ...].astype(np.float32))
                    logger.debug(
                        f"Saved input image for epoch {epoch + 1}, batch {batch_idx + 1}, sample {i + 1} at {input_path}"
                    )

                    # Save mask
                    tifffile.imwrite(mask_path, mask[0, ...].astype(np.float32))
                    logger.debug(
                        f"Saved mask for epoch {epoch + 1}, batch {batch_idx + 1}, sample {i + 1} at {mask_path}"
                    )
    return TP, TN, FP, FN, loss


def _train_single_epoch(
    nn, optimizer, criterion, train_dataloader, config, epoch, output_folder: Path
):
    epoch_loss = 0.0
    for batch_idx, batch in enumerate(
        tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{config.epochs}")
    ):
        inputs, masks = batch
        inputs, masks = inputs.to(config.device), masks.to(config.device)

        optimizer.zero_grad()
        outputs = nn(inputs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # Save predictions, inputs, and masks for the first batch of each epoch
        if batch_idx == 0 and epoch % 100 == 0:
            predictions = torch.sigmoid(outputs).cpu().detach().numpy()
            inputs_np = inputs.cpu().detach().numpy()
            masks_np = masks.cpu().detach().numpy()

            for i, (prediction, input_img, mask) in enumerate(
                zip(predictions, inputs_np, masks_np)
            ):
                prediction_path = (
                    output_folder
                    / f"epoch_{epoch + 1}_batch_{batch_idx + 1}_sample_{i + 1}_prediction.tiff"
                )
                input_path = (
                    output_folder
                    / f"epoch_{epoch + 1}_batch_{batch_idx + 1}_sample_{i + 1}_input.tiff"
                )
                mask_path = (
                    output_folder
                    / f"epoch_{epoch + 1}_batch_{batch_idx + 1}_sample_{i + 1}_mask.tiff"
                )
                # Save prediction
                tifffile.imwrite(prediction_path, prediction[0, ...].astype(np.float32))
                logger.debug(
                    f"Saved prediction for epoch {epoch + 1}, batch {batch_idx + 1}, sample {i + 1} at {prediction_path}"
                )

                # Save input image
                tifffile.imwrite(input_path, input_img[0, ...].astype(np.float32))
                logger.debug(
                    f"Saved input image for epoch {epoch + 1}, batch {batch_idx + 1}, sample {i + 1} at {input_path}"
                )

                # Save mask
                tifffile.imwrite(mask_path, mask[0, ...].astype(np.float32))
                logger.debug(
                    f"Saved mask for epoch {epoch + 1}, batch {batch_idx + 1}, sample {i + 1} at {mask_path}"
                )
    return epoch_loss


def _train(
    nn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    config: BaseConfig,
    output_folder: Path,
) -> None:
    # Last time that the eval loss improved
    epochs_since_last_improvement = 0
    last_better_eval_loss = np.inf

    for epoch in range(config.epochs):
        nn.train()
        epoch_loss = _train_single_epoch(
            nn, optimizer, criterion, train_dataloader, config, epoch, output_folder
        )

        TP, TN, FP, FN, val_loss = _calculate_metrics(
            nn,
            valid_dataloader,
            criterion,
            epoch,
            "val",
            output_folder,
            save_results=epoch % 100 == 0,
        )
        # Calculate metrics
        accuracy = tntmetrics.accuracy(TP, FP, FN, TN)
        precision = tntmetrics.precision(TP, FP)
        recall = tntmetrics.recall(TP, FN)
        jaccard = tntmetrics.jaccard_index(TP, FP, FN)
        dice = tntmetrics.dice_coefficient(TP, FP, FN)
        tversky = tntmetrics.tversky_index(
            TP, FP, FN, config.eval_tversky_alpha, config.eval_tversky_beta
        )
        focal_tversky = tntmetrics.focal_tversky_loss(
            TP,
            FP,
            FN,
            config.eval_tversky_alpha,
            config.eval_tversky_beta,
            config.eval_tversky_gamma,
        )

        mlflow.log_metrics(
            {
                "train/loss": epoch_loss,
                "val/loss": val_loss,
                "val/accuracy": accuracy,
                "val/precision": precision,
                "val/recall": recall,
                "val/jaccard": jaccard,
                "val/dice": dice,
                "val/tversky": tversky,
                "val/focaltversky": focal_tversky,
            },
            step=epoch,
        )
        logger.info(
            f"Epoch {epoch + 1}/{config.epochs}, Train/Loss: {epoch_loss:.4f}  Val/Loss: {val_loss}"
        )
        logger.info(
            f"Epoch {epoch + 1}/{config.epochs}, Val/Acc: {accuracy}, Val/Prec: {precision}, Val/Recall: {recall}"
        )
        logger.info(f"Last improved {epochs_since_last_improvement} epochs ago.")

        # Track best model weights
        if last_better_eval_loss > val_loss:
            last_better_eval_loss = val_loss
            epochs_since_last_improvement = 0
            best_model_weights = nn.state_dict()  # Save best weights
        else:
            epochs_since_last_improvement += 1

        if epochs_since_last_improvement >= config.notimprovement_tolerance:
            logger.info(
                f"Have not improved in the last {epochs_since_last_improvement} epochs! Exitting"
            )
            break

    # After training, load best weights
    if "best_model_weights" in locals():
        nn.load_state_dict(best_model_weights)
        logger.info("Loaded best model weights from training.")
    mlflow.log_metric("epochs_trained", epoch - epochs_since_last_improvement)


def _run_test_inference(
    nn: torch.nn.Module, dataloader: DataLoader, config: BaseConfig
) -> Tuple[List, List, List]:
    """Run inference on the test set and collect inputs, masks, and predictions."""
    nn.eval()
    inputs = []
    masks = []
    predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluation on test set"):
            input_data, mask = batch
            input_data, mask = input_data.to(config.device), mask.to(config.device)
            prediction = nn(input_data)
            prediction = torch.nn.functional.sigmoid(prediction)

            inputs.append(input_data.cpu().detach().numpy())
            masks.append(mask.cpu().detach().numpy())
            predictions.append(prediction.cpu().detach().numpy())

    return inputs, masks, predictions


def _save_test_outputs(
    inputs: List,
    masks: List,
    predictions: List,
    output_folder: Path,
    logger: logging.Logger,
) -> None:
    """Save test predictions, inputs, and masks."""
    for batch_idx, (input_batch, mask_batch, prediction_batch) in enumerate(
        zip(inputs, masks, predictions)
    ):
        for i, (input_img, mask, prediction) in enumerate(
            zip(input_batch, mask_batch, prediction_batch)
        ):
            prediction_path = (
                output_folder
                / f"test_batch_{batch_idx + 1}_sample_{i + 1}_prediction.tiff"
            )
            input_path = (
                output_folder / f"test_batch_{batch_idx + 1}_sample_{i + 1}_input.tiff"
            )
            mask_path = (
                output_folder / f"test_batch_{batch_idx + 1}_sample_{i + 1}_mask.tiff"
            )

            # Save prediction
            tifffile.imwrite(prediction_path, prediction[0, ...].astype(np.float32))
            logger.debug(
                f"Saved test prediction for batch {batch_idx + 1}, sample {i + 1} at {prediction_path}"
            )

            # Save input image
            tifffile.imwrite(input_path, input_img[0, ...].astype(np.float32))
            logger.debug(
                f"Saved test input image for batch {batch_idx + 1}, sample {i + 1} at {input_path}"
            )

            # Save mask
            tifffile.imwrite(mask_path, mask[0, ...].astype(np.float32))
            logger.debug(
                f"Saved test mask for batch {batch_idx + 1}, sample {i + 1} at {mask_path}"
            )


def _calculate_test_metrics(
    inputs: List, masks: List, predictions: List, config: BaseConfig, epoch: int
) -> Dict:
    """Calculate and log test metrics."""
    # Flatten arrays for metric calculation
    flat_masks = np.concatenate(masks, axis=0).flatten()
    flat_predictions = np.concatenate(predictions, axis=0).flatten()

    # Generate PR curve
    pr_curve = skmetrics.PrecisionRecallDisplay.from_predictions(
        y_true=flat_masks,
        y_pred=flat_predictions,
    )
    mlflow.log_figure(pr_curve.figure_, "pr-curve.png")

    # Calculate metrics
    binary_predictions = (flat_predictions > 0.5).astype(bool)
    binary_masks = flat_masks == 1.0

    # Custom metrics
    TP, FP, FN, TN = tntmetrics.calculate_batch_stats(
        binary_predictions.astype(np.uint8) * 255,
        binary_masks.astype(np.uint8) * 255,
        0,
        255,
    )

    jaccard = tntmetrics.jaccard_index(TP, FP, FN)
    dice = tntmetrics.dice_coefficient(TP, FP, FN)
    tversky = tntmetrics.tversky_index(
        TP, FP, FN, config.eval_tversky_alpha, config.eval_tversky_beta
    )
    focal_tversky = tntmetrics.focal_tversky_loss(
        TP,
        FP,
        FN,
        config.eval_tversky_alpha,
        config.eval_tversky_beta,
        config.eval_tversky_gamma,
    )
    accuracy = tntmetrics.accuracy(TP, FP, FN, TN)
    precision = tntmetrics.precision(TP, FP)
    recall = tntmetrics.recall(TP, FN)

    metrics = {
        "test/jaccard": jaccard,
        "test/dice": dice,
        "test/tversky": tversky,
        "test/focal_tversky": focal_tversky,
        "test/accuracy": accuracy,
        "test/precision": precision,
        "test/recall": recall,
    }

    mlflow.log_metrics(metrics, step=epoch)
    return metrics


def _test(
    nn: torch.nn.Module,
    test_dataloader: DataLoader,
    config: BaseConfig,
    output_folder: Path,
    logger: logging.Logger,
    epoch: int,
) -> Dict:
    """Run complete testing process."""
    logger.info("Starting test evaluation")

    # Standard test set evaluation
    inputs, masks, predictions = _run_test_inference(nn, test_dataloader, config)
    _save_test_outputs(inputs, masks, predictions, output_folder, logger)
    metrics = _calculate_test_metrics(inputs, masks, predictions, config, epoch)

    logger.info(
        f"Test set metrics: Dice={metrics['test/dice']:.4f}, Jaccard={metrics['test/jaccard']:.4f}"
    )
    return metrics


def main(
    input_folder: Path,
    output_folder: Path,
    logger: logging.Logger,
    config: BaseConfig,
    mlflow_address: str = "localhost",
    mlflow_port: str = "800",
) -> None:
    set_all_seeds(config.seed)

    output_folder_path = Path(output_folder)
    output_folder_path.mkdir(exist_ok=True, parents=True)

    # Configure output logging to file
    fh = logging.FileHandler(str(output_folder_path / "training.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    # Prepare mlflow
    logger.info(f"Using MLFlow serve at {mlflow_address}:{mlflow_port}")
    logger.info("Trying to create mlflow tracking")
    mlflow.set_tracking_uri(uri=f"http://{mlflow_address}:{mlflow_port}")

    # Prepare dataloaders
    train_dataloader, test_dataloader, valid_dataloader = _prepare_datasets(
        input_folder, config.seed, config
    )
    visualize_transform_effects(
        train_dataloader,
        num_samples=5,
        output_folder=output_folder_path / "transform_check",
    )

    # Create net
    nn = create_neural_network(config, 1, 1).to(config.device)
    optimizer = torch.optim.AdamW(
        nn.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    criterion = create_loss_criterion(config)

    # MLFlow
    with mlflow.start_run() as run:
        mlflow.log_params(config.__dict__)
        mlflow.log_param("model_signature", nn.get_signature())

        # Run training
        _train(
            nn,
            optimizer,
            criterion,
            train_dataloader,
            valid_dataloader,
            config,
            output_folder,
        )

        # Run testing
        test_metrics = _test(nn, test_dataloader, config, output_folder, logger, config.epochs - 1)

        # After training, log the final model
        with torch.no_grad():
            nn.eval()

            # Also save the model
            checkpoint_path = output_folder / "model_final.pth"
            torch.save(nn.state_dict(), checkpoint_path)
            logger.info(f"Model checkpoint saved at {checkpoint_path}")

            # NOTE: this can be enabled to also log the model into mlfow, but it takes a long time and
            #       it is not necesary gfor our purposes.
            # sample_input = torch.randn(config.batch_size, 1, *config.crop_size).to(config.device)
            # sample_output = nn(sample_input)
            # print("saving the model")
            # # Log the model as a simple artifact - much faster
            # # mlflow.log_artifact(str(checkpoint_path), "model")
            # print("done")

        # Dump the config into .json file
        with open(output_folder / "config.json", "w") as jsonfile:
            config_dict = asdict(config)
            config_dict["mlflow_run_name"] = run.info.run_name
            config_dict["mlflow_run_id"] = run.info.run_id
            config_dict["model_signature"] = nn.get_signature() 
            config_dict["test_dice"] = test_metrics['test/dice']
            config_dict["test_jaccard"] = test_metrics['test/jaccard']
            json.dump(config_dict, jsonfile, indent=2, default=str)


def parse_tuple_arg(arg_string: str, arg_name: str) -> tuple:
    """Parse comma-separated string into tuple of integers."""
    try:
        values = [int(x.strip()) for x in arg_string.split(",")]
        if len(values) != 3:
            raise ValueError(f"{arg_name} must have exactly 3 values")
        return tuple(values)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid {arg_name}: {arg_string}. {str(e)}")


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Training script for neural network.")
    parser.add_argument(
        "input_folder",
        type=Path,
        help="Path to the input folder containing the dataset.",
    )
    parser.add_argument(
        "output_folder",
        type=Path,
        help="Path to the output folder for saving models and logs.",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs."
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training."
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for data splitting."
    )
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the dataset.")
    parser.add_argument(
        "--mlflow_address",
        type=str,
        default="127.0.0.1",
        help="IP address of the MLFlow server",
    )
    parser.add_argument(
        "--mlflow_port", type=str, default="8000", help="Port of the MLFlow server"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "anisotropicunet",
            "anisotropicunet_se",
            "anisotropicunet_csam",
            "anisotropicunet_usenet",
            "basicunet",
        ],
        default="anisotropicunet",
        help="Model architecture to use",
    )
    parser.add_argument(
        "--model_depth",
        type=int,
        default=3,
        help="Depth of the UNet model (number of down/up sampling blocks)",
    )
    parser.add_argument(
        "--base_channels",
        type=int,
        default=64,
        help="Number of channels in the first layer",
    )
    parser.add_argument(
        "--channel_growth",
        type=int,
        default=2,
        help="Factor to multiply channels by at each depth",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay (L2 penalty) for the optimizer. Default: 0.0",
    )
    parser.add_argument(
        "--horizontal_kernel",
        type=str,
        default="1,3,3",
        help="Horizontal kernel size as comma-separated values (depth,height,width). Default: '1,3,3'",
    )
    parser.add_argument(
        "--horizontal_padding",
        type=str,
        default="0,1,1",
        help="Horizontal padding as comma-separated values (depth,height,width). Default: '0,1,1'",
    )
    parser.add_argument(
        "--downscale_kernel",
        type=str,
        default="1,2,2",
        help="Downscale kernel size as comma-separated values (depth,height,width). Default: '1,2,2'",
    )
    parser.add_argument(
        "--downscale_stride",
        type=str,
        default="1,1,1",
        help="Downscale stride as comma-separated values (depth,height,width). Default: '1,1,1'",
    )
    parser.add_argument(
        "--upscale_kernel",
        type=str,
        default="1,2,2",
        help="Upscale kernel size as comma-separated values (depth,height,width). Default: '1,2,2'",
    )
    parser.add_argument(
        "--upscale_stride",
        type=str,
        default="1,2,2",
        help="Upscale padding as comma-separated values (depth,height,width). Default: '1,2,2'",
    )
    parser.add_argument(
        "--reduction_factor",
        type=int,
        default=16,
        help="Reduction factor for SE blocks (only used with anisotropicunet_se). Default: 16",
    )

    args = parser.parse_args()

    # Parse the tuple arguments
    horizontal_kernel = parse_tuple_arg(args.horizontal_kernel, "horizontal_kernel")
    horizontal_padding = parse_tuple_arg(args.horizontal_padding, "horizontal_padding")
    downscale_kernel = parse_tuple_arg(args.downscale_kernel, "downscale_kernel")
    downscale_stride = parse_tuple_arg(args.downscale_stride, "downscale_stride")
    upscale_kernel = parse_tuple_arg(args.upscale_kernel, "upscale_kernel")
    upscale_stride = parse_tuple_arg(args.upscale_stride, "upscale_stride")

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Log the arguments
    logger.info(f"Arguments: {args}")

    # Map string model types to enum
    model_type_map = {
        "anisotropicunet": ModelType.AnisotropicUNet,
        "anisotropicunet_se": ModelType.AnisotropicUNetSE,
        "anisotropicunet_csam": ModelType.AnisotropicUNetCSAM,
        "anisotropicunet_usenet": ModelType.AnisotropicUNetUSENet,
        "basicunet": ModelType.UNet3D,
    }
    if args.model not in model_type_map:
        raise ValueError(
            f"Unknown model type provided. Legal values are: {model_type_map.keys()}"
        )
    model_type = model_type_map[args.model]
    logging.info(f"Got {model_type} model")

    # TODO: code duplication
    # Create appropriate config based on model type
    if (
        model_type == ModelType.AnisotropicUNetSE
        or model_type == ModelType.AnisotropicUNetUSENet
    ):
        config = AnisotropicUNetSEConfig(
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
            num_workers=args.num_workers,
            shuffle=args.shuffle,
            input_folder=str(args.input_folder),
            model_type=model_type,
            model_depth=args.model_depth,
            base_channels=args.base_channels,
            channel_growth=args.channel_growth,
            horizontal_kernel=horizontal_kernel,
            horizontal_padding=horizontal_padding,
            upscale_kernel=upscale_kernel,
            upscale_stride=upscale_stride,
            downscale_kernel=downscale_kernel,
            downscale_stride=downscale_stride,
            seed=args.seed,
            weight_decay=args.weight_decay,
            reduction_factor=args.reduction_factor,
        )
    elif (
        model_type == ModelType.AnisotropicUNet
        or model_type == ModelType.AnisotropicUNetCSAM
    ):
        config = AnisotropicUNetConfig(
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
            num_workers=args.num_workers,
            shuffle=args.shuffle,
            input_folder=str(args.input_folder),
            model_type=model_type,
            model_depth=args.model_depth,
            base_channels=args.base_channels,
            channel_growth=args.channel_growth,
            horizontal_kernel=horizontal_kernel,
            horizontal_padding=horizontal_padding,
            upscale_kernel=upscale_kernel,
            upscale_stride=upscale_stride,
            downscale_kernel=downscale_kernel,
            downscale_stride=downscale_stride,
            seed=args.seed,
            weight_decay=args.weight_decay,
        )
    elif model_type == ModelType.UNet3D:
        config = BaseConfig(
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
            num_workers=args.num_workers,
            shuffle=args.shuffle,
            input_folder=str(args.input_folder),
            model_type=model_type_map[args.model],
            seed=args.seed,
            weight_decay=args.weight_decay,
        )
    else:
        # NOTE: this should never happen
        assert "Unknown model type"

    print(f"Shuffle: {config.shuffle}")

    # Run training
    main(
        args.input_folder,
        args.output_folder,
        logger,
        config,
        args.mlflow_address,
        args.mlflow_port,
    )
