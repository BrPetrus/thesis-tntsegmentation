from turtle import forward
import pandas as pd
import sklearn
from tntseg.utilities.dataset.datasets import TNTDataset, load_dataset_metadata
from tntseg.nn.models.unet3d_basic import UNet3d
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import logging
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import albumentations as A
import tifffile
import mlflow
from sklearn import metrics as skmetrics
import sklearn.metrics as skmetrics
import tntseg.utilities.metrics.metrics as tntmetrics
from typing import List, Tuple
from torch.types import Tensor
import tntseg.utilities.metrics.metrics_torch as tntloss

@dataclass
class Config:
    epochs: int
    lr: float
    batch_size: int
    device: str
    num_workers: int
    shuffle: bool
    device: str
    input_folder: str
    seed: int = 42
    dataset_std: float = 0.07579
    dataset_mean: float = 0.05988
    test_size: float = 1/3
    notimprovement_tolerance: int = 50
    eval_tversky_alpha: float = 0.8
    eval_tversky_beta: float = 0.2
    eval_tversky_gamma: float = 2
    use_cross_entropy: bool = True
    cross_entropy_loss_weight: float = 0.4
    ce_use_weights: bool = True
    ce_pos_weight: float  = ((3602171) / 67845.) # Negative/positive ratio to penalize
    # ce_pos_weight: float  = 67845 / 3602171  # Negative/positive ratio to penalize
    use_dice_loss: bool = True
    dice_loss_weight: float = 0.6
    use_focal_tversky_loss: bool = False
    focal_tversky_loss_weight: float = 1.0
    train_focal_tversky_alpha: float = 0.8
    train_focal_tversky_beta: float = 0.2
    train_focal_tversky_gamma: float = 2
    weight_decay: float = 0.0  # Add this line
    crop_size: Tuple[int, int, int] = (7, 80, 80)

def create_loss_criterion(config: Config) -> nn.Module:
    loss_functions = []
    weights = []

    if config.use_cross_entropy:
        if config.ce_use_weights:
            # Use weights
            # weight = torch.tensor([1, config.ce_pos_weight, 1, 1, 1])
            loss_functions.append(nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config.ce_pos_weight)))
            # loss_functions.append(nn.BCEWithLogitsLoss(pos_weight=weight))
        else:
            loss_functions.append(nn.BCEWithLogitsLoss())
        weights.append(config.cross_entropy_loss_weight)
    if config.use_dice_loss:
        loss_functions.append(tntloss.DiceLoss())
        weights.append(config.dice_loss_weight)
    if config.use_focal_tversky_loss:
        loss_functions.append(
            tntloss.FocalTverskyLoss(
                alpha=config.train_focal_tversky_alpha,
                beta=config.train_focal_tversky_beta,
                gamma=config.train_focal_tversky_gamma
            ))
        weights.append(config.focal_tversky_loss_weight)
    return CombinedLoss(loss_functions, weights)


class CombinedLoss(nn.Module):
    def __init__(self, loss_functions: List[nn.Module], loss_weights: List[float]):
        super().__init__()
        if len(loss_functions) != len(loss_weights):
            raise ValueError("Number of loss functions must match number of weights")
        self.loss_functions = nn.ModuleList(loss_functions)
        self.loss_weights = loss_weights
    def forward(self, pred: Tensor, mask: Tensor) -> Tensor:
        total_loss = 0.0
        for loss_fun, weight in zip(self.loss_functions, self.loss_weights):
            if isinstance(loss_fun, nn.BCEWithLogitsLoss):
                loss = loss_fun(pred, mask)
            else:
                loss = loss_fun(torch.sigmoid(pred), mask)
            total_loss += weight*loss
            logger.debug(f"{type(loss_fun)} - {loss}")
        return total_loss

def _prepare_datasets(input_folder: Path, seed: int, validation_ratio = 1/3.) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Load metadata
    input_folder_train = input_folder / "train"
    if not input_folder_train.exists():
        raise RuntimeError(f"Missing train subfolder at '{input_folder_train}'")
    input_folder_test = input_folder / "test"
    if not input_folder_test.exists():
        raise RuntimeError(f"Missing testing subfolder at '{input_folder_test}'")
    
    df_train = load_dataset_metadata(input_folder_train / "IMG", input_folder_train / "GT_MERGED_LABELS") 
    test_x = load_dataset_metadata(input_folder_test / "IMG", input_folder_test / "GT_MERGED_LABELS")

    # Train/Test Split
    train_x, valid_x = train_test_split(df_train, test_size=validation_ratio, random_state=seed)

    logger.info(f"Train {len(train_x)} Test {len(test_x)} Validation {len(valid_x)}")
    total = len(train_x) + len(test_x) + len(valid_x)
    logger.info(f"Train {len(train_x) / total * 100}% Test {len(test_x)/total*100}% Validation {len(valid_x)/total*100}%")

    # Define transforms
    transforms_train = A.Compose([
        A.Normalize(
            mean=config.dataset_mean,
            std=config.dataset_std,
            max_pixel_value=1.0,
            p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.RandomBrightnessContrast(p=0.5),
        A.Rotate(),
        A.RandomCrop3D(size=config.crop_size),
        A.ToTensor3D()
    ])
    transform_test = A.Compose([
        A.Normalize(
            mean=config.dataset_mean,
            std=config.dataset_std,
            max_pixel_value=1.0,
            p=1.0
        ),
        A.CenterCrop3D(size=config.crop_size),  # TODO: is this okay?
        A.ToTensor3D()
    ])

    # Create datasets
    train_dataset = TNTDataset(train_x, transforms=transforms_train)
    test_dataset = TNTDataset(test_x, transforms=transform_test)
    valid_dataset = TNTDataset(valid_x, transforms=transform_test)

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
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
    )
    return train_dataloader, test_dataloader, valid_dataloader

# TODO: think abou treplacing with batchstats from utlities metrics
def _calculate_metrics(nn: torch.nn.Module, dataloader: DataLoader, criterion: torch.nn.Module, epoch: int, prefix: str, output_folder: Path) -> Tuple[int, int, int, int, float]:
    # Run evaluation on validation set
    with torch.no_grad():
        nn.eval()
        loss = 0.0
        total = 0
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Validation")):
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
            if batch_idx == 0:
                predictions = torch.sigmoid(outputs).cpu().detach().numpy()
                inputs_np = inputs.cpu().detach().numpy()
                masks_np = masks.cpu().detach().numpy()
                
                for i, (prediction, input_img, mask) in enumerate(zip(predictions, inputs_np, masks_np)):
                    prediction_path = output_folder / f"epoch_{prefix}_{epoch+1}_batch_{batch_idx+1}_sample_{i+1}_prediction.tiff"
                    input_path = output_folder / f"epoch_{prefix}_{epoch+1}_batch_{batch_idx+1}_sample_{i+1}_input.tiff"
                    mask_path = output_folder / f"epoch_{prefix}_{epoch+1}_batch_{batch_idx+1}_sample_{i+1}_mask.tiff"
                    # Save prediction
                    tifffile.imwrite(prediction_path, prediction[0, ...].astype(np.float32))
                    logger.debug(f"Saved prediction for epoch {epoch+1}, batch {batch_idx+1}, sample {i+1} at {prediction_path}")
                    
                    # Save input image
                    tifffile.imwrite(input_path, input_img[0, ...].astype(np.float32))
                    logger.debug(f"Saved input image for epoch {epoch+1}, batch {batch_idx+1}, sample {i+1} at {input_path}")
                    
                    # Save mask
                    tifffile.imwrite(mask_path, mask[0, ...].astype(np.float32))
                    logger.debug(f"Saved mask for epoch {epoch+1}, batch {batch_idx+1}, sample {i+1} at {mask_path}")
    return TP, TN, FP, FN, loss


def _train_single_epoch(nn, optimizer, criterion, train_dataloader, config, epoch, output_folder: Path):
    epoch_loss = 0.
    for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")):
        inputs, masks = batch
        inputs, masks = inputs.to(config.device), masks.to(config.device)

        optimizer.zero_grad()
        outputs = nn(inputs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # Save predictions, inputs, and masks for the first batch of each epoch
        if batch_idx == 0:
            predictions = torch.sigmoid(outputs).cpu().detach().numpy()
            inputs_np = inputs.cpu().detach().numpy()
            masks_np = masks.cpu().detach().numpy()
                
            for i, (prediction, input_img, mask) in enumerate(zip(predictions, inputs_np, masks_np)):
                prediction_path = output_folder / f"epoch_{epoch+1}_batch_{batch_idx+1}_sample_{i+1}_prediction.tiff"
                input_path = output_folder / f"epoch_{epoch+1}_batch_{batch_idx+1}_sample_{i+1}_input.tiff"
                mask_path = output_folder / f"epoch_{epoch+1}_batch_{batch_idx+1}_sample_{i+1}_mask.tiff"
                    # Save prediction
                tifffile.imwrite(prediction_path, prediction[0, ...].astype(np.float32))
                logger.debug(f"Saved prediction for epoch {epoch+1}, batch {batch_idx+1}, sample {i+1} at {prediction_path}")
                    
                    # Save input image
                tifffile.imwrite(input_path, input_img[0, ...].astype(np.float32))
                logger.debug(f"Saved input image for epoch {epoch+1}, batch {batch_idx+1}, sample {i+1} at {input_path}")
                    
                    # Save mask
                tifffile.imwrite(mask_path, mask[0, ...].astype(np.float32))
                logger.debug(f"Saved mask for epoch {epoch+1}, batch {batch_idx+1}, sample {i+1} at {mask_path}")
    return epoch_loss

def _train(nn: torch.nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module, train_dataloader: DataLoader, valid_dataloader: DataLoader, config: Config, output_folder: Path) -> None:
    # Last time that the eval loss improved
    epochs_since_last_improvement = 0
    last_better_eval_loss = np.inf

    for epoch in range(config.epochs):
        nn.train()
        epoch_loss = _train_single_epoch(nn, optimizer, criterion, train_dataloader, config, epoch, output_folder)
            
        TP, TN, FP, FN, val_loss = _calculate_metrics(nn, valid_dataloader, criterion, epoch, 'val', output_folder)
        # Calculate metrics
        accuracy = tntmetrics.accuracy(TP, FP, FN, TN)
        precision = tntmetrics.precision(TP, FP)
        recall = tntmetrics.recall(TP, FN)
        jaccard = tntmetrics.jaccard_index(TP, FP, FN)
        dice = tntmetrics.dice_coefficient(TP, FP, FN)
        tversky = tntmetrics.tversky_index(TP, FP, FN, config.eval_tversky_alpha, config.eval_tversky_beta)
        focal_tversky = tntmetrics.focal_tversky_loss(TP, FP, FN, config.eval_tversky_alpha, config.eval_tversky_beta, config.eval_tversky_gamma)

        mlflow.log_metrics({
            "train/loss": epoch_loss,
            "val/loss": val_loss,
            "val/accuracy": accuracy,
            "val/precision": precision, 
            "val/recall": recall,
            "val/jaccard": jaccard,
            "val/dice": dice,
            "val/tversky": tversky,
            "val/focaltversky": focal_tversky
        }, step=epoch)
        logger.info(f"Epoch {epoch+1}/{config.epochs}, Train/Loss: {epoch_loss:.4f}  Val/Loss: {val_loss}")
        logger.info(f"Epoch {epoch+1}/{config.epochs}, Val/Acc: {accuracy}, Val/Prec: {precision}, Val/Recall: {recall}")
        logger.info(f"Last improved {epochs_since_last_improvement} epochs ago.")

        # Track best model weights
        if last_better_eval_loss > val_loss:
            last_better_eval_loss = val_loss
            epochs_since_last_improvement = 0
            best_model_weights = nn.state_dict()  # Save best weights
        else:
            epochs_since_last_improvement += 1
        
        if epochs_since_last_improvement >= config.notimprovement_tolerance:
            logger.info(f"Have not improved in the last {epochs_since_last_improvement} epochs! Exitting")
            break

    # After training, load best weights
    if 'best_model_weights' in locals():
        nn.load_state_dict(best_model_weights)
        logger.info("Loaded best model weights from training.")

def _run_test_inference(nn: torch.nn.Module, dataloader: DataLoader, config: Config) -> Tuple[List, List, List]:
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

def _save_test_outputs(inputs: List, masks: List, predictions: List, output_folder: Path, logger: logging.Logger) -> None:
    """Save test predictions, inputs, and masks."""
    for batch_idx, (input_batch, mask_batch, prediction_batch) in enumerate(zip(inputs, masks, predictions)):
        for i, (input_img, mask, prediction) in enumerate(zip(input_batch, mask_batch, prediction_batch)):
            prediction_path = output_folder / f"test_batch_{batch_idx+1}_sample_{i+1}_prediction.tiff"
            input_path = output_folder / f"test_batch_{batch_idx+1}_sample_{i+1}_input.tiff"
            mask_path = output_folder / f"test_batch_{batch_idx+1}_sample_{i+1}_mask.tiff"
            
            # Save prediction
            tifffile.imwrite(prediction_path, prediction[0, ...].astype(np.float32))
            logger.debug(f"Saved test prediction for batch {batch_idx+1}, sample {i+1} at {prediction_path}")
            
            # Save input image
            tifffile.imwrite(input_path, input_img[0, ...].astype(np.float32))
            logger.debug(f"Saved test input image for batch {batch_idx+1}, sample {i+1} at {input_path}")
            
            # Save mask
            tifffile.imwrite(mask_path, mask[0, ...].astype(np.float32))
            logger.debug(f"Saved test mask for batch {batch_idx+1}, sample {i+1} at {mask_path}")

def _calculate_test_metrics(inputs: List, masks: List, predictions: List, config: Config, epoch: int) -> dict:
    """Calculate and log test metrics."""
    # Flatten arrays for metric calculation
    flat_inputs = np.concatenate(inputs, axis=0).flatten()
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
    binary_masks = (flat_masks == 1.0)
    
    # Scikit-learn metrics
    sk_accuracy = skmetrics.accuracy_score(binary_masks, binary_predictions)
    sk_f1score = skmetrics.f1_score(binary_masks, binary_predictions)
    sk_recall = skmetrics.recall_score(binary_masks, binary_predictions)
    sk_precision = skmetrics.precision_score(binary_masks, binary_predictions)
    
    # Custom metrics
    TP, FP, FN, TN = tntmetrics.calculate_batch_stats(
        binary_predictions.astype(np.uint8) * 255, 
        binary_masks.astype(np.uint8) * 255, 
        0, 255
    )
    
    jaccard = tntmetrics.jaccard_index(TP, FP, FN)
    dice = tntmetrics.dice_coefficient(TP, FP, FN)
    tversky = tntmetrics.tversky_index(TP, FP, FN, config.eval_tversky_alpha, config.eval_tversky_beta)
    focal_tversky = tntmetrics.focal_tversky_loss(
        TP, FP, FN, config.eval_tversky_alpha, config.eval_tversky_beta, config.eval_tversky_gamma
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
        "test/accuracy_skmetric": sk_accuracy,
        "test/f1score_skmetric": sk_f1score,
        "test/recall_skmetric": sk_recall,
        "test/precision_skmetric": sk_precision,
    }
    
    mlflow.log_metrics(metrics, step=epoch)
    return metrics

def _run_test_quadrant(nn: torch.nn.Module, test_x: pd.DataFrame, transform_test: A.Compose, 
                      quad_idx: int, config: Config, output_folder: Path, 
                      logger: logging.Logger, epoch: int) -> dict:
    """Run inference on a specific quadrant of test images."""
    quad_name = ["top-left", "top-right", "bottom-left", "bottom-right"][quad_idx]
    logger.info(f"Processing {quad_name} quadrant (idx: {quad_idx})...")
    
    # Create dataset for this quadrant with tiling
    quad_dataset = TNTDataset(
        dataframe=test_x,
        load_masks=True,  # We need masks for metrics
        transforms=transform_test,
        quad_mode=True,
        quad_idx=quad_idx,
        tile=True,
        tile_size=(7, 64, 64),
        overlap=0
    )
    
    quad_dataloader = DataLoader(
        quad_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    # Run inference
    predictions = []
    metadata = []
    all_masks = []
    
    nn.eval()  # Ensure model is in eval mode
    with torch.no_grad():
        for batch_data in tqdm(quad_dataloader, desc=f"Inference on quad {quad_idx}"):
            inputs, masks, meta = batch_data  # Unpack (volume, mask, metadata)
            inputs = inputs.unsqueeze(1).to(config.device)  # Add channel dimension
            
            outputs = nn(inputs)
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            
            # Store predictions, masks, and metadata for stitching
            for i in range(probs.shape[0]):
                predictions.append(probs[i, 0])  # Remove channel dimension
                all_masks.append(masks[i].cpu().numpy())
                
                # Handle metadata - convert tensors to Python types if needed
                if isinstance(meta, dict):
                    # If meta is a dictionary of lists/tensors
                    meta_item = {k: v[i].item() if isinstance(v[i], torch.Tensor) else v[i] 
                                for k, v in meta.items()}
                else:
                    # If meta is a list of dictionaries
                    meta_item = meta[i]
                
                metadata.append(meta_item)
    
    # Stitch predictions and masks
    logger.info(f"Stitching quad {quad_idx} predictions...")
    quad_output_path = output_folder / f"quad_{quad_idx}"
    quad_output_path.mkdir(exist_ok=True, parents=True)
    
    # Stitch predictions
    stitched_preds = TNTDataset.stitch_predictions(
        predictions, metadata, output_path=str(quad_output_path / "predictions")
    )
    
    # Stitch masks
    stitched_masks = TNTDataset.stitch_predictions(
        all_masks, metadata, output_path=str(quad_output_path / "masks")
    )
    
    # Calculate metrics for this quadrant
    quad_metrics = {}
    for img_idx in stitched_preds:
        if img_idx not in stitched_masks:
            logger.warning(f"No mask found for image {img_idx} in quadrant {quad_idx}")
            continue
        
        pred = stitched_preds[img_idx]
        mask = stitched_masks[img_idx]
        
        # Binarize prediction
        binary_pred = (pred > 0.5).astype(np.uint8)
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        # Save binary prediction
        tifffile.imwrite(
            str(quad_output_path / f"binary_pred_img_{img_idx}.tif"), 
            binary_pred * 255
        )
        
        # Calculate stats
        TP, FP, FN, TN = tntmetrics.calculate_batch_stats(
            binary_pred * 255, binary_mask * 255, 0, 255
        )
        
        # Calculate metrics
        metrics_values = {
            "accuracy": tntmetrics.accuracy(TP, FP, FN, TN),
            "precision": tntmetrics.precision(TP, FP),
            "recall": tntmetrics.recall(TP, FN),
            "jaccard": tntmetrics.jaccard_index(TP, FP, FN),
            "dice": tntmetrics.dice_coefficient(TP, FP, FN),
            "tversky": tntmetrics.tversky_index(TP, FP, FN, config.eval_tversky_alpha, config.eval_tversky_beta),
            "focal_tversky": tntmetrics.focal_tversky_loss(TP, FP, FN, config.eval_tversky_alpha, config.eval_tversky_beta, config.eval_tversky_gamma)
        }
        
        quad_metrics[f"img_{img_idx}"] = metrics_values
        
        # Log metrics to MLflow
        for metric_name, value in metrics_values.items():
            mlflow.log_metric(f"tiled_test/quad_{quad_idx}_img_{img_idx}_{metric_name}", value, step=epoch)
    
    # Log average metrics for this quadrant
    if quad_metrics:
        avg_metrics = {k: 0.0 for k in next(iter(quad_metrics.values()))}
        for metrics_dict in quad_metrics.values():
            for metric_name, value in metrics_dict.items():
                avg_metrics[metric_name] += value
        
        # Average the metrics
        for metric_name in avg_metrics:
            avg_metrics[metric_name] /= len(quad_metrics)
            mlflow.log_metric(f"tiled_test/quad_{quad_idx}_avg_{metric_name}", avg_metrics[metric_name], step=epoch)
        
        logger.info(f"Quadrant {quad_idx} testing completed")
        logger.info(f"Average Dice: {avg_metrics['dice']:.4f}")
        logger.info(f"Average Jaccard: {avg_metrics['jaccard']:.4f}")
        
        return avg_metrics
    else:
        logger.warning(f"No metrics calculated for quadrant {quad_idx}")
        return {}

def _test(nn: torch.nn.Module, test_dataloader: DataLoader, config: Config, 
         output_folder: Path, logger: logging.Logger, epoch: int, test_x: pd.DataFrame,
         transform_test: A.Compose, quad_idx: int = None) -> None:
    """Run complete testing process."""
    logger.info("Starting test evaluation")
    
    # Standard test set evaluation
    inputs, masks, predictions = _run_test_inference(nn, test_dataloader, config)
    _save_test_outputs(inputs, masks, predictions, output_folder, logger)
    metrics = _calculate_test_metrics(inputs, masks, predictions, config, epoch)
    
    logger.info(f"Test set metrics: Dice={metrics['test/dice']:.4f}, Jaccard={metrics['test/jaccard']:.4f}")
    
    # Quadrant-based evaluation if specified
    if quad_idx is not None:
        quad_metrics = _run_test_quadrant(nn, test_x, transform_test, quad_idx, config, output_folder, logger, epoch)
        if quad_metrics:
            logger.info(f"Quadrant {quad_idx} evaluation complete: "
                       f"Dice={quad_metrics['dice']:.4f}, Jaccard={quad_metrics['jaccard']:.4f}")

def main(input_folder: Path, output_folder: Path, logger: logging.Logger, config: Config, quad_idx: int = None, mlflow_address: str = "localhost", mlflow_port: str = "800") -> None:
    output_folder_path = Path(output_folder)
    output_folder_path.mkdir(exist_ok=True, parents=True)

    # Configure output logging to file
    fh = logging.FileHandler(str(output_folder_path / 'training.log'))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    # Prepare mlflow
    logger.info(f"Using MLFlow serve at {args.mlflow_address}:{args.mlflow_port}")
    logger.info("Trying to create mlflow tracking")
    mlflow.set_tracking_uri(uri=f"http://{args.mlflow_address}:{args.mlflow_port}")

    # Prepare dataloaders
    train_dataloader, test_dataloader, valid_dataloader = _prepare_datasets(input_folder, config.seed) 
    
    # Get test_x and transform_test for quadrant testing
    input_folder_test = input_folder / "test"
    test_x = load_dataset_metadata(input_folder_test / "IMG", input_folder_test / "GT_MERGED_LABELS")
    transform_test = A.Compose([
        A.Normalize(
            mean=config.dataset_mean,
            std=config.dataset_std,
            max_pixel_value=1.0,
            p=1.0
        ),
        A.CenterCrop3D(size=config.crop_size),
        A.ToTensor3D()
    ])

    # Create net
    nn = UNet3d(1, 1).to(config.device)
    optimizer = torch.optim.Adam(nn.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = create_loss_criterion(config)

    # MLFlow
    with mlflow.start_run():
        mlflow.log_params(config.__dict__)

        # Run training
        _train(nn, optimizer, criterion, train_dataloader, valid_dataloader, config, output_folder)

        # Run testing
        _test(nn, test_dataloader, config, output_folder, logger, config.epochs-1, test_x, transform_test, quad_idx)

        # After training, log the final model
        mlflow.pytorch.log_model(nn, "model")

        # Also save the model
        checkpoint_path = output_folder / f"model_final.pth"
        torch.save(nn.state_dict(), checkpoint_path)
        logger.info(f"Model checkpoint saved at {checkpoint_path}")

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Training script for neural network.")
    parser.add_argument("input_folder", type=Path, help="Path to the input folder containing the dataset.")
    parser.add_argument("output_folder", type=Path, help="Path to the output folder for saving models and logs.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data splitting.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the dataset.")
    parser.add_argument("--mlflow_address", type=str, default="127.0.0.1",
                        help="IP address of the MLFlow server")
    parser.add_argument("--mlflow_port", type=str, default="8000",
                        help="Port of the MLFlow server")
    parser.add_argument("--quad_idx", type=int, choices=[0, 1, 2, 3], default=None,
                      help="Quadrant index for evaluation (0: top-left, 1: top-right, 2: bottom-left, 3: bottom-right)")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                      help="Weight decay (L2 penalty) for the optimizer. Default: 0.0")

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Log the arguments
    logger.info(f"Arguments: {args}")

    # Config
    config = Config(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        input_folder=str(args.input_folder),
        seed=args.seed,
        weight_decay=args.weight_decay,
    )

    # Run training
    main(args.input_folder, args.output_folder, logger, config, args.quad_idx, args.mlflow_address, args.mlflow_port)



