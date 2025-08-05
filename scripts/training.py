from turtle import forward
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
    dataset_std: float = 0.07579
    dataset_mean: float = 0.05988
    test_size: float = 1/3
    notimprovement_tolerance: int = 20
    eval_tversky_alpha: float = 0.8
    eval_tversky_beta: float = 0.2
    eval_tversky_gamma: float = 2
    use_cross_entropy: bool = True
    cross_entropy_loss_weight: float = 0.5
    ce_use_weights: bool = True
    ce_weights: Tuple[float, float] = (1./3602171, 1./67845) 
    use_dice_loss: bool = True
    dice_loss_weight: float = 0.5
    use_focal_tversky_loss: bool = False
    focal_tversky_loss_weight: float = 1.0
    train_focal_tversky_alpha: float = 0.8
    train_focal_tversky_beta: float = 0.2
    train_focal_tversky_gamma: float = 2

def create_loss_criterion(config: Config) -> nn.Module:
    loss_functions = []
    weights = []

    if config.use_cross_entropy:
        if config.ce_use_weights:
            # Use weights
            loss_functions.append(nn.BCEWithLogitsLoss(weight=torch.tensor(config.ce_weights)))
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
        return total_loss


def main(input_folder: Path, mask_folder: Path, output_folder: Path, logger: logging.Logger, seed: int, config: Config) -> None:
    # Load metadata
    df = load_dataset_metadata(input_folder, mask_folder)

    # Train/Test Split
    train_x, test_x = train_test_split(df, test_size=1/3., random_state=seed)
    train_x, valid_x = train_test_split(train_x, test_size=1/5, random_state=seed)

    logger.info(f"Train {len(train_x)} Test {len(test_x)} Validation {len(valid_x)}")

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
        A.RandomCrop3D(size=(7,64, 64)),
        A.ToTensor3D()
    ])
    transform_test = A.Compose([
        A.Normalize(
            mean=config.dataset_mean,
            std=config.dataset_std,
            max_pixel_value=1.0,
            p=1.0
        ),
        A.CenterCrop3D(size=(7,64,64)),  # TODO: is this okay?
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

    # Create net
    nn = UNet3d(1, 1).to(config.device)

    # Training loop
    optimizer = torch.optim.Adam(nn.parameters(), lr=config.lr)
    criterion = create_loss_criterion(config)

    # MLFlow
    with mlflow.start_run():
        mlflow.log_params(config.__dict__)

        # Last time that the eval loss improved
        epochs_since_last_improvement = 0
        last_better_eval_loss = np.inf

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
                
            # Run evaluation on validation set
            with torch.no_grad():
                nn.eval()
                val_loss = 0.0
                total = 0
                TP = 0
                FP = 0
                FN = 0
                TN = 0
                for batch_idx, batch in enumerate(tqdm(valid_dataloader, desc=f"Validation")):
                    inputs, masks = batch
                    inputs, masks = inputs.to(config.device), masks.to(config.device)

                    outputs = nn(inputs)
                    loss = criterion(outputs, masks)
                    val_loss += loss

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
                            prediction_path = output_folder / f"epoch_val_{epoch+1}_batch_{batch_idx+1}_sample_{i+1}_prediction.tiff"
                            input_path = output_folder / f"epoch_val_{epoch+1}_batch_{batch_idx+1}_sample_{i+1}_input.tiff"
                            mask_path = output_folder / f"epoch_val_{epoch+1}_batch_{batch_idx+1}_sample_{i+1}_mask.tiff"
                            # Save prediction
                            tifffile.imwrite(prediction_path, prediction[0, ...].astype(np.float32))
                            logger.debug(f"Saved prediction for epoch {epoch+1}, batch {batch_idx+1}, sample {i+1} at {prediction_path}")
                            
                            # Save input image
                            tifffile.imwrite(input_path, input_img[0, ...].astype(np.float32))
                            logger.debug(f"Saved input image for epoch {epoch+1}, batch {batch_idx+1}, sample {i+1} at {input_path}")
                            
                            # Save mask
                            tifffile.imwrite(mask_path, mask[0, ...].astype(np.float32))
                            logger.debug(f"Saved mask for epoch {epoch+1}, batch {batch_idx+1}, sample {i+1} at {mask_path}")

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


            if last_better_eval_loss > val_loss:
                last_better_eval_loss = val_loss
                epochs_since_last_improvement = 0
            else:
                epochs_since_last_improvement += 1
            
            if epochs_since_last_improvement >= config.notimprovement_tolerance:
                logger.info(f"Have not improved in the last {epochs_since_last_improvement} epochs! Exitting")
                break

        # Test set
        nn.eval()
        inputs = []
        masks = []
        predictions = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Evaluation on test set")):
                input, mask = batch
                input, mask = input.to(config.device), mask.to(config.device)
                prediction = nn(input)
                prediction = torch.nn.functional.sigmoid(prediction)

                inputs.append(input.cpu().detach().numpy())
                masks.append(mask.cpu().detach().numpy())
                predictions.append(prediction.cpu().detach().numpy())


        # Save predictions, inputs, and masks for the test set
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

            
        # NOTE: last batch might have wrong size
        inputs = np.concat(inputs, axis=0).flatten()
        masks = np.concat(masks, axis=0).flatten()
        predictions = np.concat(predictions, axis=0).flatten()

        # Find PR curve
        pr_curve = skmetrics.PrecisionRecallDisplay.from_predictions(
            y_true=masks,
            y_pred=predictions,
        )
        mlflow.log_figure(pr_curve.figure_, "pr-curve.png")

        # Find Accuracy, Recall, Precision, and F1 Score
        binary_predictions = (predictions > 0.5).astype(np.bool)
        binary_masks = (masks == 1.0)
        sk_accuracy = skmetrics.accuracy_score(binary_masks, binary_predictions)
        sk_f1score = skmetrics.f1_score(binary_masks, binary_predictions)
        sk_recall = skmetrics.recall_score(binary_masks, binary_predictions)
        sk_precision = skmetrics.precision_score(binary_masks, binary_predictions)
        # Calculate additional metrics using test set predictions

        # a =tntmetrics.calc_stats(binary_predictions, binary_masks)
        TP, FP, FN, TN = tntmetrics.calculate_batch_stats(binary_predictions.astype(np.uint8) * 255, binary_masks.astype(np.uint8)*255, 0, 255)
        jaccard = tntmetrics.jaccard_index(TP, FP, FN)
        dice = tntmetrics.dice_coefficient(TP, FP, FN)
        tversky = tntmetrics.tversky_index(TP, FP, FN, config.eval_tversky_alpha, config.eval_tversky_beta)
        focal_tversky = tntmetrics.focal_tversky_loss(TP, FP, FN, config.eval_tversky_alpha, config.eval_tversky_beta, config.eval_tversky_gamma)
        accuracy = tntmetrics.accuracy(TP, FP, FN, TN)
        precision = tntmetrics.precision(TP, FP)
        recall = tntmetrics.recall(TP, FN)
        mlflow.log_metrics({
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
        }, step=epoch)

        # After training, you might log the final model
        mlflow.pytorch.log_model(nn, "model")

        # Also save the model
        checkpoint_path = output_folder / f"model_final.pth"
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
    parser.add_argument("--mlflow_address", type=str, default="127.0.0.1",
                        help="IP address of the MLFlow server")
    parser.add_argument("--mlflow_port", type=str, default="8000",
                        help="Port of the MLFlow server")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=1.0e-3,
                        help="Learning rate for the optimizer.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use for training (e.g., 'cpu', 'cuda').")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of workers for data loading.")
    parser.add_argument("--shuffle", type=bool, default=False,
                        help="Whether to shuffle the dataset during training.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for training.")
    

    # Parse arguments & Setup
    args = parser.parse_args()
    output_folder = Path(args.output_folder)
    input_folder = Path(args.input_folder)
    log_folder = Path(args.log_folder)
    mask_folder = Path(args.mask_folder)
    if not input_folder.exists():
        raise RuntimeError(f"Specified input folder '{input_folder}' does not exist!")
    if not mask_folder.exists():
        raise RuntimeError("Specified mask folder does not exist!")
    output_folder.mkdir(parents=True, exist_ok=True)
    log_folder.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=log_folder / 'training.log', encoding='utf-8', level=logging.DEBUG, format='%(asctime)s %(message)s')
    logger.debug(f"Input folder: {args.input_folder}")
    logger.debug(f"Output folder: {args.output_folder}")
    logger.debug(f"Log folder: {args.log_folder}")
    logger.info(f"Using MLFlow serve at {args.mlflow_address}:{args.mlflow_port}")
    
    logger.info("Trying to create mlflow tracking")
    mlflow.set_tracking_uri(uri=f"http://{args.mlflow_address}:{args.mlflow_port}")

    config = Config(
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        batch_size=args.batch_size,
    )

    # Run training
    main(input_folder, mask_folder, output_folder, logger, args.random_state, config)
    
    
    
