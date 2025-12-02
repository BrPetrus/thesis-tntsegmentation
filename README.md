# TNTSeg - 3D Tunnel Segmentation for Microscopy Images

A deep learning pipeline for segmenting tunnel-like structures in 3D microscopy images using anisotropic U-Net architectures.

## Table of Contents

- [Quick Start](#quick-start)
- [Setup](#setup)
- [Usage](#usage)
  - [Inference (Quick Start)](#inference-quick-start)
  - [Data Preparation](#data-preparation)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Reproducibility](#reproducibility)
- [Project Structure](#project-structure)

---

## Quick Start

### Running Inference on New Images

The fastest way to use a trained model:

```bash
# Setup environment (first time only)
uv sync --extra torch-cpu  # or torch-gpu for CUDA support

# Run inference
uv run python scripts/inference.py \
    --model path/to/model.pth \
    --image path/to/image.tif \
    --output results/ \
    --crop-size 16 256 256 \
    --tile-overlap 10
```

This will generate:
- `results/probability.tif` - Probability map (0-1 range)
- `results/binary.tif` - Binary segmentation (threshold at 0.5)

See [Inference Guide](#inference-quick-start) for details.

---

## Setup

### Environment Setup with `uv`

This project officially uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python environment management.

#### 1. Install `uv`

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via pip
pip install uv
```

#### 2. Clone the Repository

```bash
git clone <repository-url>
cd DP-WIP
```

#### 3. Install Dependencies

Choose the appropriate torch backend:

```bash
# For CPU-only
uv sync --extra torch-cpu

# For GPU (CUDA 12.4)
uv sync --extra torch-gpu

# For development with Jupyter
uv sync --extra torch-cpu --extra jupyter
```

This will:
- Create a virtual environment in `.venv/`
- Install all dependencies from `pyproject.toml`
- Install the `tntseg` package in editable mode

#### 4. Activate the Environment

```bash
# Linux/macOS
source .venv/bin/activate

# Or use uv run to run commands without activation
uv run python scripts/inference.py ...
```

---

## Usage

### Inference (Quick Start)

Run trained models on new images without ground truth evaluation.

#### Basic Usage

```bash
uv run python scripts/inference.py \
    --model checkpoints/model.pth \
    --image data/volume.tif \
    --output results/ \
    --crop-size 16 256 256 \
    --tile-overlap 10
```

#### Key Parameters

- `--model` - Path to trained model checkpoint (`.pth` file)
- `--image` - Path to input 3D TIFF image
- `--output` - Output directory for results
- `--crop-size` - Tile size (Z Y X), e.g., `16 256 256`
- `--tile-overlap` - Overlap between tiles in pixels (reduces boundary artifacts)
- `--batch-size` - Batch size for inference (default: 4)
- `--device` - `cuda` or `cpu` (auto-detected by default)

#### With Post-Processing

```bash
uv run python scripts/inference.py \
    --model checkpoints/model.pth \
    --image data/volume.tif \
    --output results/ \
    --crop-size 16 256 256 \
    --tile-overlap 10 \
    --apply-postprocessing \
    --prediction-threshold 0.5 \
    --minimum-size 100
```

Post-processing options:
- `--apply-postprocessing` - Enable morphological post-processing
- `--prediction-threshold` - Threshold for binarization (default: 0.5)
- `--minimum-size` - Remove objects smaller than this (in pixels)

#### Output Files

Without ground truth:
- `probability.tif` - Continuous probability map (float32, 0-1 range)
- `binary.tif` - Binary segmentation (uint8, 0 or 255)
- `processed_prediction.tif` - Post-processed labeled instances (if enabled)

---

### Data Preparation

Split your dataset into train/test quadrants for cross-validation.

#### Basic Usage

```bash
uv run python -m tntseg.utilities.dataset.split_data \
    data/raw/ \
    data/processed/ \
    --train_quad 1
```

#### Parameters

- `input_folder` - Folder containing `IMG/*.tif` and `GT_MERGED_LABELS/*.tif`
- `output_folder` - Output folder for split data
- `--train_quad` - Quadrant for training (1-4):
  - 1 = top-right
  - 2 = top-left
  - 3 = bottom-left
  - 4 = bottom-right
- `--min_size` - Minimum patch size in Z Y X (default: `7 32 32`)
- `--random_crops_train` - Additional random crops for training (default: 0)
- `--random_crops_test` - Additional random crops for testing (default: 0)
- `--overlap_threshold` - Fraction threshold for tunnel inclusion (default: 0.2)

#### Output Structure

```
data/processed/
├── train/
│   ├── IMG/
│   └── GT_MERGED_LABELS/
└── test/
    ├── IMG/
    └── GT_MERGED_LABELS/
```

---

### Training

#### Using the Training Script

Train a model with customizable architecture and hyperparameters:

```bash
uv run python scripts/training.py \
    --data data/processed/train/ \
    --output-dir checkpoints/ \
    --model anisotropicunet_csam \
    --epochs 1000 \
    --batch-size 32 \
    --lr 0.0001 \
    --device cuda
```

#### Key Parameters

Model architecture:
- `--model` - Model type: `anisotropicunet`, `anisotropicunet_se`, `anisotropicunet_csam`, `anisotropicunet_usenet`, `unet3d`
- `--model-depth` - Number of encoder/decoder levels (default: 3)
- `--base-channels` - Starting number of channels (default: 32)

Training:
- `--epochs` - Number of training epochs
- `--batch-size` - Batch size per device
- `--lr` - Learning rate (default: 0.0001)
- `--weight-decay` - L2 regularization (default: 0.0001)
- `--seed` - Random seed for reproducibility

#### Using MLflow for Experiment Tracking

Start MLflow server:

```bash
# Start MLflow UI (runs in background)
uv run mlflow ui --port 8800 &

# Open in browser
xdg-open http://localhost:8800
```

Training automatically logs to MLflow:
- Hyperparameters
- Training/validation metrics
- Model checkpoints
- Configuration files

---

### Evaluation

Evaluate trained models on test data with ground truth.

#### Basic Evaluation

```bash
uv run python scripts/evaluate_models.py \
    --model-dir checkpoints/2025-12-01_10-00-00/quad1_model/ \
    --data data/processed/test/ \
    --output results/ \
    --tile-overlap 10
```

#### With Post-Processing Analysis

```bash
uv run python scripts/evaluate_models.py \
    --model-dir checkpoints/2025-12-01_10-00-00/quad1_model/ \
    --data data/processed/test/ \
    --output results/ \
    --tile-overlap 10 \
    --run-postprocessing \
    --prediction-threshold 0.5 \
    --recall-threshold 0.6 \
    --minimum-size 100
```

#### Output Files

Evaluation generates:
- `evaluation_metrics.csv` - Dice, Jaccard, precision, recall, etc.
- `visualised_tiling_lines_XXpx.bmp` - Visualization of tiling strategy
- `postprocess_volume_X/` - Post-processing analysis per volume:
  - Matched/unmatched tunnel visualizations
  - Confusion matrices
  - Instance-level metrics

---

## Reproducibility

### Full Training Pipeline

Use the provided shell script to train on all quadrants with consistent settings:

```bash
cd scripts/
bash run.sh
```

Edit `run.sh` to configure:
- Input data paths (`INPUT_ROOT`, `EVAL_ROOT`)
- Model architecture and hyperparameters
- Training settings (epochs, batch size, learning rate)
- Post-processing parameters

The script will:
1. Train a model for each quadrant (quad1-4)
2. Evaluate each model on the corresponding test set
3. Save all results in timestamped directories
4. Log experiments to MLflow

### Overlap Study

Evaluate models with different tile overlaps:

```bash
cd scripts/
bash run_overlap_study.sh path/to/trained/models/
```

This will test overlaps: 0, 10, 20, 30, 40 pixels and generate comparative metrics.

---

## Project Structure

```
DP-WIP/
├── scripts/                      # Main scripts
│   ├── inference.py             # Run inference on new images
│   ├── training.py              # Train models
│   ├── evaluate_models.py       # Evaluate with ground truth
│   ├── postprocess.py           # Post-processing utilities
│   ├── tilingutilities.py       # Tiling and stitching
│   ├── run.sh                   # Full pipeline automation
│   └── run_overlap_study.sh     # Overlap analysis
│
├── tntseg/                       # Main package
│   ├── nn/                      # Neural network models
│   │   └── models/              # U-Net architectures
│   └── utilities/               # Utility modules
│       └── dataset/             # Dataset handling
│           └── split_data.py    # Data splitting script
│
├── pyproject.toml               # Project dependencies
├── uv.lock                      # Locked dependencies
└── README.md                    # This file
```

---

## Common Workflows

### 1. Train and Evaluate a New Model

```bash
# 1. Split data
uv run python -m tntseg.utilities.dataset.split_data \
    data/raw/ data/processed/ --train_quad 1

# 2. Start MLflow (optional)
uv run mlflow ui --port 8800 &

# 3. Train
uv run python scripts/training.py \
    --data data/processed/train/ \
    --output-dir checkpoints/ \
    --epochs 1000

# 4. Evaluate
uv run python scripts/evaluate_models.py \
    --model-dir checkpoints/latest/quad1_model/ \
    --data data/processed/test/ \
    --output results/
```

### 2. Use Pretrained Model for Inference

```bash
# Just run inference - no ground truth needed
uv run python scripts/inference.py \
    --model pretrained/model.pth \
    --image new_data/volume.tif \
    --output predictions/ \
    --crop-size 16 256 256
```

### 3. Reproduce Thesis Results

```bash
# Run complete pipeline with all quadrants
cd scripts/
bash run.sh

# Then analyze overlap impact
bash run_overlap_study.sh output/output-train-all-quads/
```

---

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or crop size:
```bash
--batch-size 16 --crop-size 8 128 128
```

### Dependencies Not Found

Resync environment:
```bash
uv sync --reinstall
```

### MLflow Connection Issues

Check server is running:
```bash
lsof -i :8800
```

Restart if needed:
```bash
pkill -f "mlflow"
uv run mlflow ui --port 8800 &
```

---

## Citation

If you use this code in your research, please cite:

```
[Your thesis citation]
```

---

## License

See [LICENSE](LICENSE) file for details.
