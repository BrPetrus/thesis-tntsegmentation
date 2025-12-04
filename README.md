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
- [License](#license)

---

## Quick Start

### Running Inference on New Images

The fastest way to use a pre-trained model (included in the archive):

```bash
# Setup environment (first time only)
uv sync --extra torch-cpu  # or torch-gpu for CUDA support

# Run inference with included model (using CSAM 3D architecture, trained on quad 1)
uv run python scripts/inference.py \
    models/anisotropic_csam_3d/quad1_model/model_final.pth \
    path/to/your/image.tif \
    results/ \
    --tile_overlap 10 \
    --postprocess
```

**Tip:** Multiple architectures are available in both 2D and 3D versions:
- `anisotropic_csam_2d` / `anisotropic_csam_3d` - With CSAM attention
- `anisotropic_usenet_2d` / `anisotropic_usenet_3d` - With USE-Net modules
- `anisotropic_basic_2d` / `anisotropic_basic_3d` - Basic anisotropic
- `unet3d` - Standard 3D U-Net baseline

Each architecture has models trained on all 4 quadrants.

**Note:** The model's `config.json` file must be in the same directory as the `.pth` file. This file contains essential information about the model architecture and training parameters.

This will generate:
- `results/probability.tif` - Probability map (0-1 range)
- `results/binary.tif` - Binary segmentation (threshold at 0.5)
- `results/processed_prediction.tif` - Post-processed result (if `--postprocess` enabled)

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

#### 2. Extract and Navigate to the Archive

```bash
# Extract the archive (if compressed)
tar -xzf thesis-tntsegmentation.tar.gz  # or unzip thesis-tntsegmentation.zip

# Navigate to the project directory
cd thesis-tntsegmentation
```

The archive includes:
- `models/` - Pre-trained model checkpoints for all architectures (CSAM 2D/3D, USE-Net 2D/3D, Basic 2D/3D, 3D U-Net) trained on all quadrants
- `data/` - Dataset and evaluation data
- Source code and scripts

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
    checkpoints/model.pth \
    data/volume.tif \
    results/ \
    --tile_overlap 10
```

**Important:** The model directory must contain both `model.pth` and `config.json` files. The config file is automatically created during training and contains the model architecture and hyperparameters.

#### Key Parameters

- `model_path` - Path to trained model checkpoint (`.pth` file)
- `image_path` - Path to input 3D TIFF image
- `output_dir` - Output directory for results
- `--tile_overlap` - Overlap between tiles in pixels (reduces boundary artifacts, default: 0)
- `--batch_size` - Batch size for inference (default: 8)
- `--device` - `cuda` or `cpu` (auto-detected by default)
- `--no_probability` - Skip saving probability map
- `--no_binary` - Skip saving binary prediction

#### With Post-Processing

```bash
uv run python scripts/inference.py \
    checkpoints/model.pth \
    data/volume.tif \
    results/ \
    --tile_overlap 10 \
    --postprocess \
    --prediction_threshold 0.5 \
    --minimum_size 100
```

Post-processing options:
- `--postprocess` - Enable morphological post-processing
- `--prediction_threshold` - Threshold for binarization (default: 0.5)
- `--minimum_size` - Remove objects smaller than this in pixels (default: 100)

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
- `--train_quad` - Quadrant to HOLD OUT for testing (1-4). Training patches come from OTHER 3 quadrants:
  - 1 = top-left (trains on top-right, bottom-left, bottom-right)
  - 2 = top-right (trains on top-left, bottom-left, bottom-right)
  - 3 = bottom-left (trains on top-left, top-right, bottom-right)
  - 4 = bottom-right (trains on top-left, top-right, bottom-left)
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
    --recall-threshold 0.5 \
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

Many of the thesis results were generated using automated scripts that systematically train and evaluate models across all quadrants. These scripts call `training.py` and `evaluate_models.py` with different parameters in an automated fashion.

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
1. Train a model for each quadrant (quad1-4) using `training.py`
2. Evaluate each model on the corresponding test set using `evaluate_models.py`
3. Save all results in timestamped directories
4. Log experiments to MLflow

**Note:** This script essentially automates the manual training and evaluation workflow by calling the same underlying scripts with different parameters. See `run.sh` for full implementation details.

### Overlap Study

The overlap study was used to systematically evaluate the impact of different tile overlap values on segmentation quality:

```bash
cd scripts/
bash run_overlap_study.sh path/to/trained/models/
```

This script tests overlaps of 0, 10, 20, 30, 40 pixels by repeatedly calling `evaluate_models.py` with different `--tile_overlap` values, generating comparative metrics for analysis.

**Note:** For full details on the automation logic and parameter combinations used, refer to the scripts themselves (`run.sh` and `run_overlap_study.sh`).

---

## Project Structure

```
thesis-tntsegmentation/
├── models/                           # Pre-trained model checkpoints
│   ├── anisotropic_csam_2d/         # Anisotropic U-Net with CSAM (2D version)
│   │   ├── quad1_model/             # Trained on quadrant 1
│   │   │   ├── model_final.pth      # Model weights
│   │   │   └── config.json          # Model configuration
│   │   ├── quad2_model/             # Trained on quadrant 2
│   │   ├── quad3_model/             # Trained on quadrant 3
│   │   └── quad4_model/             # Trained on quadrant 4
│   ├── anisotropic_csam_3d/         # Anisotropic U-Net with CSAM (3D version)
│   │   └── ***
│   ├── anisotropic_usenet_3d/       # Anisotropic U-Net with USE-Net (3D version)
│   │   └── ***
│   ├── anisotropic_usenet_2d/       # Anisotropic U-Net with USE-Net (2D version)
│   │   └── ***
│   ├── anisotropic_basic_2d/        # Basic Anisotropic U-Net (2D version)
│   │   └── ***
│   ├── anisotropic_basic_3d/        # Basic Anisotropic U-Net (3D version)
│   │   └── ***
│   └── unet3d/                      # Standard 3D U-Net (baseline)
│   │   └── ***
│
├── data/                             # Dataset and evaluation data
│   ├── raw/                         # Original raw data
│   └── processed/                   # Processed quadrant splits
│       ├── quad1/                   # Training/test split for quad 1
│       ├── quad2/                   # Training/test split for quad 2
│       ├── quad3/                   # Training/test split for quad 3
│       └── quad4/                   # Training/test split for quad 4
│
├── scripts/                          # Main scripts
│   ├── inference.py                 # Run inference on new images
│   ├── training.py                  # Train models
│   ├── evaluate_models.py           # Evaluate with ground truth
│   ├── postprocess.py               # Post-processing utilities
│   ├── tilingutilities.py           # Tiling and stitching functions
│   ├── training_utils.py            # Training helper functions
│   ├── config.py                    # Configuration utilities
│   ├── run.sh                       # Full pipeline automation
│   ├── run_overlap_study.sh         # Overlap analysis
│   └── graphs/                      # Analysis and visualization scripts
│       ├── aggregateanalysis.py     # Aggregate results analysis
│       ├── archcomparisonquad.py    # Architecture comparison
│       ├── quadbasedanalysis.py     # Quadrant-based analysis
│       ├── visualize_models.py      # Model visualization
│       ├── create_metric_tables.py  # Generate metric tables
│       └── estimate_label_size.py   # Label size estimation
│
├── tntseg/                           # Main Python package
│   ├── __init__.py
│   ├── nn/                          # Neural network components
│   │   ├── models/                  # Model architectures
│   │   │   ├── anisounet3d_basic.py       # Basic anisotropic U-Net
│   │   │   ├── anisounet3d_seblock.py     # With Squeeze-Excitation
│   │   │   ├── anisounet3d_csnet.py       # With CSAM attention
│   │   │   ├── anisounet3d_usenet.py      # With USE-Net modules
│   │   │   └── unet3d_basic.py            # Standard 3D U-Net
│   │   ├── modules.py               # Reusable network modules
│   │   ├── squeeze_excitation.py    # SE block implementation
│   │   └── csnet_affinity_modules.py # CSAM attention modules
│   └── utilities/                   # Utility modules
│       ├── dataset/                 # Dataset handling
│       │   ├── split_data.py        # Quadrant-based data splitting
│       │   ├── datasets.py          # PyTorch dataset classes
│       │   └── analyze.py           # Dataset analysis tools
│       └── metrics/                 # Evaluation metrics
│           ├── metrics.py           # NumPy-based metrics
│           └── metrics_torch.py     # PyTorch-based metrics
│
├── pyproject.toml                    # Project dependencies and metadata
├── LICENSE                           # License file
└── README.md                         # This file
```

---

## License

See [LICENSE](LICENSE) file for details.
