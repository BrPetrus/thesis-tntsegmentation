"""
Benchmarking tool for 3D segmentation models.

Loads trained models from checkpoint files and their associated config.json
to benchmark performance metrics including:
- Peak VRAM usage across different batch sizes
- Inference time
- Model parameters count
- Throughput (samples per second)

Copyright 2025 Bruno Petrus

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import torch
import numpy as np
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple, List
import time

from tntseg.nn.models.anisounet3d_basic import AnisotropicUNet3D
from tntseg.nn.models.anisounet3d_seblock import AnisotropicUNet3DSE
from tntseg.nn.models.anisounet3d_csnet import AnisotropicUNet3DCSAM
from tntseg.nn.models.unet3d_basic import UNet3d
from tntseg.nn.models.anisounet3d_usenet import AnisotropicUSENet


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config_from_checkpoint(checkpoint_path: Path) -> Dict:
    """Load training configuration from checkpoint directory.
    
    Parameters
    ----------
    checkpoint_path : Path
        Path to the checkpoint file (.pth)
        
    Returns
    -------
    Dict
        Configuration dictionary containing model and training parameters
        
    Raises
    ------
    FileNotFoundError
        If config.json is not found in the checkpoint directory
    """
    model_dir = checkpoint_path.parent
    config_path = model_dir / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"No config found at {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    logger.info(f"Loaded configuration from {config_path}")
    return config


def create_model_from_config(config: Dict) -> torch.nn.Module:
    """Create a model instance based on loaded configuration.
    
    Parameters
    ----------
    config : Dict
        Configuration dictionary with model parameters
        
    Returns
    -------
    torch.nn.Module
        Instantiated model
        
    Raises
    ------
    ValueError
        If model_type is not recognized
    """
    model_type = config["model_type"]

    if model_type == "anisotropicunet":
        return AnisotropicUNet3D(
            n_channels_in=1,
            n_classes_out=1,
            depth=config["model_depth"],
            base_channels=config["base_channels"],
            channel_growth=config["channel_growth"],
            horizontal_kernel=tuple(config["horizontal_kernel"]),
            horizontal_padding=tuple(config["horizontal_padding"]),
            upscale_kernel=tuple(config["upscale_kernel"]),
            upscale_stride=tuple(config["upscale_stride"]),
            downscale_kernel=tuple(config["downscale_kernel"]),
            downscale_stride=tuple(config["downscale_stride"]),
        )
    elif model_type == "anisotropicunet_se":
        return AnisotropicUNet3DSE(
            n_channels_in=1,
            n_classes_out=1,
            depth=config["model_depth"],
            base_channels=config["base_channels"],
            channel_growth=config["channel_growth"],
            horizontal_kernel=tuple(config["horizontal_kernel"]),
            horizontal_padding=tuple(config["horizontal_padding"]),
            squeeze_factor=config.get("reduction_factor", 16),
            upscale_kernel=tuple(config["upscale_kernel"]),
            upscale_stride=tuple(config["upscale_stride"]),
            downscale_kernel=tuple(config["downscale_kernel"]),
            downscale_stride=tuple(config["downscale_stride"]),
        )
    elif model_type == "anisotropicunet_csam":
        return AnisotropicUNet3DCSAM(
            n_channels_in=1,
            n_classes_out=1,
            depth=config["model_depth"],
            base_channels=config["base_channels"],
            channel_growth=config["channel_growth"],
            horizontal_kernel=tuple(config["horizontal_kernel"]),
            horizontal_padding=tuple(config["horizontal_padding"]),
            upscale_kernel=tuple(config["upscale_kernel"]),
            upscale_stride=tuple(config["upscale_stride"]),
            downscale_kernel=tuple(config["downscale_kernel"]),
            downscale_stride=tuple(config["downscale_stride"]),
        )
    elif model_type == "anisotropicunet_usenet":
        return AnisotropicUSENet(
            n_channels_in=1,
            n_classes_out=1,
            depth=config["model_depth"],
            base_channels=config["base_channels"],
            channel_growth=config["channel_growth"],
            horizontal_kernel=tuple(config["horizontal_kernel"]),
            horizontal_padding=tuple(config["horizontal_padding"]),
            upscale_kernel=tuple(config["upscale_kernel"]),
            upscale_stride=tuple(config["upscale_stride"]),
            downscale_kernel=tuple(config["downscale_kernel"]),
            downscale_stride=tuple(config["downscale_stride"]),
        )
    elif model_type == "unet3d":
        return UNet3d(n_channels_in=1, n_classes_out=1)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_model_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters in model.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model to analyze
        
    Returns
    -------
    Tuple[int, int]
        (total_parameters, trainable_parameters)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def benchmark_memory(
    model: torch.nn.Module,
    batch_size: int,
    channels: int = 1,
    depth: int = 7,
    height: int = 64,
    width: int = 64,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    num_runs: int = 200,
) -> Dict[str, float]:
    """Benchmark peak VRAM usage for a model.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model to benchmark
    batch_size : int
        Batch size for inference
    channels : int, optional
        Number of input channels, by default 1
    depth : int, optional
        Depth of input volume, by default 7
    height : int, optional
        Height of input volume, by default 64
    width : int, optional
        Width of input volume, by default 64
    device : torch.device, optional
        Device to run benchmark on, by default CUDA if available
    num_runs : int, optional
        Number of runs to average over (excluding warm-up), by default 200
        
    Returns
    -------
    Dict[str, float]
        Dictionary with peak_memory_gb, peak_memory_mb, and avg_time_ms
    """
    model.eval()
    model.to(device)

    peak_memory_mb_list = []
    inference_times = []

    with torch.no_grad():
        # Warm-up run and initial cache clear
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        
        dummy_input = torch.randn(
            batch_size, channels, depth, height, width,
            device=device,
            dtype=torch.float32
        )
        _ = model(dummy_input)
        
        # Actual benchmark runs (without cache clearing between runs)
        for i in range(num_runs):
            # Only reset stats at the beginning of each run, don't empty cache
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            # Create dummy input (before timing)
            dummy_input = torch.randn(
                batch_size, channels, depth, height, width,
                device=device,
                dtype=torch.float32
            )

            # Benchmark inference only (not tensor creation)
            start_time = time.perf_counter()
            _ = model(dummy_input)
            end_time = time.perf_counter()
            inference_times.append((end_time - start_time) * 1000)  # Convert to ms

            # Get peak memory usage
            if device.type == "cuda":
                peak_allocated_bytes = torch.cuda.max_memory_allocated(device)
                peak_allocated_mb = peak_allocated_bytes / (1024**2)
                peak_memory_mb_list.append(peak_allocated_mb)

    # Calculate statistics
    avg_peak_memory_mb = np.mean(peak_memory_mb_list) if peak_memory_mb_list else 0
    avg_peak_memory_gb = avg_peak_memory_mb / 1024
    avg_inference_time_ms = np.mean(inference_times)
    std_inference_time_ms = np.std(inference_times)

    return {
        "peak_memory_gb": round(avg_peak_memory_gb, 4),
        "peak_memory_mb": round(avg_peak_memory_mb, 2),
        "avg_inference_time_ms": round(avg_inference_time_ms, 4),
        "std_inference_time_ms": round(std_inference_time_ms, 4),
        "throughput_samples_per_sec": round(
            batch_size / (avg_inference_time_ms / 1000), 2
        ),
    }


def benchmark_model(
    checkpoint_path: Path,
    batch_sizes: List[int] = None,
    device: str = "cuda",
) -> Dict:
    """Full benchmark suite for a model.
    
    Parameters
    ----------
    checkpoint_path : Path
        Path to checkpoint file
    batch_sizes : List[int], optional
        Batch sizes to benchmark, by default [1, 2, 4, 8]
    device : str, optional
        Device to use ('cuda' or 'cpu'), by default 'cuda'
        
    Returns
    -------
    Dict
        Benchmark results including memory, timing, and parameters
    """
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8]

    device_obj = torch.device(device)

    # Load config and create model
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    config = load_config_from_checkpoint(checkpoint_path)
    model = create_model_from_config(config)

    # Load checkpoint weights
    checkpoint = torch.load(checkpoint_path, map_location=device_obj)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Count parameters
    total_params, trainable_params = count_model_parameters(model)

    # Run benchmarks
    results = {
        "checkpoint_path": str(checkpoint_path),
        "model_type": config["model_type"],
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "device": str(device_obj),
        "batch_size_benchmarks": {},
    }

    logger.info(f"Model type: {config['model_type']}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    for batch_size in batch_sizes:
        logger.info(f"\nBenchmarking batch size: {batch_size}")
        try:
            metrics = benchmark_memory(model, batch_size, device=device_obj)
            results["batch_size_benchmarks"][batch_size] = metrics

            logger.info(
                f"  Peak VRAM: {metrics['peak_memory_gb']:.4f} GB "
                f"({metrics['peak_memory_mb']:.2f} MB)"
            )
            logger.info(
                f"  Avg inference time: {metrics['avg_inference_time_ms']:.4f} ms "
                f"(± {metrics['std_inference_time_ms']:.4f} ms)"
            )
            logger.info(
                f"  Throughput: {metrics['throughput_samples_per_sec']:.2f} "
                "samples/sec"
            )
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning(f"  Out of memory for batch size {batch_size}")
                results["batch_size_benchmarks"][batch_size] = {"error": "OOM"}
            else:
                raise

    return results


def print_benchmark_summary(results: Dict) -> None:
    """Print a formatted summary of benchmark results.
    
    Parameters
    ----------
    results : Dict
        Results dictionary from benchmark_model
    """
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    print(f"Checkpoint: {Path(results['checkpoint_path']).name}")
    print(f"Model Type: {results['model_type']}")
    print(f"Device: {results['device']}")
    print(f"Total Parameters: {results['total_parameters']:,}")
    print(f"Trainable Parameters: {results['trainable_parameters']:,}")

    print("\nBatch Size Benchmarks:")
    print("-"*70)

    for batch_size, metrics in results["batch_size_benchmarks"].items():
        if "error" in metrics:
            print(f"  Batch {batch_size}: {metrics['error']}")
        else:
            print(f"  Batch {batch_size}:")
            print(f"    Peak VRAM: {metrics['peak_memory_gb']:.4f} GB "
                  f"({metrics['peak_memory_mb']:.2f} MB)")
            print(f"    Avg Inference Time: {metrics['avg_inference_time_ms']:.4f} ms "
                  f"(± {metrics['std_inference_time_ms']:.4f} ms)")
            print(f"    Throughput: {metrics['throughput_samples_per_sec']:.2f} "
                  "samples/sec")
    print("="*70 + "\n")


def main():
    """Main entry point for benchmarking tool."""
    parser = argparse.ArgumentParser(
        description="Benchmark 3D segmentation models"
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to checkpoint file or directory containing checkpoints",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1],
        help="Batch sizes to benchmark (default: 1 2 4 8)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to benchmark on (default: cuda)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results (optional)",
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.is_file():
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        return

    results = benchmark_model(
        checkpoint_path,
        batch_sizes=args.batch_sizes,
        device=args.device,
    )
    print_benchmark_summary(results)

    # Save results to JSON if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
