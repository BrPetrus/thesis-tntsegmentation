import numpy as np
import torch
from typing import List, Tuple, Union, Optional, TypeVar
from dataclasses import dataclass
from enum import StrEnum, auto

from numpy.typing import NDArray
from torch.types import Tensor

T = TypeVar('T', bound=np.generic)

                # overlap: int = 0) -> List[Tuple[NDArray[T], int, int, int]]:
def tile_volume(volume: NDArray[T], tile_size: Tuple[int, int, int],
                overlap: int = 0) -> Tuple[NDArray[T], List[Tuple[int, int, int]]]:
    if volume.ndim != 3:
        raise ValueError("Expected 3D data")
    depth, rows, cols = volume.shape
    
    d_size, r_size, c_size = tile_size
    tiles = []
    tiles_position = []
    
    for d in range(0, depth, d_size - overlap):
        for r in range(0, rows, r_size - overlap):
            for c in range(0, cols, c_size - overlap):
                # Calculate end points with overlap
                d_end = min(d + d_size, depth)
                r_end = min(r + r_size, rows)
                c_end = min(c + c_size, cols)
                
                # Adjust start points if at the end
                d_start = max(d_end - d_size, 0)
                r_start = max(r_end - r_size, 0)
                c_start = max(c_end - c_size, 0)
                
                # Extract tile and store with its position
                tile = volume[d_start:d_end, r_start:r_end, c_start:c_end]
                tiles.append(tile)
                tiles_position.append((d_start, r_start, c_start))
    
    return tiles, tiles_position

class AggregationMethod(StrEnum):
    Mean = auto()
    Max = auto()
    Min = auto()

def stitch_volume(tiles: List[torch.Tensor],
                  positions: List[Tuple[int, int, int]], 
                  original_shape: Tuple[int, int, int], 
                  aggregation_method: AggregationMethod = AggregationMethod.Mean) -> NDArray[T]:
    # Initialize output volume and count matrix for averaging
    depth, rows, cols = original_shape
    result = torch.zeros(original_shape, dtype=tiles[0].dtype)
    counts = torch.zeros(original_shape, dtype=torch.int32)

    # Process each tile
    for idx, (d_start, r_start, c_start) in enumerate(positions):
        tile = tiles[idx].squeeze(dim=0)

        d_end = d_start + tile.shape[0]
        r_end = r_start + tile.shape[1]
        c_end = c_start + tile.shape[2]
        
        if aggregation_method == AggregationMethod.Mean:
            result[d_start:d_end, r_start:r_end, c_start:c_end] += tile
            counts[d_start:d_end, r_start:r_end, c_start:c_end] += 1
        elif aggregation_method == AggregationMethod.Max:
            current = result[d_start:d_end, r_start:r_end, c_start:c_end]
            result[d_start:d_end, r_start:r_end, c_start:c_end] = np.maximum(current, tile)
        else:  # Min
            current = result[d_start:d_end, r_start:r_end, c_start:c_end]
            mask = (current == 0) | (tile < current)
            result[d_start:d_end, r_start:r_end, c_start:c_end][mask] = tile[mask]

    # Compute final result based on aggregation method
    if aggregation_method == AggregationMethod.Mean:
        # Avoid division by zero
        counts[counts == 0] = 1
        result = result / counts

    return result