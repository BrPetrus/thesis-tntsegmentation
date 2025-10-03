import numpy as np
import torch
from typing import List, Tuple, Union, Optional, TypeVar
from dataclasses import dataclass
from enum import StrEnum, auto

from numpy.typing import NDArray
from torch.types import Tensor

T = TypeVar('T', bound=np.generic)

def _draw_diagonal_in_square(volume: NDArray, val=(0, 0, 0)):
    rows, cols, _ = volume.shape
    assert rows == cols, "Just squares are supported"
    for i in range(rows):
        volume[i, i] = val
        volume[rows-i-1, i] = val

def tile_volume(volume: NDArray[T], tile_size: Tuple[int, int, int],
                overlap: int = 0) -> Tuple[NDArray[T], List[Tuple[int, int, int]]]:
    if volume.ndim != 3:
        raise ValueError("Expected 3D data")
    depth, rows, cols = volume.shape
    
    d_size, r_size, c_size = tile_size
    tiles = []
    tiles_position = []
    
    for d_start in range(0, depth, d_size):
        for r_start in range(0, rows, r_size - overlap):
            for c_start in range(0, cols, c_size - overlap):
                d_end = d_start + d_size
                r_end = r_start + r_size
                c_end = c_start + c_size

                d_end_allowed = min(depth, d_end)
                r_end_allowed = min(rows, r_end)
                c_end_allowed = min(cols, c_end)

                assert d_end == d_end_allowed, "Unsupported depth"  # NOTE: we are supporting just our data right now
                
                if r_end_allowed != r_end:
                    diff = r_end - r_end_allowed

                    # Shift the tile
                    r_start -= diff
                    r_end -= diff
                if c_end_allowed != c_end:
                    diff = c_end - c_end_allowed

                    # Shift the tile
                    c_start -= diff
                    c_end -= diff

                tile = volume[d_start:d_end, r_start:r_end, c_start:c_end]

                assert tile.shape == tile_size, f"{tile.shape} != {tile_size} for d,r,c = {d_start},{r_start},{c_start}"

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
                  aggregation_method: AggregationMethod = AggregationMethod.Mean,
                  visualise_lines: bool = False) -> NDArray[T] | Tuple[NDArray[T], NDArray[np.uint8]]:
    # Initialize output volume and count matrix for averaging
    depth, rows, cols = original_shape
    result = torch.zeros(original_shape, dtype=tiles[0].dtype)
    counts = torch.zeros(original_shape, dtype=torch.int32)

    if visualise_lines:
        visualisation = np.zeros((rows, cols, 3), dtype=np.uint8)
        visualisation[:, :] = (255, 255, 255)

    # Process each tile
    for idx, (d_start, r_start, c_start) in enumerate(positions):
        tile = tiles[idx].squeeze(dim=0)
        tile_depth, tile_rows, tile_cols = tile.shape

        d_ideal_end = d_start + tile_depth
        r_ideal_end = r_start + tile_rows
        c_ideal_end = c_start + tile_cols

        d_end = min(d_ideal_end, depth)
        r_end = min(r_ideal_end, rows)
        c_end = min(c_ideal_end, cols)

        tile_d_end = tile_depth - (d_ideal_end - d_end)
        tile_r_end = tile_rows - (r_ideal_end - r_end)
        tile_c_end = tile_cols - (c_ideal_end - c_end)

        # tile = tile[d_start:d_end, r_start:r_end, c_start:c_end]  # Overlaping might produce extra padding on the borders
        tile = tile[:tile_d_end, :tile_r_end, :tile_c_end]

        assert d_ideal_end - d_end == tile_depth - tile.shape[0]
        assert r_ideal_end - r_end == tile_rows - tile.shape[1]
        assert c_ideal_end - c_end == tile_cols - tile.shape[2]
        
        if aggregation_method == AggregationMethod.Mean:
            assert tile.shape == result[d_start:d_end, r_start:r_end, c_start:c_end].shape, f"{tile.shape}; {result[d_start:d_end,r_start:r_end, c_start:c_end].shape}"
            result[d_start:d_end, r_start:r_end, c_start:c_end] += tile
            counts[d_start:d_end, r_start:r_end, c_start:c_end] += 1
        elif aggregation_method == AggregationMethod.Max:
            current = result[d_start:d_end, r_start:r_end, c_start:c_end]
            result[d_start:d_end, r_start:r_end, c_start:c_end] = np.maximum(current, tile)
        else:  # Min
            current = result[d_start:d_end, r_start:r_end, c_start:c_end]
            mask = (current == 0) | (tile < current)
            result[d_start:d_end, r_start:r_end, c_start:c_end][mask] = tile[mask]

        if visualise_lines:
            val = (255, 0, 0) if idx % 2 == 0 else (0, 0, 255)
            visualisation[r_start:r_end, c_start] = val  # left line
            visualisation[r_end-1, c_start:c_end] = val  # bottom line
            visualisation[r_start, c_start:c_end] = val  # top line
            visualisation[r_start:r_end, c_end-1] = val  # right line
            _draw_diagonal_in_square(visualisation[r_start:r_end, c_start:c_end], val=val)  # Diagonals

    # Compute final result based on aggregation method
    if aggregation_method == AggregationMethod.Mean:
        # Avoid division by zero
        counts[counts == 0] = 1
        result = result / counts

    if visualise_lines:
        return result, visualisation
    return result