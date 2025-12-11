"""
3D Volume Tiling and Stitching Utilities.

This module provides functionality for splitting 3D volumes into overlapping tiles
and reconstructing volumes from tile predictions. 

Key Features
------------
- Tile splitting: Divide large 3D volumes into manageable tiles with optional overlap
- Tile stitching: Reconstruct full-volume predictions from individual tile predictions
- Multiple aggregation methods: Mean, Max, and Min for handling overlapping regions
- Visualization: Optional visualization of tile boundaries and overlaps
- Type preservation: Maintains data types through tiling/stitching pipeline

Architecture Constraints
------------------------
- Depth (z-dimension): Must fit entirely within GPU memory (no tiling in z)
- Height and Width: Can be tiled with overlap for flexibility
- This design is optimized for anisotropic 3D medical data

Usage Example
-------------
>>> # Split large volume into tiles
>>> tiles, positions = tile_volume(volume, tile_size=(7, 64, 64), overlap=16)
>>> 
>>> # Process tiles through network
>>> predictions = []
>>> for tile in tiles:
>>>     pred = model(tile)
>>>     predictions.append(pred)
>>> 
>>> # Reconstruct full volume
>>> full_pred = stitch_volume(predictions, positions, original_shape=volume.shape)


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

import numpy as np
import torch
from typing import List, Tuple, TypeVar
from enum import StrEnum, auto

from numpy.typing import NDArray

T = TypeVar("T", bound=np.generic)


def _draw_diagonal_in_square(volume: NDArray, val=(0, 0, 0)) -> None:
    """
    Draw diagonals in a square region (in-place operation).

    Used for visualization of tile boundaries. Modifies the input array in-place.

    Parameters
    ----------
    volume : ndarray
        2D or 3D array representing a square region. Last dimension is used for
        multi-channel data (e.g., RGB visualization).
    val : tuple of int, optional
        Value(s) to set on diagonal pixels. For RGB, should be (R, G, B).
        Default is (0, 0, 0) for black.

    Raises
    ------
    AssertionError
        If the region is not square (rows != cols)

    Notes
    -----
    Draws both main diagonal (top-left to bottom-right) and anti-diagonal
    (top-right to bottom-left).
    """
    rows, cols, _ = volume.shape
    assert rows == cols, "Just squares are supported"
    for i in range(rows):
        volume[i, i] = val
        volume[rows - i - 1, i] = val


def tile_volume(
    volume: NDArray[T], tile_size: Tuple[int, int, int], overlap: int = 0
) -> Tuple[List[NDArray[T]], List[Tuple[int, int, int]]]:
    """
    Split a 3D volume into overlapping tiles.

    Divides a large 3D volume into smaller tiles for processing. Tiles can overlap
    in height and width dimensions. The depth dimension must fit entirely in each tile
    (no tiling in z-direction).

    Tiling Strategy
    ---------------
    - Depth (z): Not tiled; each tile spans full depth
    - Height and Width: Tiled with configurable overlap
    - Edge tiles are automatically shifted to fit volume boundaries
    - All tiles have exactly the specified tile_size

    Parameters
    ----------
    volume : ndarray
        3D volume to tile, shape (depth, height, width)
    tile_size : tuple of int
        Size of each tile as (z_size, height_size, width_size)
    overlap : int, optional
        Number of overlapping pixels between adjacent tiles in height/width.
        No overlap by default (overlap=0).
        Default is 0.

    Returns
    -------
    tiles : list of ndarray
        List of extracted tiles, each with shape tile_size
    positions : list of tuple
        List of (z_start, r_start, c_start) positions for each tile in original volume.
        Used to reconstruct the full volume during stitching.

    Raises
    ------
    ValueError
        If volume is not 3D
    AssertionError
        If any dimension of volume doesn't match tile_size (this only applies to depth
        dimension; height/width are automatically adjusted with overlap)

    Notes
    -----
    - The depth dimension (z_size) is NOT supported for multi-depth tiling.
      Each tile must contain the full depth of the volume.
    - Edge tiles are shifted inward to maintain exact tile_size
    - Stride = tile_size - overlap ensures proper tiling coverage
    """
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

                assert d_end == d_end_allowed, (
                    "Unsupported depth"
                )  # NOTE: we are supporting just our data right now

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

                assert tile.shape == tile_size, (
                    f"{tile.shape} != {tile_size} for d,r,c = {d_start},{r_start},{c_start}"
                )

                tiles.append(tile)
                tiles_position.append((d_start, r_start, c_start))

    return tiles, tiles_position


class AggregationMethod(StrEnum):
    """
    Enumeration of aggregation methods for handling overlapping tiles.

    When tiles overlap during stitching, overlapping regions must be aggregated.
    This enum defines the available aggregation strategies.

    Attributes
    ----------
    Mean : str
        Average overlapping predictions. Recommended for most use cases.
        Provides smooth blending of predictions across tile boundaries.
    Max : str
        Take maximum value in overlapping regions. Useful for binary segmentation
        or when you want to keep the strongest predictions.
    Min : str
        Take minimum value in overlapping regions. Less common; useful for specific
        applications like detecting low-confidence regions.

    """

    Mean = auto()
    Max = auto()
    Min = auto()


def stitch_volume(
    tiles: List[torch.Tensor],
    positions: List[Tuple[int, int, int]],
    original_shape: Tuple[int, int, int],
    aggregation_method: AggregationMethod = AggregationMethod.Mean,
    visualise_lines: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, NDArray[np.uint8]]:
    """
    Reconstruct a full 3D volume from overlapping tile predictions.

    Stitches together tile predictions back into the original volume space,
    handling overlapping regions according to the specified aggregation method.

    Aggregation Strategy
    --------------------
    - Mean: Averages overlapping predictions (smoothest result)
    - Max: Takes maximum value in overlaps (preserves strong predictions)
    - Min: Takes minimum value in overlaps (highlights low-confidence regions)

    Parameters
    ----------
    tiles : list of torch.Tensor
        List of predicted tiles from the model. Each tile should have shape
        (batch, channels, depth, height, width) or (channels, depth, height, width).
        Batch dimension is squeezed if present.
    positions : list of tuple
        List of (z_start, r_start, c_start) positions for each tile,
        matching the output from tile_volume().
    original_shape : tuple of int
        Shape of the original full volume as (depth, height, width).
    aggregation_method : AggregationMethod, optional
        Method for aggregating overlapping regions. Default is Mean.
    visualise_lines : bool, optional
        If True, also generate a visualization image showing tile boundaries.
        Tiles are colored alternately red and blue, with diagonals drawn.
        Default is False.

    Returns
    -------
    result : torch.Tensor
        Reconstructed full volume with shape original_shape
    visualisation : ndarray, optional
        Only returned if visualise_lines=True. RGB visualization of tile boundaries,
        shape (height, width, 3) with dtype uint8.

    Raises
    ------
    AssertionError
        If tile shapes don't match expected dimensions or aggregation fails
    """
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
            assert (
                tile.shape == result[d_start:d_end, r_start:r_end, c_start:c_end].shape
            ), (
                f"{tile.shape}; {result[d_start:d_end, r_start:r_end, c_start:c_end].shape}"
            )
            result[d_start:d_end, r_start:r_end, c_start:c_end] += tile
            counts[d_start:d_end, r_start:r_end, c_start:c_end] += 1
        elif aggregation_method == AggregationMethod.Max:
            current = result[d_start:d_end, r_start:r_end, c_start:c_end]
            result[d_start:d_end, r_start:r_end, c_start:c_end] = torch.maximum(
                current, tile
            )
        else:  # Min
            current = result[d_start:d_end, r_start:r_end, c_start:c_end]
            mask = (current == 0) | (tile < current)
            result[d_start:d_end, r_start:r_end, c_start:c_end][mask] = tile[mask]

        if visualise_lines:
            val = (255, 0, 0) if idx % 2 == 0 else (0, 0, 255)
            visualisation[r_start:r_end, c_start] = val  # left line
            visualisation[r_end - 1, c_start:c_end] = val  # bottom line
            visualisation[r_start, c_start:c_end] = val  # top line
            visualisation[r_start:r_end, c_end - 1] = val  # right line
            _draw_diagonal_in_square(
                visualisation[r_start:r_end, c_start:c_end], val=val
            )  # Diagonals

    # Compute final result based on aggregation method
    if aggregation_method == AggregationMethod.Mean:
        # Avoid division by zero
        counts[counts == 0] = 1
        result = result / counts

    if visualise_lines:
        return result, visualisation
    return result
