"""Sliding-window tiling generator and geospatial coordinate mapping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyproj
import rasterio
from rasterio.windows import Window


@dataclass
class TileWindow:
    """Represents a single sliding-window tile slice."""

    col_off: int
    row_off: int
    width: int
    height: int
    tile_idx: int
    total_tiles: int
    transform: rasterio.Affine
    image: Optional[np.ndarray] = None  # (H, W, 3) in RGB


class SlidingWindowTiler:
    """Generates sliding-window tiles to prevent OOM errors on large raster mosaics

    and maps local detections back to global raster and projected CRS coordinates.
    """

    def __init__(
        self,
        tile_size: Union[int, Tuple[int, int]] = 400,
        overlap: float = 0.20,
    ):
        """Initialize the SlidingWindowTiler.

        Args:
            tile_size: Size of tile in pixels (int for square or (height, width)).
            overlap: Overlap fraction between adjacent windows (e.g. 0.20 for 20% overlap).
        """
        if isinstance(tile_size, int):
            self.tile_h = tile_size
            self.tile_w = tile_size
        else:
            self.tile_h, self.tile_w = tile_size

        if not (0.0 <= overlap < 1.0):
            raise ValueError(f"Overlap must be in range [0.0, 1.0), got {overlap}")
        self.overlap = overlap

    def compute_offsets(self, total_len: int, win_len: int) -> List[int]:
        """Compute 1D window start offsets guaranteeing complete boundary coverage."""
        if total_len <= win_len:
            return [0]

        step = max(1, int(win_len * (1.0 - self.overlap)))
        offsets = list(range(0, total_len - win_len, step))

        # Ensure last window touches the final boundary exactly
        last_offset = total_len - win_len
        if offsets[-1] != last_offset:
            offsets.append(last_offset)

        return offsets

    def get_window_grid(
        self,
        raster_width: int,
        raster_height: int,
        base_transform: Optional[rasterio.Affine] = None,
    ) -> List[TileWindow]:
        """Compute the full 2D grid of TileWindows for a given raster size."""
        x_offsets = self.compute_offsets(raster_width, self.tile_w)
        y_offsets = self.compute_offsets(raster_height, self.tile_h)

        total_tiles = len(x_offsets) * len(y_offsets)
        tile_windows: List[TileWindow] = []

        tile_idx = 0
        for row_off in y_offsets:
            win_h = min(self.tile_h, raster_height - row_off)
            for col_off in x_offsets:
                win_w = min(self.tile_w, raster_width - col_off)

                if base_transform is not None:
                    tile_transform = rasterio.windows.transform(
                        Window(col_off, row_off, win_w, win_h), base_transform
                    )
                else:
                    tile_transform = rasterio.Affine.identity()

                tile_windows.append(
                    TileWindow(
                        col_off=col_off,
                        row_off=row_off,
                        width=win_w,
                        height=win_h,
                        tile_idx=tile_idx,
                        total_tiles=total_tiles,
                        transform=tile_transform,
                    )
                )
                tile_idx += 1

        return tile_windows

    def iterate_raster(
        self,
        raster_path_or_src: Union[str, Path, rasterio.DatasetReader],
        bands: Tuple[int, int, int] = (1, 2, 3),
    ) -> Generator[TileWindow, None, None]:
        """Iterate over raster tiles yielding TileWindows with RGB numpy image data.

        Args:
            raster_path_or_src: Path to raster or open rasterio DatasetReader.
            bands: 1-indexed band numbers to read as RGB (default (1, 2, 3)).

        Yields:
            TileWindow: Window containing image array of shape (H, W, 3).
        """
        close_on_exit = False
        if isinstance(raster_path_or_src, (str, Path)):
            src = rasterio.open(str(raster_path_or_src))
            close_on_exit = True
        else:
            src = raster_path_or_src

        try:
            tile_grid = self.get_window_grid(src.width, src.height, src.transform)

            for tile in tile_grid:
                window = Window(tile.col_off, tile.row_off, tile.width, tile.height)
                # Read selected bands
                arr = src.read(bands, window=window)  # Shape: (3, H, W)
                # Transpose to (H, W, 3) for standard vision pipelines
                img_rgb = np.transpose(arr, (1, 2, 0))

                tile.image = img_rgb
                yield tile
        finally:
            if close_on_exit:
                src.close()

    def iterate_array(
        self,
        image_array: np.ndarray,
        base_transform: Optional[rasterio.Affine] = None,
    ) -> Generator[TileWindow, None, None]:
        """Iterate over an in-memory numpy array (H, W, C)."""
        h, w = image_array.shape[:2]
        tile_grid = self.get_window_grid(w, h, base_transform)

        for tile in tile_grid:
            tile.image = image_array[
                tile.row_off : tile.row_off + tile.height,
                tile.col_off : tile.col_off + tile.width,
            ]
            yield tile

    @staticmethod
    def map_local_to_global_pixels(
        tile_window: TileWindow,
        predictions_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Map tile-local bounding box pixel coordinates back to global raster pixel coordinates.

        Args:
            tile_window: The tile window the predictions were made on.
            predictions_df: DataFrame with ['xmin', 'ymin', 'xmax', 'ymax', ...]

        Returns:
            pd.DataFrame: Copy of DataFrame with updated coordinates.
        """
        if predictions_df is None or len(predictions_df) == 0:
            return pd.DataFrame(
                columns=[
                    "xmin",
                    "ymin",
                    "xmax",
                    "ymax",
                    "score",
                    "label",
                    "tile_idx",
                ]
            )

        df = predictions_df.copy()
        df["xmin"] = df["xmin"] + tile_window.col_off
        df["ymin"] = df["ymin"] + tile_window.row_off
        df["xmax"] = df["xmax"] + tile_window.col_off
        df["ymax"] = df["ymax"] + tile_window.row_off
        df["tile_idx"] = tile_window.tile_idx

        return df

    @staticmethod
    def map_pixels_to_crs(
        detections_df: pd.DataFrame,
        global_transform: rasterio.Affine,
        crs: Optional[rasterio.crs.CRS] = None,
    ) -> pd.DataFrame:
        """Convert global pixel coordinates to metric projected CRS coordinates

        and compute real-world crown dimensions (width, height, area).

        Args:
            detections_df: DataFrame with global pixel coordinates ['xmin', 'ymin', 'xmax', 'ymax'].
            global_transform: The raster's global Affine transform.
            crs: Optional CRS of the raster.

        Returns:
            pd.DataFrame: Enriched DataFrame with geospatial coordinates and crown metrics.
        """
        if detections_df is None or len(detections_df) == 0:
            return pd.DataFrame(
                columns=[
                    "xmin",
                    "ymin",
                    "xmax",
                    "ymax",
                    "score",
                    "label",
                    "geo_xmin",
                    "geo_ymin",
                    "geo_xmax",
                    "geo_ymax",
                    "center_x",
                    "center_y",
                    "width_m",
                    "height_m",
                    "crown_diameter_m",
                ]
            )

        df = detections_df.copy()

        # Transform pixel corners to CRS coordinates
        # Top-left (xmin, ymin) and bottom-right (xmax, ymax)
        x0, y0 = global_transform * (df["xmin"].values, df["ymin"].values)
        x1, y1 = global_transform * (df["xmax"].values, df["ymax"].values)

        x0, x1 = np.asarray(x0), np.asarray(x1)
        y0, y1 = np.asarray(y0), np.asarray(y1)

        geo_xmin = np.minimum(x0, x1)
        geo_xmax = np.maximum(x0, x1)
        geo_ymin = np.minimum(y0, y1)
        geo_ymax = np.maximum(y0, y1)

        df["geo_xmin"] = geo_xmin
        df["geo_ymin"] = geo_ymin
        df["geo_xmax"] = geo_xmax
        df["geo_ymax"] = geo_ymax

        df["center_x"] = (geo_xmin + geo_xmax) / 2.0
        df["center_y"] = (geo_ymin + geo_ymax) / 2.0

        width_m = geo_xmax - geo_xmin
        height_m = geo_ymax - geo_ymin

        df["width_m"] = width_m
        df["height_m"] = height_m

        # Crown diameter: geometric mean of major and minor axes
        df["crown_diameter_m"] = np.sqrt(np.maximum(1e-6, width_m * height_m))

        # Add WGS84 geographic center if projected CRS is supplied
        if crs is not None and crs.is_projected:
            try:
                transformer = pyproj.Transformer.from_crs(
                    crs, "EPSG:4326", always_xy=True
                )
                lons, lats = transformer.transform(
                    df["center_x"].values, df["center_y"].values
                )
                df["lon"] = lons
                df["lat"] = lats
            except Exception:
                pass

        return df
