"""DeepForest RetinaNet detector wrapper supporting Apple Silicon MPS GPU and CPU

with boundary Non-Maximum Suppression (NMS) across sliding tiles.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import rasterio
import torch
import torchvision.ops
from deepforest import main

from canopy_core.ingest import RasterMetadata, inspect_raster
from canopy_core.tiler import SlidingWindowTiler, TileWindow

logger = logging.getLogger(__name__)


class TreeDetector:
    """Tree Crown Detector using DeepForest RetinaNet with Apple Silicon MPS GPU acceleration

    and tile boundary Non-Maximum Suppression (NMS).
    """

    def __init__(
        self,
        device: Optional[str] = None,
        score_threshold: float = 0.30,
        iou_threshold: float = 0.25,
    ):
        """Initialize the Tree Crown Detector.

        Args:
            device: Computing device ('mps', 'cuda', 'cpu'). If None, auto-detected.
            score_threshold: Minimum confidence score to retain detection (default 0.30).
            iou_threshold: IoU threshold for boundary NMS suppression (default 0.25).
        """
        if device is None:
            if torch.backends.mps.is_available():
                self.device_str = "mps"
            elif torch.cuda.is_available():
                self.device_str = "cuda"
            else:
                self.device_str = "cpu"
        else:
            self.device_str = device.lower()

        self.device = torch.device(self.device_str)
        self.score_threshold = float(score_threshold)
        self.iou_threshold = float(iou_threshold)

        # Initialize DeepForest model
        self.df_model = main.deepforest()
        self.df_model.load_model()
        self.model = self.df_model.model
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Initialized TreeDetector on device: {self.device_str}")

    def predict_tile(
        self,
        image: np.ndarray,
        score_threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """Run inference on an individual image tile (H, W, 3) in RGB format.

        Args:
            image: NumPy array of shape (H, W, 3) in RGB order (uint8 [0, 255] or float32 [0, 1]).
            score_threshold: Optional override for confidence threshold.

        Returns:
            pd.DataFrame: DataFrame containing detected bounding boxes ['xmin', 'ymin', 'xmax', 'ymax', 'score', 'label'].
        """
        threshold = self.score_threshold if score_threshold is None else float(score_threshold)

        # Ensure correct shape and data type
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected image of shape (H, W, 3), got {image.shape}")

        if image.dtype == np.uint8:
            img_norm = image.astype(np.float32) / 255.0
        elif image.max() > 1.0:
            img_norm = image.astype(np.float32) / 255.0
        else:
            img_norm = image.astype(np.float32)

        # Convert to torch tensor (C, H, W) and transfer to device
        tensor = torch.tensor(img_norm, dtype=torch.float32).permute(2, 0, 1).to(self.device)

        with torch.no_grad():
            predictions = self.model([tensor])[0]

        boxes = predictions["boxes"].detach().cpu().numpy()
        scores = predictions["scores"].detach().cpu().numpy()
        labels = predictions["labels"].detach().cpu().numpy()

        if len(boxes) == 0:
            return pd.DataFrame(columns=["xmin", "ymin", "xmax", "ymax", "score", "label"])

        # Filter by confidence threshold
        mask = scores >= threshold
        boxes_filt = boxes[mask]
        scores_filt = scores[mask]
        labels_filt = labels[mask]

        if len(boxes_filt) == 0:
            return pd.DataFrame(columns=["xmin", "ymin", "xmax", "ymax", "score", "label"])

        # Label mapping from model label dictionary
        label_dict = getattr(self.df_model, "label_dict", {1: "Tree"})
        label_names = [label_dict.get(int(l), "Tree") for l in labels_filt]

        df = pd.DataFrame(
            {
                "xmin": boxes_filt[:, 0],
                "ymin": boxes_filt[:, 1],
                "xmax": boxes_filt[:, 2],
                "ymax": boxes_filt[:, 3],
                "score": scores_filt,
                "label": label_names,
            }
        )

        return df

    def apply_nms(
        self,
        boxes_df: pd.DataFrame,
        iou_threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """Eliminate duplicate bounding boxes across overlapping tile boundaries

        using Non-Maximum Suppression (NMS).

        Args:
            boxes_df: DataFrame with ['xmin', 'ymin', 'xmax', 'ymax', 'score'].
            iou_threshold: IoU overlap threshold (default: self.iou_threshold).

        Returns:
            pd.DataFrame: Deduplicated DataFrame.
        """
        if boxes_df is None or len(boxes_df) == 0:
            return boxes_df

        iou_thresh = self.iou_threshold if iou_threshold is None else float(iou_threshold)

        boxes_tensor = torch.tensor(
            boxes_df[["xmin", "ymin", "xmax", "ymax"]].values,
            dtype=torch.float32,
        )
        scores_tensor = torch.tensor(
            boxes_df["score"].values,
            dtype=torch.float32,
        )

        # Apply torchvision NMS
        keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_thresh)
        keep_indices_np = keep_indices.cpu().numpy()

        dedup_df = boxes_df.iloc[keep_indices_np].reset_index(drop=True)
        return dedup_df

    def predict_tiled_raster(
        self,
        raster_path: Union[str, Path],
        tiler: Optional[SlidingWindowTiler] = None,
        score_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, RasterMetadata]:
        """Perform end-to-end sliding-window tree detection on a large raster mosaic,

        deduplicate overlapping boundary boxes with global NMS, and map detections
        to projected metric CRS coordinates.

        Args:
            raster_path: Path to the GeoTIFF raster file.
            tiler: Optional SlidingWindowTiler instance (default: 400x400 with 20% overlap).
            score_threshold: Optional confidence score threshold.
            iou_threshold: Optional boundary NMS IoU threshold.

        Returns:
            tuple: (detections_df, raster_metadata)
        """
        raster_metadata = inspect_raster(raster_path)

        if tiler is None:
            tiler = SlidingWindowTiler(tile_size=400, overlap=0.20)

        all_tile_predictions: list[pd.DataFrame] = []

        # Iterate over sliding window tiles
        for tile in tiler.iterate_raster(raster_path):
            tile_df = self.predict_tile(tile.image, score_threshold=score_threshold)
            if len(tile_df) > 0:
                global_px_df = tiler.map_local_to_global_pixels(tile, tile_df)
                all_tile_predictions.append(global_px_df)

        if not all_tile_predictions:
            empty_df = pd.DataFrame(
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
            return empty_df, raster_metadata

        raw_df = pd.concat(all_tile_predictions, ignore_index=True)

        # Apply global boundary Non-Maximum Suppression
        dedup_df = self.apply_nms(raw_df, iou_threshold=iou_threshold)

        # Map pixel coordinates to metric projected CRS coordinates
        final_df = tiler.map_pixels_to_crs(
            dedup_df,
            global_transform=raster_metadata.transform,
            crs=rasterio.crs.CRS.from_string(raster_metadata.crs)
            if raster_metadata.crs != "UNDEFINED"
            else None,
        )

        return final_df, raster_metadata
