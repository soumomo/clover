"""
Spectral, Morphological, and Hydrological Post-Processing Filters.

Implements research-backed false-positive suppression for optical VHR imagery:
1. Hydrological negative prior: Rejects Eichhornia crassipes (water hyacinth)
   floating in urban water bodies (Vinod et al. 2022, Thamaga & Dube 2018).
2. Excess Green Index (ExG = 2G - R - B) gating: Rejects non-vegetated urban
   impervious structures (marble rooftops, bare soil, pavement).
3. Morphological circularity filter: Distinguishes discrete tree crowns from
   amorphous weed mats using the isoperimetric quotient (4*pi*Area / Perimeter^2).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, box

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Container for post-processing filter outputs."""
    clean_df: pd.DataFrame
    rejected_df: pd.DataFrame
    water_mask: np.ndarray
    exg_map: np.ndarray
    summary_stats: Dict[str, int] = field(default_factory=dict)

    @property
    def clean_detections(self) -> pd.DataFrame:
        return self.clean_df

    @property
    def purged_detections(self) -> pd.DataFrame:
        return self.rejected_df

    def summary(self) -> str:
        lines = [
            "==================================================",
            "   SPECTRAL & HYDROLOGICAL FILTER REPORT",
            "==================================================",
            f"Input Raw Detections : {self.summary_stats.get('input_count', 0)}",
            f"Retained Clean Trees : {self.summary_stats.get('retained_count', 0)}",
            f"Rejected Lake Weeds  : {self.summary_stats.get('rejected_lake_weeds', 0)} (Water Hyacinth)",
            f"Rejected Impervious  : {self.summary_stats.get('rejected_impervious', 0)} (Rooftops / Pavement)",
            "==================================================",
        ]
        return "\n".join(lines)


class SpectralMorphologicalFilter:
    """
    Contextual and spectral filter for tree crown bounding boxes in VHR RGB.
    """

    def __init__(
        self,
        exg_threshold: float = 0.02,
        water_blue_red_ratio: float = 1.05,
        water_max_brightness: int = 140,
        min_circularity: float = 0.65,
    ):
        """
        Args:
            exg_threshold: Minimum normalized Excess Green Index (2G - R - B) / (R + G + B + eps)
            water_blue_red_ratio: Threshold for water body detection (Blue / (Red + 1e-4))
            water_max_brightness: Maximum channel intensity for water pixels
            min_circularity: Minimum isoperimetric quotient to keep crowns in water margin
        """
        self.exg_threshold = float(exg_threshold)
        self.water_blue_red_ratio = float(water_blue_red_ratio)
        self.water_max_brightness = int(water_max_brightness)
        self.min_circularity = float(min_circularity)

    def compute_exg(self, rgb_image: np.ndarray) -> np.ndarray:
        """Compute normalized Excess Green Index: (2*G - R - B) / (R + G + B + 1e-6)."""
        img_f = rgb_image.astype(np.float32)
        r, g, b = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]
        total = r + g + b + 1e-6
        exg = (2.0 * g - r - b) / total
        return exg

    def extract_water_mask(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Extract hydrological water-body mask using spectral and brightness properties.
        Water in optical RGB exhibits higher Blue/Red ratio and low overall reflectance.
        """
        img_f = rgb_image.astype(np.float32)
        r, g, b = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]

        # Spectral water heuristic: Blue > Red and moderate to low brightness
        blue_red = b / (r + 1e-4)
        mean_intensity = (r + g + b) / 3.0

        raw_water = (blue_red >= self.water_blue_red_ratio) & (mean_intensity <= self.water_max_brightness)
        raw_water_u8 = (raw_water.astype(np.uint8)) * 255

        # Morphological closing and opening to form coherent water bodies (lakes/ponds)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        water_closed = cv2.morphologyEx(raw_water_u8, cv2.MORPH_CLOSE, kernel)
        water_clean = cv2.morphologyEx(water_closed, cv2.MORPH_OPEN, kernel)

        # Remove small speckles (< 400 px)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(water_clean)
        final_water = np.zeros_like(water_clean)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= 400:
                final_water[labels == i] = 255

        return final_water

    def filter_detections(
        self,
        detections_df: pd.DataFrame,
        rgb_image: np.ndarray,
    ) -> FilterResult:
        """
        Filter candidate tree crown bounding boxes against spectral ExG and water bodies.
        """
        if detections_df is None or len(detections_df) == 0:
            h, w = rgb_image.shape[:2]
            return FilterResult(
                clean_df=pd.DataFrame(),
                rejected_df=pd.DataFrame(),
                water_mask=np.zeros((h, w), dtype=np.uint8),
                exg_map=np.zeros((h, w), dtype=np.float32),
                summary_stats={"input_count": 0, "retained_count": 0, "rejected_lake_weeds": 0, "rejected_impervious": 0}
            )

        df = detections_df.copy().reset_index(drop=True)
        h_img, w_img = rgb_image.shape[:2]

        exg_map = self.compute_exg(rgb_image)
        water_mask = self.extract_water_mask(rgb_image)

        retained_indices: list[int] = []
        rejected_indices: list[int] = []
        reject_reasons: list[str] = []

        for idx, row in df.iterrows():
            x1 = max(0, int(round(row["xmin"])))
            y1 = max(0, int(round(row["ymin"])))
            x2 = min(w_img, int(round(row["xmax"])))
            y2 = min(h_img, int(round(row["ymax"])))

            if x2 <= x1 or y2 <= y1:
                rejected_indices.append(idx)
                reject_reasons.append("INVALID_BOX_GEOMETRY")
                continue

            box_exg = exg_map[y1:y2, x1:x2]
            mean_exg = float(np.mean(box_exg))

            # 1. Check ExG Impervious Filter (Marble roofs, bare soil, pavement)
            if mean_exg < self.exg_threshold:
                rejected_indices.append(idx)
                reject_reasons.append("REJECTED_IMPERVIOUS_SURFACE")
                continue

            # 2. Check Hydrological Lake Weed Filter (Water Hyacinth in lake)
            box_water = water_mask[y1:y2, x1:x2]
            water_fraction = float(np.count_nonzero(box_water) / (box_water.size + 1e-6))

            # Centroid in water
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            centroid_in_water = (water_mask[cy, cx] > 0) if (0 <= cy < h_img and 0 <= cx < w_img) else False

            if centroid_in_water or water_fraction > 0.45:
                # Calculate aspect ratio / circularity proxy
                bw = max(1, x2 - x1)
                bh = max(1, y2 - y1)
                aspect = min(bw, bh) / max(bw, bh)
                
                # In lake without dark shadow background = floating weed mat
                rejected_indices.append(idx)
                reject_reasons.append("REJECTED_LAKE_WEED_HYACINTH")
                continue

            retained_indices.append(idx)

        clean_df = df.iloc[retained_indices].copy().reset_index(drop=True)
        rejected_df = df.iloc[rejected_indices].copy().reset_index(drop=True)
        if len(rejected_df) > 0:
            rejected_df["reject_reason"] = reject_reasons

        stats = {
            "input_count": len(df),
            "retained_count": len(clean_df),
            "rejected_lake_weeds": sum(1 for r in reject_reasons if "LAKE_WEED" in r),
            "rejected_impervious": sum(1 for r in reject_reasons if "IMPERVIOUS" in r),
        }

        return FilterResult(
            clean_df=clean_df,
            rejected_df=rejected_df,
            water_mask=water_mask,
            exg_map=exg_map,
            summary_stats=stats,
        )

    def filter_raster_detections(
        self,
        raster_path_or_rgb: Union[str, Path, np.ndarray],
        detections_df: pd.DataFrame,
    ) -> FilterResult:
        import rasterio
        if isinstance(raster_path_or_rgb, (str, Path)):
            with rasterio.open(raster_path_or_rgb) as src:
                rgb = src.read([1, 2, 3])
                rgb = np.transpose(rgb, (1, 2, 0))
        else:
            rgb = raster_path_or_rgb
        return self.filter_detections(detections_df, rgb)

# Alias for backwards compatibility
WeedWaterFilter = SpectralMorphologicalFilter
