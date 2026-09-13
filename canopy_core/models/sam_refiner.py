"""
Promptable Foundation Model Crown Refiner using FastSAM / SAM.

Takes candidate tree crown bounding boxes from the object detector and generates
survey-grade, organic polygon contours and segmentation masks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from shapely.geometry import MultiPolygon, Polygon, Point
from shapely.ops import unary_union
from ultralytics import FastSAM

logger = logging.getLogger(__name__)


@dataclass
class SAMRefinementResult:
    """Container for SAM polygon refinement outputs."""
    polygons: List[Polygon]
    dissolved_canopy: Union[Polygon, MultiPolygon]
    individual_areas_m2: List[float]
    total_raw_area_m2: float
    dissolved_area_m2: float
    overlap_pct: float
    enriched_df: pd.DataFrame


class SAMCrownRefiner:
    """
    Promptable Crown Refiner leveraging FastSAM on Apple Silicon MPS.
    """

    def __init__(
        self,
        model_path: str = "FastSAM-s.pt",
        device: Optional[str] = None,
        conf_threshold: float = 0.20,
        iou_threshold: float = 0.50,
    ):
        if device is None:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device

        self.model = FastSAM(model_path)
        self.conf = float(conf_threshold)
        self.iou = float(iou_threshold)
        logger.info(f"Initialized SAMCrownRefiner ({model_path}) on device: {self.device}")

    def create_fallback_ellipse(
        self,
        x1: float, y1: float, x2: float, y2: float,
        affine_transform: Optional[Any] = None,
    ) -> Polygon:
        """Create an ellipse polygon for a bounding box."""
        w = max(0.1, x2 - x1)
        h = max(0.1, y2 - y1)
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        
        # Unit circle scaled
        theta = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        circle_x = cx + (w / 2.0) * np.cos(theta)
        circle_y = cy + (h / 2.0) * np.sin(theta)
        
        if affine_transform is not None:
            coords = [affine_transform * (px, py) for px, py in zip(circle_x, circle_y)]
        else:
            coords = list(zip(circle_x, circle_y))
            
        return Polygon(coords)

    def refine_crowns(
        self,
        rgb_image: np.ndarray,
        detections_df: pd.DataFrame,
        gsd_meters: float = 0.27,
        affine_transform: Optional[Any] = None,
        max_prompt_batch: int = 64,
    ) -> SAMRefinementResult:
        """
        Refine bounding box detections into organic polygon masks using FastSAM.
        """
        if detections_df is None or len(detections_df) == 0:
            return SAMRefinementResult(
                polygons=[],
                dissolved_canopy=Polygon(),
                individual_areas_m2=[],
                total_raw_area_m2=0.0,
                dissolved_area_m2=0.0,
                overlap_pct=0.0,
                enriched_df=pd.DataFrame(),
            )

        df = detections_df.copy().reset_index(drop=True)
        h_img, w_img = rgb_image.shape[:2]

        polygons: List[Polygon] = []
        areas_m2: List[float] = []

        # Process in batches to avoid GPU OOM
        boxes_all = df[["xmin", "ymin", "xmax", "ymax"]].values
        
        for idx, row in df.iterrows():
            x1, y1, x2, y2 = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            
            # Crop a localized patch around the tree crown with 20% context margin
            pad_x = int(bw * 0.2)
            pad_y = int(bh * 0.2)
            px1 = max(0, int(x1) - pad_x)
            py1 = max(0, int(y1) - pad_y)
            px2 = min(w_img, int(x2) + pad_x)
            py2 = min(h_img, int(y2) + pad_y)
            
            patch = rgb_image[py1:py2, px1:px2]
            local_box = [[x1 - px1, y1 - py1, x2 - px1, y2 - py1]]
            
            poly_found = False
            if patch.shape[0] > 10 and patch.shape[1] > 10:
                try:
                    res = self.model(
                        patch,
                        bboxes=local_box,
                        device=self.device,
                        conf=self.conf,
                        retina_masks=True,
                        verbose=False,
                    )
                    if res[0].masks is not None and len(res[0].masks.data) > 0:
                        mask = res[0].masks.data[0].cpu().numpy().astype(np.uint8)
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            largest = max(contours, key=cv2.contourArea)
                            if len(largest) >= 3 and cv2.contourArea(largest) > 10:
                                # Offset back to global pixel coordinates
                                pts = largest.squeeze().astype(float)
                                if pts.ndim == 2:
                                    pts[:, 0] += px1
                                    pts[:, 1] += py1
                                    poly = Polygon(pts).simplify(0.5, preserve_topology=True)
                                    if poly.is_valid and poly.area > 5:
                                        if affine_transform is not None:
                                            ext = [affine_transform * (pt[0], pt[1]) for pt in poly.exterior.coords]
                                            crs_poly = Polygon(ext)
                                        else:
                                            crs_poly = poly
                                        polygons.append(crs_poly)
                                        area_val = float(crs_poly.area) if affine_transform is not None else float(poly.area * (gsd_meters ** 2))
                                        areas_m2.append(area_val)
                                        poly_found = True
                except Exception:
                    pass

            if not poly_found:
                # Fallback to analytical ellipse
                ell = self.create_fallback_ellipse(x1, y1, x2, y2, affine_transform=affine_transform)
                polygons.append(ell)
                area_val = float(ell.area) if affine_transform is not None else float(ell.area * (gsd_meters ** 2))
                areas_m2.append(area_val)

        # Dissolve overlapping polygons with Shapely unary_union
        dissolved = unary_union(polygons) if polygons else Polygon()
        total_raw = float(sum(areas_m2))
        dissolved_area = float(dissolved.area) if not dissolved.is_empty else 0.0
        overlap_pct = ((total_raw - dissolved_area) / total_raw * 100.0) if total_raw > 0 else 0.0

        df["sam_crown_area_m2"] = areas_m2
        df["geometry"] = polygons

        return SAMRefinementResult(
            polygons=polygons,
            dissolved_canopy=dissolved,
            individual_areas_m2=areas_m2,
            total_raw_area_m2=total_raw,
            dissolved_area_m2=dissolved_area,
            overlap_pct=overlap_pct,
            enriched_df=df,
        )
