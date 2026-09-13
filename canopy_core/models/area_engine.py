"""Canopy area calculation engine: ellipse shape factor, Shapely unary_union dissolve,

canopy cover percentage, Li et al. (2023) bias calibration, and Jucker et al. (2016) AGB allometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.affinity import scale
from shapely.geometry import MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union

from canopy_core.ingest import RasterMetadata


@dataclass
class CanopyMetrics:
    """Comprehensive canopy metrics and allometric Above-Ground Biomass (AGB) summary."""

    tree_count: int
    aoi_area_m2: float
    aoi_area_ha: float
    tree_density_per_ha: float

    # Crown Dimensions
    mean_crown_diameter_m: float
    std_crown_diameter_m: float
    median_crown_diameter_m: float
    mean_crown_area_m2: float

    # Canopy Area & Coverage
    total_raw_crown_area_m2: float
    dissolved_canopy_area_m2: float
    crown_overlap_area_m2: float
    crown_overlap_pct: float
    canopy_cover_pct_raw: float

    # Li et al. (PNAS Nexus 2023) +20% Calibration
    canopy_cover_pct_calibrated: float
    total_calibrated_canopy_area_m2: float

    # Jucker et al. (2016) Allometric AGB Estimates
    agb_total_kg: float
    agb_total_mg: float  # Metric tonnes
    agb_lower_bound_mg: float  # -35% confidence bound
    agb_upper_bound_mg: float  # +35% confidence bound
    agb_density_mg_per_ha: float

    # Carbon Stock (IPCC 47% carbon fraction)
    carbon_stock_mg_c: float
    carbon_lower_bound_mg_c: float
    carbon_upper_bound_mg_c: float

    citations: Dict[str, str] = field(
        default_factory=lambda: {
            "shape_factor": "pi/4 (0.7854) circumscribed ellipse crown model",
            "dissolve": "Shapely unary_union for non-overlapping spatial canopy cover",
            "calibration": "Li et al. (PNAS Nexus 2023) +20% crown underestimation bias calibration",
            "allometry": "Jucker et al. (2016/2017) GCB RS allometric equation ln(AGB) = -0.328 + 2.404*ln(CD) with +/-35% confidence bounds",
        }
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a structured dictionary."""
        return {
            "tree_count": self.tree_count,
            "aoi_area_m2": round(self.aoi_area_m2, 2),
            "aoi_area_ha": round(self.aoi_area_ha, 4),
            "tree_density_per_ha": round(self.tree_density_per_ha, 1),
            "mean_crown_diameter_m": round(self.mean_crown_diameter_m, 2),
            "std_crown_diameter_m": round(self.std_crown_diameter_m, 2),
            "median_crown_diameter_m": round(self.median_crown_diameter_m, 2),
            "mean_crown_area_m2": round(self.mean_crown_area_m2, 2),
            "total_raw_crown_area_m2": round(self.total_raw_crown_area_m2, 2),
            "dissolved_canopy_area_m2": round(self.dissolved_canopy_area_m2, 2),
            "crown_overlap_area_m2": round(self.crown_overlap_area_m2, 2),
            "crown_overlap_pct": round(self.crown_overlap_pct, 2),
            "canopy_cover_pct_raw": round(self.canopy_cover_pct_raw, 2),
            "canopy_cover_pct_calibrated": round(self.canopy_cover_pct_calibrated, 2),
            "total_calibrated_canopy_area_m2": round(self.total_calibrated_canopy_area_m2, 2),
            "agb_total_mg": round(self.agb_total_mg, 2),
            "agb_bounds_mg": [round(self.agb_lower_bound_mg, 2), round(self.agb_upper_bound_mg, 2)],
            "agb_density_mg_per_ha": round(self.agb_density_mg_per_ha, 2),
            "carbon_stock_mg_c": round(self.carbon_stock_mg_c, 2),
            "carbon_bounds_mg_c": [round(self.carbon_lower_bound_mg_c, 2), round(self.carbon_upper_bound_mg_c, 2)],
            "citations": self.citations,
        }

    def format_table(self) -> str:
        """Format the KPI summary into a readable markdown table."""
        lines = [
            "| Metric | Value | Unit / Reference |",
            "| :--- | :--- | :--- |",
            f"| **Tree Crown Count** | {self.tree_count:,} | stems |",
            f"| **AOI Total Area** | {self.aoi_area_m2:,.1f} | m² ({self.aoi_area_ha:.3f} ha) |",
            f"| **Stand Tree Density** | {self.tree_density_per_ha:,.1f} | stems / hectare |",
            f"| **Mean Crown Diameter** | {self.mean_crown_diameter_m:.2f} ± {self.std_crown_diameter_m:.2f} | meters |",
            f"| **Median Crown Diameter** | {self.median_crown_diameter_m:.2f} | meters |",
            f"| **Mean Crown Area (Ellipse)** | {self.mean_crown_area_m2:.2f} | m² (Shape factor π/4 = 0.7854) |",
            f"| **Total Raw Crown Area (Sum)** | {self.total_raw_crown_area_m2:,.2f} | m² |",
            f"| **Dissolved Canopy Area (Union)** | {self.dissolved_canopy_area_m2:,.2f} | m² (Shapely unary_union) |",
            f"| **Crown Overlap Fraction** | {self.crown_overlap_pct:.2f}% | ({self.crown_overlap_area_m2:,.1f} m² overlap) |",
            f"| **Canopy Cover % (Raw)** | **{self.canopy_cover_pct_raw:.2f}%** | Area(Union) / Area(AOI) |",
            f"| **Canopy Cover % (Calibrated)** | **{self.canopy_cover_pct_calibrated:.2f}%** | **Li et al. (PNAS Nexus 2023) +20%** |",
            f"| **Calibrated Canopy Area** | {self.total_calibrated_canopy_area_m2:,.2f} | m² |",
            f"| **Above-Ground Biomass (AGB)** | **{self.agb_total_mg:.2f} Mg** | **Jucker et al. (2016)** |",
            f"| **AGB 95% Confidence Bounds** | **[{self.agb_lower_bound_mg:.2f}, {self.agb_upper_bound_mg:.2f}] Mg** | **Explicit ±35% confidence bounds** |",
            f"| **AGB Stand Density** | {self.agb_density_mg_per_ha:.2f} | Mg / hectare |",
            f"| **Estimated Carbon Stock** | {self.carbon_stock_mg_c:.2f} Mg C | [{self.carbon_lower_bound_mg_c:.2f}, {self.carbon_upper_bound_mg_c:.2f}] Mg C (47% C) |",
        ]
        return "\n".join(lines)


class CanopyAreaEngine:
    """Engine for computing crown areas, dissolved non-overlapping canopy cover,

    Li et al. (2023) bias calibration, and Jucker et al. (2016) AGB allometric proxy.
    """

    ELLIPSE_FACTOR = math.pi / 4.0  # 0.7853981633974483
    LI_CALIBRATION_FACTOR = 1.20  # +20% empirical crown area bias correction
    AGB_UNCERTAINTY = 0.35  # ±35% allometric confidence envelope

    # Jucker et al. (2017) Remote Sensing Allometric Parameters:
    # ln(AGB_kg) = alpha + beta * ln(CD_m)
    ALLOMETRY_PARAMS = {
        "angiosperm": {"alpha": -0.328, "beta": 2.404},
        "gymnosperm": {"alpha": -0.714, "beta": 2.658},
    }

    def __init__(
        self,
        allometry_type: str = "angiosperm",
        custom_allometry: Optional[Tuple[float, float]] = None,
        buffer_resolution: int = 16,
    ):
        """Initialize the Canopy Area Engine.

        Args:
            allometry_type: Forest type ('angiosperm' or 'gymnosperm').
            custom_allometry: Optional custom (alpha, beta) tuple for ln(AGB) = alpha + beta * ln(CD).
            buffer_resolution: Number of vertices per circle quadrant for ellipse generation (default 16).
        """
        self.allometry_type = allometry_type.lower()
        if custom_allometry is not None:
            self.alpha, self.beta = custom_allometry
        else:
            params = self.ALLOMETRY_PARAMS.get(self.allometry_type, self.ALLOMETRY_PARAMS["angiosperm"])
            self.alpha = params["alpha"]
            self.beta = params["beta"]

        self.buffer_resolution = buffer_resolution

    def create_crown_ellipses(
        self,
        detections_df: pd.DataFrame,
    ) -> List[Polygon]:
        """Generate Shapely ellipse polygons for detected tree crowns in metric CRS coordinates.

        The ellipse is centered at (center_x, center_y) with semi-axes (width_m / 2, height_m / 2).
        Its area closely matches the analytical area: pi/4 * width_m * height_m.

        Args:
            detections_df: DataFrame with ['center_x', 'center_y', 'width_m', 'height_m'].

        Returns:
            list[Polygon]: List of Shapely ellipse polygons.
        """
        if detections_df is None or len(detections_df) == 0:
            return []

        ellipses: List[Polygon] = []
        unit_circle = Point(0, 0).buffer(1.0, resolution=self.buffer_resolution)

        for _, row in detections_df.iterrows():
            cx = float(row["center_x"])
            cy = float(row["center_y"])
            rx = max(1e-4, float(row["width_m"]) / 2.0)
            ry = max(1e-4, float(row["height_m"]) / 2.0)

            # Scale and translate unit circle to crown ellipse
            ellipse_poly = scale(unit_circle, xfact=rx, yfact=ry, origin=(0, 0))
            ellipse_poly = shapely.affinity.translate(ellipse_poly, xoff=cx, yoff=cy)
            ellipses.append(ellipse_poly)

        return ellipses

    def calculate_tree_biomass(self, crown_diameters_m: Union[np.ndarray, pd.Series, List[float]]) -> np.ndarray:
        """Calculate individual tree Above-Ground Biomass (AGB in kg)

        using Jucker et al. (2016/2017) allometric formula:
        AGB = exp(alpha + beta * ln(CD))

        Args:
            crown_diameters_m: Crown diameters in meters.

        Returns:
            np.ndarray: Estimated tree biomass in kilograms.
        """
        cd_arr = np.asarray(crown_diameters_m, dtype=np.float64)
        cd_clipped = np.maximum(0.1, cd_arr)  # Prevent log(0)
        agb_kg = np.exp(self.alpha + self.beta * np.log(cd_clipped))
        return agb_kg

    def compute_metrics(
        self,
        detections_df: pd.DataFrame,
        raster_metadata: Optional[RasterMetadata] = None,
        aoi_polygon: Optional[Union[Polygon, MultiPolygon, gpd.GeoDataFrame]] = None,
        aoi_area_m2: Optional[float] = None,
    ) -> Tuple[CanopyMetrics, pd.DataFrame, Union[Polygon, MultiPolygon]]:
        """Compute full canopy metrics, dissolved coverage %, Li et al. calibrated canopy area,

        and Jucker et al. allometric AGB with ±35% bounds.

        Args:
            detections_df: DataFrame with detected tree bounding boxes in projected metric CRS.
            raster_metadata: Optional RasterMetadata to determine AOI bounds if aoi_polygon is None.
            aoi_polygon: Optional Shapely geometry or GeoDataFrame defining the AOI footprint.
            aoi_area_m2: Optional explicit AOI area in square meters.

        Returns:
            tuple: (CanopyMetrics dataclass, enriched detections_df, dissolved canopy union geometry)
        """
        df = detections_df.copy() if detections_df is not None else pd.DataFrame()

        # 1. Determine AOI geometry and Area
        aoi_geom: Optional[Union[Polygon, MultiPolygon]] = None
        if aoi_polygon is not None:
            if isinstance(aoi_polygon, gpd.GeoDataFrame):
                aoi_geom = unary_union(aoi_polygon.geometry)
            else:
                aoi_geom = aoi_polygon
            calculated_aoi_area = float(aoi_geom.area)
        elif aoi_area_m2 is not None:
            calculated_aoi_area = float(aoi_area_m2)
        elif raster_metadata is not None:
            # Full raster bounds
            b = raster_metadata.bounds
            aoi_geom = box(b[0], b[1], b[2], b[3])
            calculated_aoi_area = float(abs((b[2] - b[0]) * (b[3] - b[1])))
        else:
            # Approximate AOI from detections bounding box
            if len(df) > 0:
                minx, miny = df["geo_xmin"].min(), df["geo_ymin"].min()
                maxx, maxy = df["geo_xmax"].max(), df["geo_ymax"].max()
                aoi_geom = box(minx, miny, maxx, maxy)
                calculated_aoi_area = float(aoi_geom.area)
            else:
                calculated_aoi_area = 1.0

        if calculated_aoi_area <= 0:
            calculated_aoi_area = 1.0

        aoi_area_ha = calculated_aoi_area / 10000.0

        # Handle zero detections
        if len(df) == 0:
            empty_metrics = CanopyMetrics(
                tree_count=0,
                aoi_area_m2=calculated_aoi_area,
                aoi_area_ha=aoi_area_ha,
                tree_density_per_ha=0.0,
                mean_crown_diameter_m=0.0,
                std_crown_diameter_m=0.0,
                median_crown_diameter_m=0.0,
                mean_crown_area_m2=0.0,
                total_raw_crown_area_m2=0.0,
                dissolved_canopy_area_m2=0.0,
                crown_overlap_area_m2=0.0,
                crown_overlap_pct=0.0,
                canopy_cover_pct_raw=0.0,
                canopy_cover_pct_calibrated=0.0,
                total_calibrated_canopy_area_m2=0.0,
                agb_total_kg=0.0,
                agb_total_mg=0.0,
                agb_lower_bound_mg=0.0,
                agb_upper_bound_mg=0.0,
                agb_density_mg_per_ha=0.0,
                carbon_stock_mg_c=0.0,
                carbon_lower_bound_mg_c=0.0,
                carbon_upper_bound_mg_c=0.0,
            )
            return empty_metrics, df, Polygon()

        # 2. Individual Tree Ellipse Crown Area (pi/4 * width * height)
        df["crown_area_raw_m2"] = self.ELLIPSE_FACTOR * df["width_m"] * df["height_m"]
        df["crown_area_calibrated_m2"] = df["crown_area_raw_m2"] * self.LI_CALIBRATION_FACTOR

        # Crown Diameter
        if "crown_diameter_m" not in df.columns:
            df["crown_diameter_m"] = np.sqrt(np.maximum(1e-6, df["width_m"] * df["height_m"]))

        # 3. Individual Tree Biomass (Jucker et al. 2016)
        df["agb_tree_kg"] = self.calculate_tree_biomass(df["crown_diameter_m"].values)

        # 4. Construct Ellipse Geometries and Dissolve with Shapely unary_union
        crown_polygons = self.create_crown_ellipses(df)
        canopy_union = unary_union(crown_polygons)

        # If AOI polygon exists, clip dissolved canopy to AOI boundaries
        if aoi_geom is not None:
            canopy_in_aoi = canopy_union.intersection(aoi_geom)
        else:
            canopy_in_aoi = canopy_union

        dissolved_canopy_area = float(canopy_in_aoi.area)
        total_raw_crown_area = float(df["crown_area_raw_m2"].sum())

        crown_overlap_area = max(0.0, total_raw_crown_area - dissolved_canopy_area)
        crown_overlap_pct = (crown_overlap_area / total_raw_crown_area * 100.0) if total_raw_crown_area > 0 else 0.0

        # Non-overlapping Canopy Cover %
        canopy_cover_pct_raw = (dissolved_canopy_area / calculated_aoi_area) * 100.0
        canopy_cover_pct_raw = min(100.0, max(0.0, canopy_cover_pct_raw))

        # Li et al. (PNAS Nexus 2023) +20% Crown Area Bias Calibration
        canopy_cover_pct_calibrated = min(100.0, canopy_cover_pct_raw * self.LI_CALIBRATION_FACTOR)
        total_calibrated_canopy_area = min(calculated_aoi_area, dissolved_canopy_area * self.LI_CALIBRATION_FACTOR)

        # 5. Stand Biomass Aggregations and Confidence Bounds
        agb_total_kg = float(df["agb_tree_kg"].sum())
        agb_total_mg = agb_total_kg / 1000.0  # Convert kg to metric tons (Mg)

        # Explicit ±35% Confidence Bounds
        agb_lower_bound_mg = agb_total_mg * (1.0 - self.AGB_UNCERTAINTY)
        agb_upper_bound_mg = agb_total_mg * (1.0 + self.AGB_UNCERTAINTY)

        agb_density_mg_per_ha = agb_total_mg / aoi_area_ha if aoi_area_ha > 0 else 0.0

        # Carbon stock (47% of dry biomass)
        carbon_stock_mg_c = agb_total_mg * 0.47
        carbon_lower_bound_mg_c = agb_lower_bound_mg * 0.47
        carbon_upper_bound_mg_c = agb_upper_bound_mg * 0.47

        tree_count = len(df)
        tree_density = tree_count / aoi_area_ha if aoi_area_ha > 0 else 0.0

        metrics = CanopyMetrics(
            tree_count=tree_count,
            aoi_area_m2=calculated_aoi_area,
            aoi_area_ha=aoi_area_ha,
            tree_density_per_ha=tree_density,
            mean_crown_diameter_m=float(df["crown_diameter_m"].mean()),
            std_crown_diameter_m=float(df["crown_diameter_m"].std() if len(df) > 1 else 0.0),
            median_crown_diameter_m=float(df["crown_diameter_m"].median()),
            mean_crown_area_m2=float(df["crown_area_raw_m2"].mean()),
            total_raw_crown_area_m2=total_raw_crown_area,
            dissolved_canopy_area_m2=dissolved_canopy_area,
            crown_overlap_area_m2=crown_overlap_area,
            crown_overlap_pct=crown_overlap_pct,
            canopy_cover_pct_raw=canopy_cover_pct_raw,
            canopy_cover_pct_calibrated=canopy_cover_pct_calibrated,
            total_calibrated_canopy_area_m2=total_calibrated_canopy_area,
            agb_total_kg=agb_total_kg,
            agb_total_mg=agb_total_mg,
            agb_lower_bound_mg=agb_lower_bound_mg,
            agb_upper_bound_mg=agb_upper_bound_mg,
            agb_density_mg_per_ha=agb_density_mg_per_ha,
            carbon_stock_mg_c=carbon_stock_mg_c,
            carbon_lower_bound_mg_c=carbon_lower_bound_mg_c,
            carbon_upper_bound_mg_c=carbon_upper_bound_mg_c,
        )

        return metrics, df, canopy_in_aoi
