"""Visualization utilities for tree crown detection and canopy cover overlays."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon
import rasterio

from canopy_core.models.area_engine import CanopyMetrics


def visualize_predictions(
    image: np.ndarray,
    detections_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
    show_labels: bool = True,
    box_color: str = "#00FF66",
    box_thickness: int = 2,
) -> np.ndarray:
    """Plot detection boxes and crown ellipses onto an RGB image.

    Args:
        image: RGB NumPy array of shape (H, W, 3).
        detections_df: DataFrame with ['xmin', 'ymin', 'xmax', 'ymax', 'score'].
        output_path: Optional path to save annotated image.
        show_labels: Whether to annotate confidence scores.
        box_color: Hex color for box strokes.
        box_thickness: Line thickness.

    Returns:
        np.ndarray: Annotated image array.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=150)
    ax.imshow(image)

    for _, row in detections_df.iterrows():
        xmin = row["xmin"]
        ymin = row["ymin"]
        xmax = row["xmax"]
        ymax = row["ymax"]
        w = xmax - xmin
        h = ymax - ymin

        # Draw bounding box
        rect = patches.Rectangle(
            (xmin, ymin),
            w,
            h,
            linewidth=box_thickness,
            edgecolor=box_color,
            facecolor="none",
            alpha=0.85,
        )
        ax.add_patch(rect)

        # Draw inscribed ellipse
        ellipse = patches.Ellipse(
            (xmin + w / 2.0, ymin + h / 2.0),
            w,
            h,
            linewidth=1.5,
            edgecolor="#FFCC00",
            facecolor="none",
            linestyle="--",
            alpha=0.9,
        )
        ax.add_patch(ellipse)

        if show_labels and "score" in row:
            score = row["score"]
            diam_str = f" | {row['crown_diameter_m']:.1f}m" if "crown_diameter_m" in row else ""
            label_txt = f"{score:.2f}{diam_str}"
            ax.text(
                xmin,
                max(0, ymin - 3),
                label_txt,
                color="black",
                fontsize=7,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#00FF66", alpha=0.8, edgecolor="none"),
            )

    ax.set_axis_off()
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight", dpi=150)

    plt.close(fig)
    return image


def save_canopy_report_plot(
    image: np.ndarray,
    detections_df: pd.DataFrame,
    metrics: CanopyMetrics,
    dissolved_geometry: Optional[Union[Polygon, MultiPolygon]] = None,
    raster_transform: Optional[rasterio.Affine] = None,
    output_path: str = "output_prediction.png",
) -> str:
    """Generate a dual-panel analysis dashboard comparing detected tree crowns

    with the dissolved canopy cover footprint and KPI summary banner.

    Args:
        image: RGB image array (H, W, 3).
        detections_df: DataFrame with detected trees.
        metrics: CanopyMetrics object from CanopyAreaEngine.
        dissolved_geometry: Shapely geometry of dissolved canopy footprint in CRS coords.
        raster_transform: Raster Affine transform for converting geometry to pixel coordinates.
        output_path: Path to save the PNG report plot.

    Returns:
        str: Absolute path to the saved figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), dpi=150)

    # Panel 1: Detections overlaid on RGB
    ax1.imshow(image)
    ax1.set_title(
        f"DeepForest Tree Crown Delineation (Stems: {metrics.tree_count})",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )

    for _, row in detections_df.iterrows():
        xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
        w = xmax - xmin
        h = ymax - ymin

        rect = patches.Rectangle(
            (xmin, ymin),
            w,
            h,
            linewidth=1.2,
            edgecolor="#00FF66",
            facecolor="none",
            alpha=0.75,
        )
        ax1.add_patch(rect)

        ellipse = patches.Ellipse(
            (xmin + w / 2.0, ymin + h / 2.0),
            w,
            h,
            linewidth=1.2,
            edgecolor="#FFCC00",
            facecolor="none",
            linestyle="--",
            alpha=0.85,
        )
        ax1.add_patch(ellipse)

    ax1.set_axis_off()

    # Panel 2: Dissolved Canopy Footprint Mask Overlay
    ax2.imshow(image)
    ax2.set_title(
        f"Dissolved Canopy Cover (Raw: {metrics.canopy_cover_pct_raw:.1f}% | Calibrated: {metrics.canopy_cover_pct_calibrated:.1f}%)",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )

    # Render dissolved polygons if transform is available
    if dissolved_geometry is not None and raster_transform is not None and not dissolved_geometry.is_empty:
        inv_transform = ~raster_transform

        geoms = (
            dissolved_geometry.geoms
            if isinstance(dissolved_geometry, MultiPolygon)
            else [dissolved_geometry]
        )

        for poly in geoms:
            if poly.is_empty:
                continue
            ext_coords = np.array(poly.exterior.coords)
            # Map CRS coordinates to pixel coordinates
            px_coords = []
            for gx, gy in ext_coords:
                col, row = inv_transform * (gx, gy)
                px_coords.append([col, row])

            px_arr = np.array(px_coords)
            polygon_patch = patches.Polygon(
                px_arr,
                closed=True,
                facecolor="#00FF88",
                edgecolor="#00AA44",
                alpha=0.45,
                linewidth=1.5,
            )
            ax2.add_patch(polygon_patch)

    ax2.set_axis_off()

    # Summary KPI banner at the bottom
    kpi_text = (
        f"STAND SUMMARY:  "
        f"Stems: {metrics.tree_count}  |  "
        f"Density: {metrics.tree_density_per_ha:.0f} stems/ha  |  "
        f"Mean Crown: {metrics.mean_crown_diameter_m:.2f}m  |  "
        f"Canopy Cover: {metrics.canopy_cover_pct_raw:.1f}% (Li et al. +20%: {metrics.canopy_cover_pct_calibrated:.1f}%)  |  "
        f"AGB: {metrics.agb_total_mg:.2f} Mg [{metrics.agb_lower_bound_mg:.2f}, {metrics.agb_upper_bound_mg:.2f} Mg (±35%)]"
    )
    plt.figtext(
        0.5,
        0.03,
        kpi_text,
        wrap=True,
        horizontalalignment="center",
        fontsize=10,
        fontweight="semibold",
        bbox=dict(boxstyle="square,pad=0.5", facecolor="#F0F4F8", edgecolor="#B0C4DE"),
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    abs_output = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(abs_output), exist_ok=True)
    plt.savefig(abs_output, bbox_inches="tight", dpi=150)
    plt.close(fig)

    return abs_output
