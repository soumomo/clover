"""
Test script for Scenario C: Plain non-georeferenced JPG/PNG uploads
(e.g., drone orthomosaic export or Google Earth screenshot).
"""
import os
import cv2
import numpy as np
import pandas as pd
import rasterio
from PIL import Image

from canopy_core.models.detector import TreeDetector
from canopy_core.models.area_engine import CanopyAreaEngine
from canopy_core.tiler import SlidingWindowTiler


def create_sample_jpg_screenshot(
    src_geotiff: str = "data/samples/kolkata_central_park.tif",
    dst_jpg: str = "data/samples/sample_drone_screenshot.jpg",
) -> str:
    """Extract raw RGB from GeoTIFF and save as a standard, non-georeferenced JPG."""
    with rasterio.open(src_geotiff) as src:
        rgb = src.read([1, 2, 3])
        rgb = np.transpose(rgb, (1, 2, 0))

    pil_img = Image.fromarray(rgb)
    pil_img.save(dst_jpg, quality=95)
    print(f"[CREATED] Plain non-georeferenced JPG: {dst_jpg} ({pil_img.size[0]}x{pil_img.size[1]} px)")
    return dst_jpg


def process_plain_image(
    image_path: str,
    assumed_gsd_m: float = 0.27,  # Default 27 cm/px for high-res drone / zoom 19
    score_threshold: float = 0.30,
) -> dict:
    """Process a non-georeferenced JPG or PNG image."""
    pil_img = Image.open(image_path).convert("RGB")
    img_np = np.array(pil_img)
    h, w, c = img_np.shape
    total_area_m2 = (w * assumed_gsd_m) * (h * assumed_gsd_m)
    total_area_ha = total_area_m2 / 10000.0

    print(f"\n--- Ingesting Plain Image: {os.path.basename(image_path)} ---")
    print(f"Dimensions: {w} x {h} pixels | Channels: {c}")
    print(f"Assumed GSD: {assumed_gsd_m * 100:.1f} cm/pixel")
    print(f"Calculated Ground Footprint: {total_area_m2:.1f} m² ({total_area_ha:.2f} ha)")

    detector = TreeDetector(score_threshold=score_threshold)
    tiler = SlidingWindowTiler(tile_size=400, overlap=0.20)

    tile_predictions = []
    for tile in tiler.iterate_array(img_np):
        tile_df = detector.predict_tile(tile.image, score_threshold=score_threshold)
        if len(tile_df) > 0:
            global_px_df = tiler.map_local_to_global_pixels(tile, tile_df)
            tile_predictions.append(global_px_df)

    if not tile_predictions:
        print("[WARNING] No trees detected.")
        return {"tree_count": 0}

    raw_df = pd.concat(tile_predictions, ignore_index=True)
    dedup_df = detector.apply_nms(raw_df, iou_threshold=0.25)
    print(f"[DETECTION] Detected {len(dedup_df)} candidate tree crowns after NMS.")

    dedup_df["center_x"] = ((dedup_df["xmin"] + dedup_df["xmax"]) / 2.0) * assumed_gsd_m
    dedup_df["center_y"] = ((dedup_df["ymin"] + dedup_df["ymax"]) / 2.0) * assumed_gsd_m
    dedup_df["width_m"] = (dedup_df["xmax"] - dedup_df["xmin"]) * assumed_gsd_m
    dedup_df["height_m"] = (dedup_df["ymax"] - dedup_df["ymin"]) * assumed_gsd_m
    dedup_df["crown_diameter_m"] = np.sqrt(dedup_df["width_m"] * dedup_df["height_m"])

    area_engine = CanopyAreaEngine()
    metrics, enriched_df, dissolved_poly = area_engine.compute_metrics(
        detections_df=dedup_df,
        aoi_area_m2=total_area_m2,
    )

    annotated = img_np.copy()
    for _, row in dedup_df.iterrows():
        x1, y1 = int(row["xmin"]), int(row["ymin"])
        x2, y2 = int(row["xmax"]), int(row["ymax"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.circle(annotated, (cx, cy), 3, (0, 255, 255), -1)

    hud_h = 70
    banner = np.zeros((hud_h, w, 3), dtype=np.uint8)
    banner[:] = (20, 28, 38)

    title_text = f"PLAIN JPG/PNG INGEST (Scenario C) | GSD: {assumed_gsd_m*100:.1f} cm/px | Ground Footprint: {total_area_ha:.2f} ha"
    uncertainty_mg = metrics.agb_total_mg * 0.565 / np.sqrt(max(1, metrics.tree_count))
    stats_text = (
        f"Stems: {metrics.tree_count} | Canopy Area: {metrics.dissolved_canopy_area_m2:.1f} m2 "
        f"({metrics.canopy_cover_pct_raw:.1f}%) | Calibrated: {metrics.canopy_cover_pct_calibrated:.1f}% | "
        f"AGB: {metrics.agb_total_mg:.2f} Mg (+/-{uncertainty_mg:.2f} Mg)"
    )

    cv2.putText(banner, title_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 230, 255), 1, cv2.LINE_AA)
    cv2.putText(banner, stats_text, (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 255, 180), 1, cv2.LINE_AA)

    final_img = np.vstack([banner, annotated])

    os.makedirs("reports", exist_ok=True)
    out_path = "reports/scenario_c_jpg_result.png"
    cv2.imwrite(out_path, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
    print(f"[SAVED] Annotated visual: {out_path}")

    print("\n--- Summary for Plain JPG/PNG Ingest ---")
    print(f"Stems Counted: {metrics.tree_count}")
    print(f"Mean Crown Diameter: {metrics.mean_crown_diameter_m:.2f} m")
    print(f"Dissolved Canopy Cover: {metrics.dissolved_canopy_area_m2:.1f} m² ({metrics.canopy_cover_pct_raw:.1f}%)")
    print(f"Calibrated Canopy Cover (Li et al. +20%): {metrics.total_calibrated_canopy_area_m2:.1f} m² ({metrics.canopy_cover_pct_calibrated:.1f}%)")
    print(f"Stand Biomass: {metrics.agb_total_mg:.2f} Mg AGB (±{uncertainty_mg:.2f} Mg)")
    print(f"Carbon Stock: {metrics.carbon_stock_mg_c:.2f} Mg C")

    return {
        "tree_count": metrics.tree_count,
        "metrics": metrics,
        "out_path": out_path,
    }


if __name__ == "__main__":
    jpg_path = create_sample_jpg_screenshot()
    process_plain_image(jpg_path, assumed_gsd_m=0.27)
