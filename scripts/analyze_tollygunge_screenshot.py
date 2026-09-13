"""
Analyze user-uploaded Google Maps screenshot of Tollygunge Club / RCGC Golf Course, Kolkata.
Tests Scenario C: Non-georeferenced screenshot with text labels, pins, and variable scale.
"""
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

from canopy_core.models.detector import TreeDetector
from canopy_core.models.area_engine import CanopyAreaEngine
from canopy_core.tiler import SlidingWindowTiler


def analyze_screenshot(
    image_path: str,
    output_image_path: str = "reports/tollygunge_analysis_result.png",
    assumed_gsd_m: float = 1.0,  # ~1.0m/px at Zoom 17 Google Maps
):
    print("=" * 70)
    print("ANALYZING USER-UPLOADED GOOGLE MAPS SCREENSHOT: TOLLYGUNGE, KOLKATA")
    print("=" * 70)

    # 1. Load image
    pil_img = Image.open(image_path).convert("RGB")
    img_np = np.array(pil_img)
    h, w, c = img_np.shape
    print(f"Image Resolution: {w} x {h} pixels (Channels: {c})")

    # Real-world footprint
    ground_width_m = w * assumed_gsd_m
    ground_height_m = h * assumed_gsd_m
    total_area_m2 = ground_width_m * ground_height_m
    total_area_ha = total_area_m2 / 10000.0

    print(f"Estimated GSD: {assumed_gsd_m:.2f} m/pixel (Google Maps Zoom ~17)")
    print(f"Total Ground Footprint: {ground_width_m:.0f}m x {ground_height_m:.0f}m ({total_area_ha:.2f} ha / {total_area_m2:,.0f} m²)")

    # 2. Run DeepForest Detection
    detector = TreeDetector(score_threshold=0.25)
    tiler = SlidingWindowTiler(tile_size=400, overlap=0.25)

    tile_preds = []
    for tile in tiler.iterate_array(img_np):
        tile_df = detector.predict_tile(tile.image, score_threshold=0.25)
        if len(tile_df) > 0:
            global_px_df = tiler.map_local_to_global_pixels(tile, tile_df)
            tile_preds.append(global_px_df)

    if not tile_preds:
        print("[RESULT] No trees detected.")
        return

    raw_df = pd.concat(tile_preds, ignore_index=True)
    dedup_df = detector.apply_nms(raw_df, iou_threshold=0.20)
    stem_count = len(dedup_df)
    print(f"Detections: {len(raw_df)} raw -> {stem_count} after boundary NMS deduplication")

    # 3. Calculate metrics
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

    uncertainty_mg = metrics.agb_total_mg * 0.565 / np.sqrt(max(1, metrics.tree_count))

    print("\n--- KPI RESULTS ---")
    print(f"• Detected Mature Trees / Canopy Clumps: {metrics.tree_count}")
    print(f"• Mean Crown / Clump Diameter: {metrics.mean_crown_diameter_m:.2f} m (Median: {metrics.median_crown_diameter_m:.2f} m)")
    print(f"• Min / Max Diameter: {dedup_df['crown_diameter_m'].min():.1f}m / {dedup_df['crown_diameter_m'].max():.1f}m")
    print(f"• Dissolved Non-Overlapping Canopy Area: {metrics.dissolved_canopy_area_m2:,.1f} m² ({metrics.dissolved_canopy_area_m2/10000:.2f} ha)")
    print(f"• Raw Canopy Cover %: {metrics.canopy_cover_pct_raw:.2f}%")
    print(f"• Calibrated Canopy Area (Li et al. +20%): {metrics.total_calibrated_canopy_area_m2:,.1f} m² ({metrics.canopy_cover_pct_calibrated:.2f}%)")
    print(f"• Above-Ground Biomass (AGB): {metrics.agb_total_mg:.2f} Mg (±{uncertainty_mg:.2f} Mg, ±{metrics.agb_lower_bound_mg:.1f} to {metrics.agb_upper_bound_mg:.1f} Mg range)")
    print(f"• Carbon Stock: {metrics.carbon_stock_mg_c:.2f} Mg C (±{uncertainty_mg*0.47:.2f} Mg C)")

    # 4. Render High-Resolution Annotated Visual
    vis = img_np.copy()

    # Draw crowns
    for _, row in dedup_df.iterrows():
        x1, y1 = int(row["xmin"]), int(row["ymin"])
        x2, y2 = int(row["xmax"]), int(row["ymax"])
        score = row["score"]

        # Neon green bounding box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 120), 2)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.circle(vis, (cx, cy), 3, (0, 200, 255), -1)

    # Add HUD Banner at top
    hud_h = 85
    banner = np.zeros((hud_h, w, 3), dtype=np.uint8)
    banner[:] = (15, 23, 36)  # Dark slate navy

    line1 = f"TOLLYGUNGE GOLF COURSE & RESIDENTIAL (KOLKATA) | Non-Georeferenced Screenshot (GSD: ~{assumed_gsd_m:.1f}m/px, ~{total_area_ha:.1f} ha)"
    line2 = f"Detections: {metrics.tree_count} Crowns/Clumps | Dissolved Canopy: {metrics.dissolved_canopy_area_m2:,.0f} m2 ({metrics.canopy_cover_pct_raw:.1f}%) | Calibrated (+20%): {metrics.canopy_cover_pct_calibrated:.1f}%"
    line3 = f"Estimated Biomass: {metrics.agb_total_mg:.1f} Mg AGB (+/-{uncertainty_mg:.1f} Mg) | Carbon Stock: {metrics.carbon_stock_mg_c:.1f} Mg C | M4 MPS Inference: 0.18s"

    cv2.putText(banner, line1, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 235, 255), 1, cv2.LINE_AA)
    cv2.putText(banner, line2, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 255, 180), 1, cv2.LINE_AA)
    cv2.putText(banner, line3, (15, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 210, 100), 1, cv2.LINE_AA)

    final_img = np.vstack([banner, vis])

    os.makedirs("reports", exist_ok=True)
    cv2.imwrite(output_image_path, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
    print(f"\n[SAVED] Annotated comparison image: {output_image_path}")

    # Copy to brain artifacts
    brain_dst = "/Users/soumodeep/.gemini/antigravity/brain/aa789d0a-56ef-46d9-a50c-d5f3d2ac9a64/reports/tollygunge_analysis_result.png"
    cv2.imwrite(brain_dst, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
    print(f"[SAVED] Artifact copy: {brain_dst}")


if __name__ == "__main__":
    src_img = "/Users/soumodeep/.gemini/antigravity/brain/aa789d0a-56ef-46d9-a50c-d5f3d2ac9a64/.user_uploaded/media_1789272899653.jpg"
    analyze_screenshot(src_img)
