"""Comprehensive end-to-end testing script for Canopy Core pipeline.

Validates:
1. Raster metadata inspection and GSD validation across all literature tiers.
2. Coordinate reprojection from geographic EPSG:4326 to metric UTM (e.g. EPSG:32645 for Kolkata, EPSG:32617 for US).
3. GeoJSON and KML AOI polygon parsing and raster clipping.
4. Sliding-window tiling with overlap and coordinate back-mapping.
5. DeepForest RetinaNet tree crown inference on Apple Silicon MPS / CPU.
6. Boundary Non-Maximum Suppression (NMS) deduplication across overlapping tiles.
7. CanopyAreaEngine metrics:
   - pi/4 ellipse shape factor
   - Shapely unary_union dissolve
   - Non-overlapping Canopy Cover %
   - Li et al. (PNAS Nexus 2023) +20% crown area calibration
   - Jucker et al. (2016) allometric Above-Ground Biomass (AGB) with explicit ±35% bounds
8. Dual-panel visual prediction plot saved to output_prediction.png and formatted KPI table.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box
import torch

from deepforest import get_data
from canopy_core import (
    CanopyAreaEngine,
    CanopyMetrics,
    RasterMetadata,
    SlidingWindowTiler,
    TreeDetector,
    calculate_gsd,
    clip_raster_to_aoi,
    get_utm_epsg_for_coords,
    inspect_raster,
    parse_aoi,
    reproject_raster,
    validate_gsd,
)
from canopy_core.utils.visualization import save_canopy_report_plot


def test_gsd_validation():
    """Verify GSD calculation and validation thresholds across literature tiers."""
    print("\n--- 1. Testing GSD Calculation & Tier Validation ---")

    # Tier 1: < 0.15m (Optimal UAV / VHR)
    t1 = validate_gsd(0.08)
    assert t1["tier"] == "Tier 1" and t1["status"] == "PASS" and t1["suitable_for_individual_crowns"] is True
    print(f"  [PASS] Tier 1 (<0.15m): {t1['tier']} - {t1['rating']}")

    # Tier 2: 0.15 - 0.50m (Reliable Aerial/Satellite)
    t2 = validate_gsd(0.25)
    assert t2["tier"] == "Tier 2" and t2["status"] == "PASS" and t2["suitable_for_individual_crowns"] is True
    print(f"  [PASS] Tier 2 (0.15-0.50m): {t2['tier']} - {t2['rating']}")

    # Tier 3: 0.50 - 1.00m (Marginal / Dominant crowns only)
    t3 = validate_gsd(0.75)
    assert t3["tier"] == "Tier 3" and t3["status"] == "PASS_WITH_WARNING" and t3["suitable_for_individual_crowns"] is False
    print(f"  [PASS] Tier 3 (0.50-1.00m): {t3['tier']} - {t3['rating']}")

    # Warning: > 1.00m (Unsuitable for individual crown delineation)
    t4 = validate_gsd(1.50)
    assert t4["tier"] == "Warning" and t4["status"] == "FAIL_WARNING" and t4["suitable_for_individual_crowns"] is False
    print(f"  [PASS] Warning (>1.00m): {t4['tier']} - {t4['rating']}")


def test_utm_reprojection():
    """Verify UTM zone auto-detection and raster reprojection."""
    print("\n--- 2. Testing UTM Zone Detection & Reprojection ---")

    # Verify UTM zone calculation
    kolkata_epsg = get_utm_epsg_for_coords(88.3639, 22.5726)  # Kolkata, India
    florida_epsg = get_utm_epsg_for_coords(-81.9965, 29.6882)  # Florida, USA
    assert kolkata_epsg == 32645, f"Expected 32645 for Kolkata, got {kolkata_epsg}"
    assert florida_epsg == 32617, f"Expected 32617 for Florida, got {florida_epsg}"
    print(f"  [PASS] Kolkata UTM Zone: EPSG:{kolkata_epsg} (Zone 45N)")
    print(f"  [PASS] Florida UTM Zone: EPSG:{florida_epsg} (Zone 17N)")

    # Create a synthetic raster in EPSG:4326 (degrees) and reproject to metric UTM
    with tempfile.TemporaryDirectory() as tmpdir:
        synth_4326 = os.path.join(tmpdir, "synth_kolkata_4326.tif")
        transform = from_origin(88.36, 22.57, 0.00001, 0.00001)
        data = np.ones((3, 40, 40), dtype=np.uint8) * 128
        with rasterio.open(
            synth_4326,
            "w",
            driver="GTiff",
            height=40,
            width=40,
            count=3,
            dtype=np.uint8,
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data)

        # Reproject to auto-detected UTM (should be EPSG:32645)
        reproj_path = reproject_raster(synth_4326)
        meta = inspect_raster(reproj_path)
        assert meta.crs == "EPSG:32645", f"Expected EPSG:32645, got {meta.crs}"
        assert meta.is_projected is True
        print(f"  [PASS] Auto-reprojected EPSG:4326 raster -> {meta.crs} with GSD {meta.gsd_meters:.2f}m")


def test_aoi_parsing_and_clipping():
    """Verify parsing of GeoJSON and KML AOI polygons and raster clipping."""
    print("\n--- 3. Testing AOI Parsing (GeoJSON & KML) & Raster Clipping ---")

    sample_tif = get_data("OSBS_029.tif")
    meta = inspect_raster(sample_tif)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a bounding box covering the central 25m x 25m area in EPSG:32617
        center_x = (meta.bounds[0] + meta.bounds[2]) / 2.0
        center_y = (meta.bounds[1] + meta.bounds[3]) / 2.0
        test_poly = box(center_x - 12.5, center_y - 12.5, center_x + 12.5, center_y + 12.5)

        gdf = gpd.GeoDataFrame([{"name": "Central Core"}], geometry=[test_poly], crs="EPSG:32617")

        # 1. GeoJSON Test
        geojson_path = os.path.join(tmpdir, "core_aoi.geojson")
        gdf.to_file(geojson_path, driver="GeoJSON")
        parsed_geojson = parse_aoi(geojson_path, target_crs="EPSG:32617")
        assert len(parsed_geojson) == 1
        assert abs(parsed_geojson.area.iloc[0] - 625.0) < 1.0
        print(f"  [PASS] GeoJSON AOI parsed: {len(parsed_geojson)} feature(s), Area = {parsed_geojson.area.iloc[0]:.1f} m²")

        # 2. KML Test
        kml_path = os.path.join(tmpdir, "core_aoi.kml")
        gdf_4326 = gdf.to_crs("EPSG:4326")
        gdf_4326.to_file(kml_path, driver="KML")
        parsed_kml = parse_aoi(kml_path, target_crs="EPSG:32617")
        assert len(parsed_kml) == 1
        assert abs(parsed_kml.area.iloc[0] - 625.0) < 1.0
        print(f"  [PASS] KML AOI parsed: {len(parsed_kml)} feature(s), Area = {parsed_kml.area.iloc[0]:.1f} m²")

        # 3. Clip raster with GeoDataFrame AOI
        clipped_path = os.path.join(tmpdir, "clipped_core.tif")
        out_path, clip_meta = clip_raster_to_aoi(sample_tif, gdf, output_path=clipped_path)
        assert os.path.exists(out_path)
        clip_inspect = inspect_raster(out_path)
        assert clip_inspect.width > 0 and clip_inspect.height > 0
        print(f"  [PASS] Raster clipped to AOI: dimensions ({clip_inspect.width}x{clip_inspect.height}) px")


def test_sliding_window_tiler():
    """Verify sliding-window tile generation, overlap guarantees, and coordinate mapping."""
    print("\n--- 4. Testing Sliding-Window Tiler & Coordinate Mapping ---")

    # 400x400 image tiled into 250x250 windows with 20% overlap
    tiler = SlidingWindowTiler(tile_size=250, overlap=0.20)
    windows = tiler.get_window_grid(400, 400)
    assert len(windows) == 4, f"Expected 4 windows, got {len(windows)}"

    # Check tile coverage and boundaries
    assert windows[0].col_off == 0 and windows[0].row_off == 0
    assert windows[-1].col_off == 150 and windows[-1].row_off == 150
    print(f"  [PASS] Sliding-window grid: {len(windows)} tiles created with 20% overlap")

    # Test coordinate mapping
    df_local = gpd.pd.DataFrame([{"xmin": 10.0, "ymin": 20.0, "xmax": 40.0, "ymax": 50.0, "score": 0.85}])
    df_global = tiler.map_local_to_global_pixels(windows[3], df_local)
    assert df_global["xmin"].iloc[0] == 160.0
    assert df_global["ymin"].iloc[0] == 170.0
    print(f"  [PASS] Local tile coordinates correctly shifted to global raster pixel space")


def run_end_to_end_inference():
    """Run end-to-end detection, area calculation, calibration, biomass proxy, and visual output."""
    print("\n--- 5. Running End-to-End Tree Detection & Canopy Pipeline ---")

    sample_tif = get_data("OSBS_029.tif")
    meta = inspect_raster(sample_tif)

    print(f"Input Imagery: {meta.filepath}")
    print(f"CRS: {meta.crs} | Resolution: ({meta.resolution[0]:.2f}m, {meta.resolution[1]:.2f}m)")
    print(f"Dimensions: {meta.width} x {meta.height} px | GSD: {meta.gsd_meters:.3f}m ({meta.gsd_validation['tier']})")

    # 1. Initialize TreeDetector
    device_to_use = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Initializing DeepForest RetinaNet on device: '{device_to_use}'...")
    detector = TreeDetector(device=device_to_use, score_threshold=0.30, iou_threshold=0.25)

    # 2. Configure Sliding Window Tiler (e.g., 250x250 tiles with 20% overlap to verify multi-tile NMS deduplication)
    tiler = SlidingWindowTiler(tile_size=250, overlap=0.20)

    # 3. Predict Tiled Raster with Boundary NMS
    detections_df, _ = detector.predict_tiled_raster(sample_tif, tiler=tiler)
    print(f"Detected {len(detections_df)} tree crowns across sliding tiles after global boundary NMS.")

    assert len(detections_df) > 0, "Expected at least one detected tree in sample image!"

    # 4. Canopy Area Calculation Engine
    print("\nCalculating Canopy Cover, Li et al. (2023) Calibration, and Jucker et al. (2016) AGB...")
    engine = CanopyAreaEngine(allometry_type="angiosperm")
    metrics, df_enriched, dissolved_canopy = engine.compute_metrics(
        detections_df,
        raster_metadata=meta,
    )

    # Validations on Area Engine calculations:
    # Check ellipse shape factor
    expected_mean_area = (np.pi / 4.0) * (df_enriched["width_m"] * df_enriched["height_m"]).mean()
    assert abs(metrics.mean_crown_area_m2 - expected_mean_area) < 1e-4, "Ellipse shape factor calculation mismatch!"

    # Check that dissolved canopy area <= total raw crown area (due to overlap)
    assert metrics.dissolved_canopy_area_m2 <= metrics.total_raw_crown_area_m2 + 1e-4, "Dissolved area exceeds raw sum!"

    # Check Li et al. +20% calibration
    expected_calibrated_cover = min(100.0, metrics.canopy_cover_pct_raw * 1.20)
    assert abs(metrics.canopy_cover_pct_calibrated - expected_calibrated_cover) < 1e-4, "Li et al. calibration mismatch!"

    # Check ±35% AGB confidence bounds
    assert abs(metrics.agb_lower_bound_mg - metrics.agb_total_mg * 0.65) < 1e-4, "AGB lower bound mismatch!"
    assert abs(metrics.agb_upper_bound_mg - metrics.agb_total_mg * 1.35) < 1e-4, "AGB upper bound mismatch!"

    # 5. Print Formatted KPI Summary Table
    print("\n" + "=" * 70)
    print("                CANOPY & BIOMASS KPI SUMMARY TABLE")
    print("=" * 70)
    print(metrics.format_table())
    print("=" * 70)

    # 6. Generate and Save Visual Output
    with rasterio.open(sample_tif) as src:
        rgb_image = src.read((1, 2, 3)).transpose((1, 2, 0))

    output_png = "output_prediction.png"
    saved_path = save_canopy_report_plot(
        image=rgb_image,
        detections_df=df_enriched,
        metrics=metrics,
        dissolved_geometry=dissolved_canopy,
        raster_transform=meta.transform,
        output_path=output_png,
    )

    assert os.path.exists(saved_path), f"Output image {saved_path} was not created!"
    assert os.path.getsize(saved_path) > 10000, "Output image file size is unexpectedly small!"
    print(f"\n[PASS] Visual annotated prediction saved successfully to: {saved_path}")

    return metrics


def main():
    """Main test runner executing all unit and integration tests."""
    print("==================================================================")
    print("        GEOSPATIAL CANOPY INFERENCE & AREA PIPELINE TESTS         ")
    print("==================================================================")

    test_gsd_validation()
    test_utm_reprojection()
    test_aoi_parsing_and_clipping()
    test_sliding_window_tiler()
    metrics = run_end_to_end_inference()

    print("\n==================================================================")
    print("        ALL CANOPY CORE PIPELINE TESTS COMPLETED SUCCESSFULLY!    ")
    print("==================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
