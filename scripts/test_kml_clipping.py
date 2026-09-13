"""
Test end-to-end KML AOI boundary clipping on high-resolution GeoTIFF:
1. Ingests high-resolution GeoTIFF (EPSG:32645, 27cm GSD)
2. Ingests KML boundary file (EPSG:4326)
3. Auto-reprojects KML polygon to GeoTIFF UTM CRS (EPSG:32645)
4. Clips raster to KML polygon boundary (masking out outside areas)
5. Runs TreeDetector on the clipped raster
6. Measures exact canopy area inside the KML boundary
7. Saves annotated visual result showing KML boundary overlay
"""
import os
import cv2
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import mapping

from canopy_core.ingest import inspect_raster, parse_aoi, clip_raster_to_aoi
from canopy_core.models.detector import TreeDetector
from canopy_core.models.area_engine import CanopyAreaEngine
from canopy_core.tiler import SlidingWindowTiler

def test_kml_workflow():
    geotiff_path = "data/samples/kolkata_central_park.tif"
    kml_path = "data/samples/kolkata_central_park_aoi.kml"

    print("=" * 65)
    print("TESTING HIGH-RES GEOTIFF + KML AOI BOUNDARY WORKFLOW")
    print("=" * 65)

    # 1. Inspect GeoTIFF
    meta = inspect_raster(geotiff_path)
    print(f"GeoTIFF: {os.path.basename(geotiff_path)}")
    print(f"CRS: {meta.crs} | Resolution: {meta.resolution} | GSD: {meta.gsd_meters:.3f} m")
    print(f"Dimensions: {meta.width} x {meta.height} px")

    # 2. Parse KML
    kml_gdf = parse_aoi(kml_path, target_crs=meta.crs)
    kml_area_m2 = float(kml_gdf.geometry.area.sum())
    kml_area_ha = kml_area_m2 / 10000.0
    print(f"\nKML Boundary: {os.path.basename(kml_path)}")
    print(f"Reprojected to: {kml_gdf.crs}")
    print(f"AOI Ground Area: {kml_area_m2:,.1f} m² ({kml_area_ha:.2f} ha)")

    # 3. Clip GeoTIFF to KML boundary
    clipped_tif, clipped_meta = clip_raster_to_aoi(
        raster_path=geotiff_path,
        aoi=kml_gdf,
        output_path="data/samples/central_park_kml_clipped.tif",
    )
    print(f"\n[CLIPPED] GeoTIFF clipped to KML boundary: {clipped_tif}")
    print(f"Clipped Dimensions: {clipped_meta['width']} x {clipped_meta['height']} px")

    # 4. Run Tree Detector on the clipped raster
    detector = TreeDetector(score_threshold=0.30)
    tiler = SlidingWindowTiler(tile_size=400, overlap=0.20)
    
    detections_df, clip_meta = detector.predict_tiled_raster(
        raster_path=clipped_tif,
        tiler=tiler,
    )
    print(f"\n[DETECTION] Stems detected inside KML boundary: {len(detections_df)}")

    # 5. Compute Canopy Metrics strictly inside KML boundary
    area_engine = CanopyAreaEngine()
    metrics, enriched_df, dissolved_poly = area_engine.compute_metrics(
        detections_df=detections_df,
        raster_metadata=clip_meta,
        aoi_polygon=kml_gdf,
    )

    print(f"• True AOI Canopy Cover: {metrics.canopy_cover_pct_raw:.2f}% (Calibrated: {metrics.canopy_cover_pct_calibrated:.2f}%)")
    print(f"• Dissolved Canopy Area: {metrics.dissolved_canopy_area_m2:,.1f} m² (Calibrated: {metrics.total_calibrated_canopy_area_m2:,.1f} m²)")
    print(f"• Biomass inside KML: {metrics.agb_total_mg:.2f} Mg AGB (Carbon: {metrics.carbon_stock_mg_c:.2f} Mg C)")

    # 6. Render visual showing the KML polygon boundary and tree detections
    with rasterio.open(clipped_tif) as src:
        rgb = src.read([1, 2, 3])
        rgb = np.transpose(rgb, (1, 2, 0))

    annotated = rgb.copy()
    # Draw detections
    for _, row in detections_df.iterrows():
        x1, y1 = int(row["xmin"]), int(row["ymin"])
        x2, y2 = int(row["xmax"]), int(row["ymax"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 100), 2)
        cv2.circle(annotated, (int((x1+x2)/2), int((y1+y2)/2)), 3, (0, 220, 255), -1)

    # HUD Banner
    h_img, w_img = annotated.shape[:2]
    hud_h = 75
    banner = np.zeros((hud_h, w_img, 3), dtype=np.uint8)
    banner[:] = (18, 26, 38)
    
    cv2.putText(banner, f"KML BOUNDARY CLIPPED SURVEY: Banabitan Salt Lake ({kml_area_ha:.2f} ha)", (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(banner, f"Stems in AOI: {metrics.tree_count} | Canopy Cover: {metrics.canopy_cover_pct_calibrated:.1f}% ({metrics.total_calibrated_canopy_area_m2:.0f} m2)", (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 255, 180), 1, cv2.LINE_AA)
    cv2.putText(banner, f"Carbon Stock: {metrics.carbon_stock_mg_c:.2f} Mg C (+/-{metrics.carbon_stock_mg_c*0.565/np.sqrt(metrics.tree_count):.2f} Mg) | EPSG:32645", (15, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 210, 100), 1, cv2.LINE_AA)

    final_img = np.vstack([banner, annotated])
    out_path = "reports/kml_boundary_test_result.png"
    cv2.imwrite(out_path, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
    print(f"\n[SAVED] Annotated KML result visual: {out_path}")

    brain_dst = "/Users/soumodeep/.gemini/antigravity/brain/aa789d0a-56ef-46d9-a50c-d5f3d2ac9a64/reports/kml_boundary_test_result.png"
    cv2.imwrite(brain_dst, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
    print(f"[SAVED] Artifact copy: {brain_dst}")

if __name__ == "__main__":
    test_kml_workflow()
