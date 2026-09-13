import math, os, time, requests, io
from pathlib import Path
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
import pyproj
import geopandas as gpd

from canopy_core import TreeDetector, CanopyAreaEngine, SlidingWindowTiler, inspect_raster
from canopy_core.utils.visualization import save_canopy_report_plot

def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)

def fetch_ju_sample():
    os.makedirs("data/samples", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    # Jadavpur University Main Campus, Kolkata
    # Center: 22.4990° N, 88.3715° E
    lat, lon = 22.4990, 88.3715
    zoom = 19
    grid_size = 4 # 4x4 tiles = 1024x1024 px (~280m x 280m)
    
    cx, cy = deg2num(lat, lon, zoom)
    start_x = cx - grid_size // 2
    start_y = cy - grid_size // 2
    
    print(f"[1/4] Fetching {grid_size}x{grid_size} tile grid for Jadavpur University Main Campus...")
    rows = []
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    
    for dy in range(grid_size):
        cols = []
        for dx in range(grid_size):
            x = start_x + dx
            y = start_y + dy
            url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 200:
                tile = Image.open(io.BytesIO(r.content)).convert("RGB")
            else:
                tile = Image.new("RGB", (256, 256), (50, 80, 50))
            cols.append(np.array(tile))
        rows.append(np.hstack(cols))
    mosaic = np.vstack(rows)
    print(f"  Downloaded mosaic shape: {mosaic.shape}")
    
    # Compute bounds in WGS84
    top_lat, left_lon = num2deg(start_x, start_y, zoom)
    bot_lat, right_lon = num2deg(start_x + grid_size, start_y + grid_size, zoom)
    
    # Reproject to UTM Zone 45N (EPSG:32645)
    print("[2/4] Reprojecting to metric UTM Zone 45N (EPSG:32645)...")
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32645", always_xy=True)
    left_m, top_m = transformer.transform(left_lon, top_lat)
    right_m, bot_m = transformer.transform(right_lon, bot_lat)
    
    raw_tif = "data/samples/temp_ju_raw.tif"
    out_tif = "data/samples/jadavpur_university.tif"
    
    w, h = mosaic.shape[1], mosaic.shape[0]
    raw_transform = from_bounds(left_m, bot_m, right_m, top_m, w, h)
    
    with rasterio.open(
        out_tif,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=3,
        dtype=mosaic.dtype,
        crs="EPSG:32645",
        transform=raw_transform,
    ) as dst:
        for b in range(3):
            dst.write(mosaic[:, :, b], b + 1)
            
    print(f"  [SAVED] Georeferenced GeoTIFF: {out_tif}")
    
    # 3. Inspect Raster & Validate GSD
    print("\n[3/4] Inspecting raster and running tree detection on Apple M4 MPS...")
    meta = inspect_raster(out_tif)
    print(f"  CRS: {meta.crs} | GSD: {meta.gsd_meters:.3f}m ({meta.gsd_validation['rating']})")
    print(f"  Dimensions: {meta.width}x{meta.height} px | Area: {(meta.bounds[2]-meta.bounds[0])*(meta.bounds[3]-meta.bounds[1]):,.1f} m² ({(meta.bounds[2]-meta.bounds[0])*(meta.bounds[3]-meta.bounds[1])/10000:.2f} ha)")
    
    detector = TreeDetector()
    tiler = SlidingWindowTiler(tile_size=400, overlap=0.20)
    
    t0 = time.time()
    boxes_df, _ = detector.predict_tiled_raster(out_tif, tiler=tiler, score_threshold=0.28)
    elapsed = time.time() - t0
    print(f"  Inference completed in {elapsed:.2f}s on Apple M4 MPS! Detected {len(boxes_df)} tree crowns.")
    
    # 4. Compute Metrics & Save Reports
    print("\n[4/4] Computing non-overlapping canopy cover and biomass metrics...")
    engine = CanopyAreaEngine()
    metrics, enriched_df, dissolved_union = engine.compute_metrics(boxes_df, raster_metadata=meta)
    
    plot_path = "reports/jadavpur_university_analysis.png"
    save_canopy_report_plot(
        image=mosaic,
        detections_df=enriched_df,
        metrics=metrics,
        dissolved_geometry=dissolved_union,
        raster_transform=raw_transform,
        output_path=plot_path
    )
    print(f"  [SAVED] Dual-panel Analysis Plot: {plot_path}")
    
    if len(enriched_df) > 0:
        crown_ellipses = engine.create_crown_ellipses(enriched_df)
        gdf = gpd.GeoDataFrame(enriched_df, geometry=crown_ellipses, crs=meta.crs)
        geojson_path = "reports/jadavpur_university_crowns.geojson"
        cols_to_export = [c for c in gdf.columns if c not in ["image_path"]]
        gdf[cols_to_export].to_file(geojson_path, driver="GeoJSON")
        print(f"  [SAVED] Vector GeoJSON Crowns: {geojson_path}")
        
    print("\n" + "=" * 68)
    print("      JADAVPUR UNIVERSITY MAIN CAMPUS — TREE CANOPY METRICS")
    print("=" * 68)
    print(metrics.format_table())
    print("=" * 68)

if __name__ == "__main__":
    fetch_ju_sample()
