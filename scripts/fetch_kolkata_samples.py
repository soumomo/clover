#!/usr/bin/env python3
"""
scripts/fetch_kolkata_samples.py

Fetches and curates high-resolution remote sensing datasets for West Bengal and Kolkata:
1. Kolkata Central Park (Banabitan, Salt Lake): 22.5855 N, 88.4180 E (EPSG:32645)
2. Kolkata Maidan / Victoria Memorial Gardens: 22.5448 N, 88.3426 E (EPSG:32645)
3. Sundarbans Biosphere Mangrove Canopy: 21.9497 N, 88.8995 E (EPSG:32645)

Features:
- Downloads Zoom 19 (~27cm GSD) tiles from Esri World Imagery XYZ service.
- Handles remote areas with intelligent parent-tile fallback (e.g. Sundarbans zoom 18 upsampling).
- Reprojects Web Mercator (EPSG:3857) mosaic to WGS 84 / UTM Zone 45N (EPSG:32645) with ~0.27m GSD.
- Copies DeepForest benchmark OSBS_029.tif and OSBS_029.csv into data/samples/.
- Generates Kolkata Central Park AOI polygon in GeoJSON and KML.
- Produces downsampled coarse satellite patch (>1.5m GSD) to trigger GSD suitability warning.
- Validates and prints a verification summary table using rasterio.
"""

import os
import sys
import math
import json
import shutil
import io
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
import pyproj
from shapely.geometry import Polygon, mapping

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "samples")

# Earth constants for Web Mercator (EPSG:3857)
ORIGIN_SHIFT = 20037508.342789244
ESRI_TILE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

SITES = [
    {
        "id": "kolkata_central_park",
        "name": "Kolkata Central Park (Banabitan, Salt Lake)",
        "lat": 22.5855,
        "lon": 88.4180,
        "filename": "kolkata_central_park.tif",
        "description": "Urban dense canopy and water bodies in Salt Lake Sector II"
    },
    {
        "id": "kolkata_maidan_victoria",
        "name": "Kolkata Maidan / Victoria Memorial Gardens",
        "lat": 22.5448,
        "lon": 88.3426,
        "filename": "kolkata_maidan_victoria.tif",
        "description": "Historic urban heritage park, manicured gardens and tree avenues"
    },
    {
        "id": "sundarbans_mangrove",
        "name": "Sundarbans Biosphere Mangrove Canopy",
        "lat": 21.9497,
        "lon": 88.8995,
        "filename": "sundarbans_mangrove.tif",
        "description": "Dense tidal halophytic mangrove forest canopy in UNESCO World Heritage Sundarbans"
    }
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko)"
}

_TILE_CACHE = {}


def deg2num(lat_deg: float, lon_deg: float, zoom: int):
    """Convert WGS84 lat/lon to slippy map tile coordinates."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def fetch_tile_raw(z: int, y: int, x: int) -> bytes:
    """Fetch raw JPEG bytes for a tile with simple in-memory caching."""
    key = (z, y, x)
    if key in _TILE_CACHE:
        return _TILE_CACHE[key]
    url = ESRI_TILE_URL.format(z=z, y=y, x=x)
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = resp.read()
    _TILE_CACHE[key] = data
    return data


def fetch_tile_image(z: int, y: int, x: int) -> Image.Image:
    """
    Fetch a tile image. If tile is a zoom 19 placeholder (blank grey),
    gracefully fallback to parent tile at zoom 18 and upsample sub-quadrant.
    """
    try:
        raw = fetch_tile_raw(z, y, x)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        arr = np.array(img)
        # Check if tile is Esri placeholder (uniform grey around 204 with near-zero std)
        if z == 19 and arr.std() < 7.0 and abs(arr.mean() - 204.7) < 5.0:
            # Fallback to parent zoom 18 tile
            pz = 18
            px = x // 2
            py = y // 2
            sub_x = x % 2
            sub_y = y % 2
            p_raw = fetch_tile_raw(pz, py, px)
            p_img = Image.open(io.BytesIO(p_raw)).convert("RGB")
            crop_box = (sub_x * 128, sub_y * 128, (sub_x + 1) * 128, (sub_y + 1) * 128)
            cropped = p_img.crop(crop_box)
            return cropped.resize((256, 256), Image.Resampling.BICUBIC)
        return img
    except Exception as exc:
        print(f"Warning: Failed to fetch tile ({z}, {y}, {x}): {exc}")
        # Return fallback blank tile
        return Image.new("RGB", (256, 256), (0, 0, 0))


def download_and_mosaic_site(site: dict, zoom: int = 19, tile_radius: int = 2) -> str:
    """
    Download a tile_radius*2 by tile_radius*2 grid around site coordinates,
    mosaic in EPSG:3857, and warp to EPSG:32645 (UTM Zone 45N) at ~0.27m GSD.
    """
    lat, lon = site["lat"], site["lon"]
    cx, cy = deg2num(lat, lon, zoom)

    x_min = cx - tile_radius
    x_max = cx + tile_radius - 1
    y_min = cy - tile_radius
    y_max = cy + tile_radius - 1

    x_tiles = list(range(x_min, x_max + 1))
    y_tiles = list(range(y_min, y_max + 1))
    num_x = len(x_tiles)
    num_y = len(y_tiles)

    print(f"\n[Fetching] {site['name']} (lat={lat}, lon={lon})")
    print(f"  Tile grid: {num_x}x{num_y} ({num_x * num_y} tiles) at zoom {zoom}")

    # Fetch tiles in parallel
    tile_coords = [(zoom, y, x) for y in y_tiles for x in x_tiles]
    with ThreadPoolExecutor(max_workers=8) as pool:
        tile_imgs = list(pool.map(lambda c: fetch_tile_image(*c), tile_coords))

    # Assemble mosaic in EPSG:3857
    mosaic_w = num_x * 256
    mosaic_h = num_y * 256
    mosaic = Image.new("RGB", (mosaic_w, mosaic_h))

    for idx, (z, y, x) in enumerate(tile_coords):
        col = x - x_min
        row = y - y_min
        mosaic.paste(tile_imgs[idx], (col * 256, row * 256))

    arr_3857 = np.array(mosaic)  # Shape: (H, W, 3)

    # Calculate exact EPSG:3857 bounding box
    res_3857 = 2.0 * ORIGIN_SHIFT / (256.0 * (2.0 ** zoom))
    src_minx = -ORIGIN_SHIFT + x_min * 256.0 * res_3857
    src_maxx = -ORIGIN_SHIFT + (x_max + 1) * 256.0 * res_3857
    src_maxy = ORIGIN_SHIFT - y_min * 256.0 * res_3857
    src_miny = ORIGIN_SHIFT - (y_max + 1) * 256.0 * res_3857

    src_transform = from_bounds(src_minx, src_miny, src_maxx, src_maxy, mosaic_w, mosaic_h)
    src_crs = "EPSG:3857"
    dst_crs = "EPSG:32645"

    # Reproject to UTM Zone 45N (EPSG:32645) with target GSD = 0.27m (~27 cm)
    target_gsd = 0.27
    dst_transform, dst_w, dst_h = calculate_default_transform(
        src_crs, dst_crs, mosaic_w, mosaic_h,
        src_minx, src_miny, src_maxx, src_maxy,
        resolution=target_gsd
    )

    # Warp bands from (H, W, 3) to (3, dst_h, dst_w)
    src_bands = np.moveaxis(arr_3857, -1, 0)  # (3, H, W)
    dst_bands = np.zeros((3, dst_h, dst_w), dtype=np.uint8)

    for b in range(3):
        reproject(
            source=src_bands[b],
            destination=dst_bands[b],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear
        )

    out_path = os.path.join(DATA_DIR, site["filename"])
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=dst_h,
        width=dst_w,
        count=3,
        dtype="uint8",
        crs=dst_crs,
        transform=dst_transform,
        compress="deflate",
        photometric="RGB"
    ) as dst:
        dst.write(dst_bands)

    file_size_kb = os.path.getsize(out_path) / 1024.0
    print(f"  -> Saved {out_path}")
    print(f"     Dimensions: {dst_w}x{dst_h}, Bands: 3, GSD: {dst_transform[0]:.3f}m, Size: {file_size_kb:.1f} KB")
    return out_path


def copy_benchmark_files():
    """Copy benchmark OSBS_029.tif and OSBS_029.csv from deepforest installation."""
    print("\n[Copying Benchmark Data]")
    try:
        import deepforest
        deepforest_dir = os.path.dirname(deepforest.__file__)
        src_tif = os.path.join(deepforest_dir, "data", "OSBS_029.tif")
        src_csv = os.path.join(deepforest_dir, "data", "OSBS_029.csv")

        dst_tif = os.path.join(DATA_DIR, "OSBS_029.tif")
        dst_csv = os.path.join(DATA_DIR, "OSBS_029.csv")

        if os.path.exists(src_tif):
            shutil.copy2(src_tif, dst_tif)
            print(f"  Copied {src_tif} -> {dst_tif} ({os.path.getsize(dst_tif)/1024:.1f} KB)")
        else:
            print(f"  Warning: Benchmark {src_tif} not found!")

        if os.path.exists(src_csv):
            shutil.copy2(src_csv, dst_csv)
            print(f"  Copied {src_csv} -> {dst_csv} ({os.path.getsize(dst_csv)/1024:.1f} KB)")
        else:
            print(f"  Warning: Benchmark {src_csv} not found!")

    except ImportError:
        print("  Warning: deepforest is not installed in Python environment.")


def generate_kolkata_central_park_aoi():
    """
    Generate sample AOI boundaries:
    - data/samples/kolkata_central_park_aoi.geojson
    - data/samples/kolkata_central_park_aoi.kml
    Covering the core canopy survey zone of Kolkata Central Park (Banabitan).
    """
    print("\n[Creating AOI Boundaries]")
    # Polygon coordinates encompassing Central Park (Banabitan) canopy core area (Lon, Lat)
    # Inside the bounds of kolkata_central_park.tif [88.4166 - 88.4194 E, 22.5848 - 22.5874 N]
    coords_wgs84 = [
        [88.41680, 22.58510],
        [88.41910, 22.58510],
        [88.41925, 22.58610],
        [88.41870, 22.58715],
        [88.41720, 22.58715],
        [88.41675, 22.58620],
        [88.41680, 22.58510]  # Closed ring
    ]

    poly = Polygon(coords_wgs84)

    # Compute area in UTM 45N meters
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32645", always_xy=True)
    coords_utm = [transformer.transform(x, y) for x, y in coords_wgs84]
    poly_utm = Polygon(coords_utm)
    area_sq_m = poly_utm.area
    area_ha = area_sq_m / 10000.0

    geojson_data = {
        "type": "FeatureCollection",
        "name": "kolkata_central_park_aoi",
        "crs": {
            "type": "name",
            "properties": {
                "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
            }
        },
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "site_id": "WB_KOL_CP_001",
                    "name": "Kolkata Central Park (Banabitan) Canopy Survey AOI",
                    "park_name": "Banabitan (Central Park), Salt Lake Sector II",
                    "city": "Kolkata",
                    "district": "North 24 Parganas",
                    "state": "West Bengal",
                    "country": "India",
                    "target_crs": "EPSG:32645",
                    "survey_type": "Tree Crown Delineation & Canopy Monitoring",
                    "area_sq_m": round(area_sq_m, 2),
                    "area_hectares": round(area_ha, 4),
                    "nominal_gsd_m": 0.27
                },
                "geometry": mapping(poly)
            }
        ]
    }

    geojson_path = os.path.join(DATA_DIR, "kolkata_central_park_aoi.geojson")
    with open(geojson_path, "w") as f:
        json.dump(geojson_data, f, indent=2)
    print(f"  Created GeoJSON AOI: {geojson_path} ({os.path.getsize(geojson_path)/1024:.2f} KB)")
    print(f"  AOI Area: {area_sq_m:.1f} m² ({area_ha:.3f} ha)")

    # Build KML
    kml_coords_str = " ".join([f"{lon},{lat},0" for lon, lat in coords_wgs84])
    kml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Kolkata Central Park AOI</name>
    <description>Area of Interest for Tree Crown Delineation and Urban Forestry in Central Park (Banabitan), Salt Lake, Kolkata, West Bengal</description>
    <Style id="canopyAoiStyle">
      <LineStyle>
        <color>ff00aa00</color>
        <width>2.5</width>
      </LineStyle>
      <PolyStyle>
        <color>4d00ff00</color>
      </PolyStyle>
    </Style>
    <Placemark>
      <name>Central Park Canopy Survey AOI</name>
      <description><![CDATA[
        <b>Location:</b> Banabitan (Central Park), Salt Lake Sector II, Kolkata<br/>
        <b>Target CRS:</b> EPSG:32645 (UTM Zone 45N)<br/>
        <b>Area:</b> {area_sq_m:.1f} m² ({area_ha:.3f} ha)<br/>
        <b>Nominal GSD:</b> ~0.27 m
      ]]></description>
      <styleUrl>#canopyAoiStyle</styleUrl>
      <Polygon>
        <extrude>1</extrude>
        <altitudeMode>clampToGround</altitudeMode>
        <outerBoundaryIs>
          <LinearRing>
            <coordinates>
              {kml_coords_str}
            </coordinates>
          </LinearRing>
        </outerBoundaryIs>
      </Polygon>
    </Placemark>
  </Document>
</kml>
"""
    kml_path = os.path.join(DATA_DIR, "kolkata_central_park_aoi.kml")
    with open(kml_path, "w") as f:
        f.write(kml_content)
    print(f"  Created KML AOI: {kml_path} ({os.path.getsize(kml_path)/1024:.2f} KB)")


def generate_coarse_satellite_patch():
    """
    Downsample kolkata_central_park.tif to >1.5m GSD (target: 2.0m GSD)
    to test and trigger the GSD suitability warning.
    """
    print("\n[Generating Downsampled Coarse Satellite Patch]")
    src_path = os.path.join(DATA_DIR, "kolkata_central_park.tif")
    dst_path = os.path.join(DATA_DIR, "coarse_satellite_warning.tif")

    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source raster {src_path} not found for downsampling.")

    target_gsd = 2.0  # 2.0 meters/pixel, which is > 1.5m GSD threshold

    with rasterio.open(src_path) as src:
        bounds = src.bounds
        src_crs = src.crs

        # Calculate new width and height for 2.0m GSD
        new_width = int(round((bounds.right - bounds.left) / target_gsd))
        new_height = int(round((bounds.top - bounds.bottom) / target_gsd))

        new_transform = from_bounds(
            bounds.left, bounds.bottom, bounds.right, bounds.top, new_width, new_height
        )

        src_data = src.read()
        dst_data = np.zeros((src.count, new_height, new_width), dtype=np.uint8)

        for b in range(src.count):
            reproject(
                source=src_data[b],
                destination=dst_data[b],
                src_transform=src.transform,
                src_crs=src_crs,
                dst_transform=new_transform,
                dst_crs=src_crs,
                resampling=Resampling.average
            )

        with rasterio.open(
            dst_path,
            "w",
            driver="GTiff",
            height=new_height,
            width=new_width,
            count=src.count,
            dtype="uint8",
            crs=src_crs,
            transform=new_transform,
            compress="deflate",
            photometric="RGB"
        ) as dst:
            dst.write(dst_data)

    file_size_kb = os.path.getsize(dst_path) / 1024.0
    print(f"  Created coarse satellite patch: {dst_path}")
    print(f"  Dimensions: {new_width}x{new_height}, GSD: {target_gsd:.2f}m x {target_gsd:.2f}m (>1.5m threshold), Size: {file_size_kb:.1f} KB")


def verify_datasets():
    """Verify all datasets in data/samples/ using rasterio and print summary table."""
    print("\n" + "=" * 110)
    print(f"{'GEOSPATIAL DATASET CURATION SUMMARY':^110}")
    print("=" * 110)

    header = f"{'Filename':<30} | {'CRS':<12} | {'GSD (m)':<14} | {'Dimensions (WxH)':<18} | {'File Size':<10} | {'GSD Status':<12}"
    print(header)
    print("-" * 110)

    tif_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".tif")])

    for f in tif_files:
        path = os.path.join(DATA_DIR, f)
        with rasterio.open(path) as src:
            gsd_x = abs(src.transform[0])
            gsd_y = abs(src.transform[4])
            gsd_str = f"{gsd_x:.3f} x {gsd_y:.3f}"
            crs_str = str(src.crs) if src.crs else "None"
            dims = f"{src.width} x {src.height} (x{src.count})"
            size_kb = os.path.getsize(path) / 1024.0
            size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb/1024:.2f} MB"

            # GSD warning check
            max_gsd = max(gsd_x, gsd_y)
            if max_gsd > 1.5:
                status = "WARN (>1.5m)"
            elif max_gsd <= 0.35:
                status = "OPTIMAL"
            else:
                status = "ACCEPTABLE"

            print(f"{f:<30} | {crs_str:<12} | {gsd_str:<14} | {dims:<18} | {size_str:<10} | {status:<12}")

    print("-" * 110)
    print("\nNon-raster sample assets:")
    for f in sorted(os.listdir(DATA_DIR)):
        if not f.endswith(".tif"):
            path = os.path.join(DATA_DIR, f)
            sz_kb = os.path.getsize(path) / 1024.0
            print(f"  - {f:<35} ({sz_kb:.2f} KB)")
    print("=" * 110)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Target sample directory: {DATA_DIR}")

    # 1. Fetch and process the 3 Kolkata & West Bengal sites
    for site in SITES:
        download_and_mosaic_site(site, zoom=19, tile_radius=2)

    # 2. Copy benchmark dataset
    copy_benchmark_files()

    # 3. Create AOI boundary files (GeoJSON and KML)
    generate_kolkata_central_park_aoi()

    # 4. Generate coarse satellite warning patch (>1.5m GSD)
    generate_coarse_satellite_patch()

    # 5. Verify all files
    verify_datasets()


if __name__ == "__main__":
    main()
