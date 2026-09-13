"""Ingestion module for canopy analysis: raster inspection, GSD calculation, reprojection, and AOI clipping."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pyproj
from pyproj import Geod
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject
from shapely.geometry import box, mapping


@dataclass
class RasterMetadata:
    """Detailed metadata container for a geospatial raster."""

    filepath: str
    crs: str
    is_projected: bool
    is_geographic: bool
    width: int
    height: int
    count: int
    dtype: str
    bounds: Tuple[float, float, float, float]  # (left, bottom, right, top)
    transform: rasterio.Affine
    resolution: Tuple[float, float]  # (res_x, res_y) in native units
    gsd_meters: float
    gsd_validation: Dict[str, Any]
    nodata: Optional[float] = None
    extra_tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to a serializable dictionary."""
        return {
            "filepath": self.filepath,
            "crs": self.crs,
            "is_projected": self.is_projected,
            "is_geographic": self.is_geographic,
            "width": self.width,
            "height": self.height,
            "count": self.count,
            "dtype": self.dtype,
            "bounds": {
                "left": self.bounds[0],
                "bottom": self.bounds[1],
                "right": self.bounds[2],
                "top": self.bounds[3],
            },
            "transform": [float(x) for x in self.transform],
            "resolution": [float(x) for x in self.resolution],
            "gsd_meters": round(self.gsd_meters, 4),
            "gsd_validation": self.gsd_validation,
            "nodata": self.nodata,
        }


def calculate_gsd(
    transform: rasterio.Affine,
    crs: Optional[rasterio.crs.CRS],
    bounds: Optional[Tuple[float, float, float, float]] = None,
    shape: Optional[Tuple[int, int]] = None,
) -> float:
    """Calculate the Ground Sampling Distance (GSD) in meters.

    Handles both projected CRS (meters, feet) and geographic CRS (EPSG:4326 degrees).
    For geographic coordinates, calculates the ellipsoidal distance at the centroid of
    the bounding box using WGS84 Geod.

    Args:
        transform: Affine transformation matrix.
        crs: Rasterio CRS object or None.
        bounds: Optional (left, bottom, right, top) bounds tuple.
        shape: Optional (height, width) dimensions.

    Returns:
        float: Average ground sampling distance in meters per pixel.
    """
    res_x = abs(transform.a)
    res_y = abs(transform.e)

    if crs is None:
        # Fallback when CRS is undefined
        return float((res_x + res_y) / 2.0)

    if crs.is_projected:
        linear_units = getattr(crs, "linear_units", "metre").lower()
        if "foot" in linear_units or "feet" in linear_units:
            scale_factor = 0.3048
        elif "us survey" in linear_units:
            scale_factor = 1200.0 / 3937.0
        else:
            scale_factor = 1.0  # Assumed meters (e.g. UTM)

        return float(((res_x + res_y) / 2.0) * scale_factor)

    # Geographic CRS (degrees) - e.g. EPSG:4326
    if bounds is not None:
        center_lon = (bounds[0] + bounds[2]) / 2.0
        center_lat = (bounds[1] + bounds[3]) / 2.0
    elif shape is not None:
        center_c = shape[1] / 2.0
        center_r = shape[0] / 2.0
        center_lon, center_lat = transform * (center_c, center_r)
    else:
        center_lon, center_lat = transform * (0, 0)

    geod = Geod(ellps="WGS84")
    # Distance in X direction (longitude shift by res_x degrees)
    _, _, dx = geod.inv(center_lon, center_lat, center_lon + res_x, center_lat)
    # Distance in Y direction (latitude shift by res_y degrees)
    _, _, dy = geod.inv(center_lon, center_lat, center_lon, center_lat + res_y)

    return float((dx + dy) / 2.0)


def validate_gsd(gsd_meters: float) -> Dict[str, Any]:
    """Validate Ground Sampling Distance (GSD) against forestry remote sensing literature.

    Literature Thresholds:
    - Tier 1: < 0.15m (UAV / VHR aerial: optimal for fine individual crown delineation)
    - Tier 2: 0.15m - 0.50m (High-res aerial/satellite: reliable individual crown detection)
    - Tier 3: 0.50m - 1.00m (Medium-high resolution: dominant/emergent crowns only; edge uncertainty)
    - Warning: > 1.00m (Coarse resolution: crown delineation unreliable; canopy proxy only)

    Args:
        gsd_meters: Ground sampling distance in meters.

    Returns:
        dict: Validation results including tier, rating, status, and descriptive notes.
    """
    if gsd_meters < 0.15:
        return {
            "tier": "Tier 1",
            "rating": "Optimal",
            "status": "PASS",
            "gsd_meters": round(gsd_meters, 4),
            "suitable_for_individual_crowns": True,
            "description": (
                f"High resolution ({gsd_meters:.3f}m GSD < 0.15m). Optimal for "
                "individual tree crown delineation and fine-scale canopy architecture."
            ),
        }
    elif gsd_meters <= 0.50:
        return {
            "tier": "Tier 2",
            "rating": "Reliable",
            "status": "PASS",
            "gsd_meters": round(gsd_meters, 4),
            "suitable_for_individual_crowns": True,
            "description": (
                f"Medium-high resolution ({gsd_meters:.3f}m GSD in [0.15m, 0.50m]). "
                "Reliable for individual tree crown detection in closed/open forests."
            ),
        }
    elif gsd_meters <= 1.00:
        return {
            "tier": "Tier 3",
            "rating": "Marginal",
            "status": "PASS_WITH_WARNING",
            "gsd_meters": round(gsd_meters, 4),
            "suitable_for_individual_crowns": False,
            "description": (
                f"Coarse-medium resolution ({gsd_meters:.3f}m GSD in (0.50m, 1.00m]). "
                "Detects dominant and emergent crowns; crown boundary uncertainty is high."
            ),
        }
    else:
        return {
            "tier": "Warning",
            "rating": "Unsuitable",
            "status": "FAIL_WARNING",
            "gsd_meters": round(gsd_meters, 4),
            "suitable_for_individual_crowns": False,
            "description": (
                f"Coarse resolution ({gsd_meters:.3f}m GSD > 1.00m). Individual tree "
                "crown delineation is unreliable; aggregate fractional cover recommended."
            ),
        }


def inspect_raster(raster_path: Union[str, Path]) -> RasterMetadata:
    """Inspect and extract comprehensive metadata and GSD from a raster file.

    Args:
        raster_path: Path to the GeoTIFF or raster file.

    Returns:
        RasterMetadata: Populated dataclass containing CRS, resolution, GSD, and validation.
    """
    path_str = str(raster_path)
    with rasterio.open(path_str) as src:
        crs_obj = src.crs
        crs_str = crs_obj.to_string() if crs_obj is not None else "UNDEFINED"
        is_projected = crs_obj.is_projected if crs_obj is not None else False
        is_geographic = crs_obj.is_geographic if crs_obj is not None else False

        bounds_tuple = (
            float(src.bounds.left),
            float(src.bounds.bottom),
            float(src.bounds.right),
            float(src.bounds.top),
        )

        gsd = calculate_gsd(src.transform, crs_obj, bounds_tuple, src.shape)
        validation = validate_gsd(gsd)

        return RasterMetadata(
            filepath=os.path.abspath(path_str),
            crs=crs_str,
            is_projected=is_projected,
            is_geographic=is_geographic,
            width=src.width,
            height=src.height,
            count=src.count,
            dtype=str(src.dtypes[0]),
            bounds=bounds_tuple,
            transform=src.transform,
            resolution=(abs(src.transform.a), abs(src.transform.e)),
            gsd_meters=gsd,
            gsd_validation=validation,
            nodata=src.nodata,
            extra_tags=dict(src.tags()),
        )


def get_utm_epsg_for_coords(lon: float, lat: float) -> int:
    """Calculate the standard WGS 84 UTM EPSG code for a given longitude and latitude.

    Examples:
        - Kolkata, India (lon 88.36, lat 22.57) -> EPSG:32645 (UTM Zone 45N)
        - Florida, USA (lon -81.99, lat 29.68) -> EPSG:32617 (UTM Zone 17N)

    Args:
        lon: Longitude in degrees.
        lat: Latitude in degrees.

    Returns:
        int: EPSG code (e.g., 32645).
    """
    zone_number = int((lon + 180.0) / 6.0) % 60 + 1
    base = 32600 if lat >= 0 else 32700
    return base + zone_number


def reproject_raster(
    src_path: Union[str, Path],
    dst_path: Optional[Union[str, Path]] = None,
    target_epsg: Optional[Union[int, str]] = None,
    resampling: Resampling = Resampling.bilinear,
) -> str:
    """Reproject a raster to a local metric projected CRS (e.g. UTM) for accurate area calculations.

    If target_epsg is None and the raster is in a geographic CRS (e.g., EPSG:4326),
    the appropriate UTM zone EPSG code is automatically derived from the raster centroid.

    Args:
        src_path: Path to source raster.
        dst_path: Destination path. If None, appends `_projected.tif` to the source path.
        target_epsg: Optional explicit target EPSG code (e.g. 32645, "EPSG:32645").
        resampling: Resampling algorithm (default: Resampling.bilinear).

    Returns:
        str: Absolute path to the reprojected GeoTIFF.
    """
    src_path = str(src_path)
    if dst_path is None:
        p = Path(src_path)
        dst_path = str(p.parent / f"{p.stem}_projected.tif")
    else:
        dst_path = str(dst_path)

    with rasterio.open(src_path) as src:
        src_crs = src.crs

        if target_epsg is not None:
            dst_crs = CRS.from_user_input(target_epsg)
        else:
            if src_crs is not None and src_crs.is_projected:
                # Already projected; keep existing CRS
                dst_crs = src_crs
            else:
                # Geographic CRS or None: determine UTM zone from center coordinates
                center_c = src.width / 2.0
                center_r = src.height / 2.0
                center_x, center_y = src.transform * (center_c, center_r)

                if src_crs is not None and src_crs.is_geographic:
                    lon, lat = center_x, center_y
                else:
                    # Assume WGS84 coordinates if CRS was missing
                    lon, lat = center_x, center_y

                utm_code = get_utm_epsg_for_coords(lon, lat)
                dst_crs = CRS.from_epsg(utm_code)

        # Calculate destination transform and dimensions
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs, dst_crs, src.width, src.height, *src.bounds
        )

        dst_kwargs = src.meta.copy()
        dst_kwargs.update(
            {
                "crs": dst_crs,
                "transform": dst_transform,
                "width": dst_width,
                "height": dst_height,
            }
        )

        os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
        with rasterio.open(dst_path, "w", **dst_kwargs) as dst:
            for band in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band),
                    destination=rasterio.band(dst, band),
                    src_transform=src.transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=resampling,
                )

    return os.path.abspath(dst_path)


def parse_aoi(
    aoi_source: Union[str, Path, dict, gpd.GeoDataFrame],
    target_crs: Optional[Union[str, CRS]] = None,
) -> gpd.GeoDataFrame:
    """Parse KML or GeoJSON AOI polygon vector geometries.

    Args:
        aoi_source: File path (.kml, .geojson, .json), GeoJSON dict, or GeoDataFrame.
        target_crs: Optional CRS to reproject the GeoDataFrame to.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing parsed AOI polygons.
    """
    if isinstance(aoi_source, gpd.GeoDataFrame):
        gdf = aoi_source.copy()
    elif isinstance(aoi_source, dict):
        gdf = gpd.GeoDataFrame.from_features(aoi_source.get("features", [aoi_source]))
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
    elif isinstance(aoi_source, (str, Path)):
        source_path = str(aoi_source)
        ext = os.path.splitext(source_path)[1].lower()
        if ext == ".kml":
            gdf = gpd.read_file(source_path, driver="KML")
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
        else:
            gdf = gpd.read_file(source_path)
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
    else:
        raise ValueError(f"Unsupported AOI input type: {type(aoi_source)}")

    if target_crs is not None and gdf.crs is not None:
        gdf = gdf.to_crs(target_crs)

    return gdf


def clip_raster_to_aoi(
    raster_path: Union[str, Path],
    aoi: Union[gpd.GeoDataFrame, str, Path],
    output_path: Optional[Union[str, Path]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Clip a raster mosaic to the boundaries of an AOI polygon (KML/GeoJSON).

    Args:
        raster_path: Path to the input raster.
        aoi: GeoDataFrame or path to KML/GeoJSON defining the AOI.
        output_path: Destination path for clipped raster.

    Returns:
        tuple: (output_path, clipped_metadata_dict)
    """
    raster_path = str(raster_path)
    if output_path is None:
        p = Path(raster_path)
        output_path = str(p.parent / f"{p.stem}_clipped.tif")
    else:
        output_path = str(output_path)

    with rasterio.open(raster_path) as src:
        src_crs = src.crs

        # Ensure AOI matches raster CRS
        if not isinstance(aoi, gpd.GeoDataFrame):
            aoi_gdf = parse_aoi(aoi, target_crs=src_crs)
        else:
            aoi_gdf = aoi.to_crs(src_crs) if aoi.crs != src_crs else aoi

        geometries = [mapping(geom) for geom in aoi_gdf.geometry if geom is not None and not geom.is_empty]
        if not geometries:
            raise ValueError("No valid polygon geometries found in AOI for clipping.")

        clipped_image, clipped_transform = mask(src, geometries, crop=True)

        clipped_meta = src.meta.copy()
        clipped_meta.update(
            {
                "height": clipped_image.shape[1],
                "width": clipped_image.shape[2],
                "transform": clipped_transform,
            }
        )

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with rasterio.open(output_path, "w", **clipped_meta) as dst:
            dst.write(clipped_image)

    return os.path.abspath(output_path), clipped_meta
