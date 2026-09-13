"""Canopy Core: Geospatial Tree Crown Detection, Tiling, and Canopy Area Pipeline."""

from canopy_core.ingest import (
    RasterMetadata,
    inspect_raster,
    calculate_gsd,
    validate_gsd,
    get_utm_epsg_for_coords,
    reproject_raster,
    parse_aoi,
    clip_raster_to_aoi,
)
from canopy_core.tiler import TileWindow, SlidingWindowTiler
from canopy_core.models.detector import TreeDetector
from canopy_core.models.area_engine import CanopyAreaEngine, CanopyMetrics
from canopy_core.models.trainer import TreeTrainer, TrainingConfig
from canopy_core.models.evaluator import TreeEvaluator, EvaluationResult

__version__ = "0.1.0"

__all__ = [
    "RasterMetadata",
    "inspect_raster",
    "calculate_gsd",
    "validate_gsd",
    "get_utm_epsg_for_coords",
    "reproject_raster",
    "parse_aoi",
    "clip_raster_to_aoi",
    "TileWindow",
    "SlidingWindowTiler",
    "TreeDetector",
    "CanopyAreaEngine",
    "CanopyMetrics",
    "TreeTrainer",
    "TrainingConfig",
    "TreeEvaluator",
    "EvaluationResult",
]

from .filters import SpectralMorphologicalFilter, FilterResult
from .models.sam_refiner import SAMCrownRefiner, SAMRefinementResult
