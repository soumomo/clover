"""Post-processing and contextual filters for tree crown detection."""
from .weed_filter import SpectralMorphologicalFilter, FilterResult

__all__ = ["SpectralMorphologicalFilter", "FilterResult"]
