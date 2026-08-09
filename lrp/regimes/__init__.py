"""Prediction-regime analysis APIs."""

from .detector import (
    RegimeDetector,
    RegimeDetectorConfig,
)
from .features import RegimeFeatureExtractor

from .contracts import (
    SUPPORTED_REGIMES,
    RegimeDecision,
    RegimeFeatureSnapshot,
)

__all__ = [
    "SUPPORTED_REGIMES",
    "RegimeDecision",
    "RegimeFeatureSnapshot",
    "RegimeFeatureExtractor",
    "RegimeDetector",
    "RegimeDetectorConfig",
]
