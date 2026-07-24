"""Public adapter layer for Lotto645 Research Platform."""

from .candidate import CandidateAdapter
from .foundation import FoundationAdapter
from .signal_bridge import (
    SignalBridgeConfig,
    StatisticsSignalSnapshot,
    build_statistics_signals,
)
from .statistics import StatisticsAdapter

__all__ = [
    "CandidateAdapter",
    "FoundationAdapter",
    "SignalBridgeConfig",
    "StatisticsAdapter",
    "StatisticsSignalSnapshot",
    "build_statistics_signals",
]
