"""Public package boundary for Lotto645 Research Platform.

Project A is the orchestration and application layer of the Lotto645
Research Platform. Statistical calculations and candidate generation
remain delegated to independently versioned components through adapters.
"""

from __future__ import annotations

PROJECT_NAME = "Lotto645 Research Platform"
PROJECT_ROLE = "Project A"
PLATFORM_VERSION = (4, 0, 0)
__version__ = ".".join(str(part) for part in PLATFORM_VERSION)

FOUNDATION_REQUIRED_VERSION = "1.0.0"
STATISTICS_REQUIRED_API_VERSION = "1.0"
CANDIDATE_REQUIRED_VERSION = "0.8.0"

__all__ = [
    "CANDIDATE_REQUIRED_VERSION",
    "FOUNDATION_REQUIRED_VERSION",
    "PLATFORM_VERSION",
    "PROJECT_NAME",
    "PROJECT_ROLE",
    "STATISTICS_REQUIRED_API_VERSION",
    "__version__",
]
