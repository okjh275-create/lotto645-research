"""Management APIs for Lotto645 Research Platform."""

from .doctor import run_doctor
from .status import collect_platform_status

__all__ = [
    "collect_platform_status",
    "run_doctor",
]
