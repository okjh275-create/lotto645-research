"""Public contracts for Lotto645 Research Platform."""

from .component import (
    CapabilityRequirement,
    ComponentDescriptor,
    ComponentKind,
    ReleaseChannel,
)
from .exceptions import (
    CompatibilityError,
    ComponentNotFoundError,
    ComponentRegistrationError,
    ContractError,
    LrpError,
    VersionError,
)
from .versions import (
    Version,
    is_major_compatible,
    parse_version,
    same_release_line,
)

__all__ = [
    "CapabilityRequirement",
    "CompatibilityError",
    "ComponentDescriptor",
    "ComponentKind",
    "ComponentNotFoundError",
    "ComponentRegistrationError",
    "ContractError",
    "LrpError",
    "ReleaseChannel",
    "Version",
    "VersionError",
    "is_major_compatible",
    "parse_version",
    "same_release_line",
]
