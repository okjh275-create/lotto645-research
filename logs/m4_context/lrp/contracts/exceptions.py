"""Exceptions raised by the Lotto645 Research Platform contract layer."""

from __future__ import annotations


class LrpError(Exception):
    """Base exception for all Project A platform errors."""


class ContractError(LrpError, ValueError):
    """Raised when a component violates a platform contract."""


class VersionError(ContractError):
    """Raised when a version string is invalid or unsupported."""


class CompatibilityError(ContractError):
    """Raised when a component is incompatible with the platform."""


class ComponentRegistrationError(ContractError):
    """Raised when component registration metadata is invalid."""


class ComponentNotFoundError(ContractError, LookupError):
    """Raised when a requested platform component cannot be resolved."""
