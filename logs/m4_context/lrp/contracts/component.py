"""Component contracts used by Project A platform assembly."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .exceptions import ComponentRegistrationError
from .versions import Version, parse_version


class ComponentKind(str, Enum):
    """Supported Project A component categories."""

    FOUNDATION = "foundation"
    STATISTICS = "statistics"
    CANDIDATE = "candidate"
    RESEARCH = "research"
    OPTIMIZER = "optimizer"
    REPORTING = "reporting"


class ReleaseChannel(str, Enum):
    """Supported component release channels."""

    STABLE = "stable"
    CANDIDATE = "candidate"
    DEVELOPMENT = "development"


@dataclass(frozen=True, slots=True, order=True)
class CapabilityRequirement:
    """A named and versioned capability requirement."""

    name: str
    version: int = 1

    def __post_init__(self) -> None:
        normalized = self.name.strip() if isinstance(self.name, str) else ""

        if not normalized:
            raise ComponentRegistrationError(
                "capability name must be a non-empty string"
            )

        parts = normalized.split(".")
        if any(not part for part in parts):
            raise ComponentRegistrationError(
                "capability name must not contain empty dotted sections"
            )

        if any(
            not part.replace("_", "").replace("-", "").isalnum()
            for part in parts
        ):
            raise ComponentRegistrationError(
                "capability name contains invalid characters"
            )

        if (
            isinstance(self.version, bool)
            or not isinstance(self.version, int)
            or self.version < 1
        ):
            raise ComponentRegistrationError(
                "capability version must be a positive integer"
            )

        object.__setattr__(self, "name", normalized)


@dataclass(frozen=True, slots=True)
class ComponentDescriptor:
    """Immutable metadata for one LRP-integrated component."""

    component_id: str
    display_name: str
    kind: ComponentKind
    version: str
    release_channel: ReleaseChannel
    capabilities: tuple[CapabilityRequirement, ...] = ()
    api_version: str | None = None

    def __post_init__(self) -> None:
        component_id = (
            self.component_id.strip()
            if isinstance(self.component_id, str)
            else ""
        )
        display_name = (
            self.display_name.strip()
            if isinstance(self.display_name, str)
            else ""
        )

        if not component_id:
            raise ComponentRegistrationError(
                "component_id must be a non-empty string"
            )

        allowed = "abcdefghijklmnopqrstuvwxyz0123456789-_."
        if any(character not in allowed for character in component_id.lower()):
            raise ComponentRegistrationError(
                "component_id contains invalid characters"
            )

        if not display_name:
            raise ComponentRegistrationError(
                "display_name must be a non-empty string"
            )

        if not isinstance(self.kind, ComponentKind):
            raise ComponentRegistrationError(
                "kind must be a ComponentKind"
            )

        if not isinstance(self.release_channel, ReleaseChannel):
            raise ComponentRegistrationError(
                "release_channel must be a ReleaseChannel"
            )

        parsed = parse_version(self.version)

        api_version = self.api_version
        if api_version is not None:
            api_version = api_version.strip()
            if not api_version:
                raise ComponentRegistrationError(
                    "api_version must be non-empty when provided"
                )
            parse_version(api_version)

        capabilities = tuple(sorted(set(self.capabilities)))

        object.__setattr__(self, "component_id", component_id)
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(self, "version", str(parsed))
        object.__setattr__(self, "api_version", api_version)
        object.__setattr__(self, "capabilities", capabilities)

    @property
    def parsed_version(self) -> Version:
        """Return the normalized component version."""

        return parse_version(self.version)

    def provides(self, requirement: CapabilityRequirement) -> bool:
        """Return whether this component provides a capability."""

        if not isinstance(requirement, CapabilityRequirement):
            raise ComponentRegistrationError(
                "requirement must be a CapabilityRequirement"
            )
        return requirement in self.capabilities

    def provides_all(
        self,
        requirements: Iterable[CapabilityRequirement],
    ) -> bool:
        """Return whether all supplied capabilities are provided."""

        return all(self.provides(item) for item in tuple(requirements))
