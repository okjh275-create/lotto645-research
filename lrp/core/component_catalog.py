"""Built-in component catalog for Lotto645 Research Platform."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from lrp import (
    CANDIDATE_REQUIRED_VERSION,
    FOUNDATION_REQUIRED_VERSION,
    STATISTICS_REQUIRED_API_VERSION,
)
from lrp.contracts import (
    CapabilityRequirement,
    ComponentDescriptor,
    ComponentKind,
    ComponentNotFoundError,
    ComponentRegistrationError,
    ReleaseChannel,
)


FOUNDATION_COMPONENT_ID = "project-b.foundation"
STATISTICS_COMPONENT_ID = "project-c.statistics"
CANDIDATE_COMPONENT_ID = "project-d.candidate"


FOUNDATION_CAPABILITIES = (
    CapabilityRequirement("foundation.compatibility"),
    CapabilityRequirement("foundation.execution"),
    CapabilityRequirement("foundation.plugin_registry"),
    CapabilityRequirement("foundation.serialization"),
)

STATISTICS_CAPABILITIES = (
    CapabilityRequirement("statistics.analysis"),
    CapabilityRequirement("statistics.features"),
    CapabilityRequirement("statistics.rolling_backtest"),
    CapabilityRequirement("statistics.serialization"),
    CapabilityRequirement("statistics.uncertainty"),
)

CANDIDATE_CAPABILITIES = (
    CapabilityRequirement("candidate.diversity"),
    CapabilityRequirement("candidate.generation"),
    CapabilityRequirement("candidate.practical_selection"),
    CapabilityRequirement("candidate.ranking"),
    CapabilityRequirement("candidate.replay"),
    CapabilityRequirement("candidate.risk"),
    CapabilityRequirement("candidate.scoring"),
    CapabilityRequirement("candidate.statistics_adapter"),
)


def build_builtin_descriptors() -> tuple[ComponentDescriptor, ...]:
    """Return the canonical Project B, C and D descriptors."""

    return (
        ComponentDescriptor(
            component_id=FOUNDATION_COMPONENT_ID,
            display_name="Lotto645 Research Foundation",
            kind=ComponentKind.FOUNDATION,
            version=FOUNDATION_REQUIRED_VERSION,
            release_channel=ReleaseChannel.STABLE,
            capabilities=FOUNDATION_CAPABILITIES,
        ),
        ComponentDescriptor(
            component_id=STATISTICS_COMPONENT_ID,
            display_name="Lotto645 Statistics Engine",
            kind=ComponentKind.STATISTICS,
            version="1.0.0",
            api_version=STATISTICS_REQUIRED_API_VERSION,
            release_channel=ReleaseChannel.STABLE,
            capabilities=STATISTICS_CAPABILITIES,
        ),
        ComponentDescriptor(
            component_id=CANDIDATE_COMPONENT_ID,
            display_name="Lotto645 Candidate Engine",
            kind=ComponentKind.CANDIDATE,
            version="0.8.0.dev0",
            release_channel=ReleaseChannel.CANDIDATE,
            capabilities=CANDIDATE_CAPABILITIES,
        ),
    )


@dataclass(frozen=True, slots=True)
class ComponentCatalog:
    """Deterministic descriptor catalog owned by Project A."""

    descriptors: tuple[ComponentDescriptor, ...]

    def __post_init__(self) -> None:
        normalized = tuple(
            sorted(
                self.descriptors,
                key=lambda item: item.component_id,
            )
        )

        component_ids = tuple(
            descriptor.component_id
            for descriptor in normalized
        )
        if len(component_ids) != len(set(component_ids)):
            raise ComponentRegistrationError(
                "component IDs must be unique"
            )

        kinds = tuple(descriptor.kind for descriptor in normalized)
        if len(kinds) != len(set(kinds)):
            raise ComponentRegistrationError(
                "only one built-in component per kind is supported"
            )

        object.__setattr__(self, "descriptors", normalized)

    @classmethod
    def builtin(cls) -> "ComponentCatalog":
        """Create the standard B/C/D platform catalog."""

        return cls(build_builtin_descriptors())

    def get(self, component_id: str) -> ComponentDescriptor:
        """Return a component by identifier."""

        normalized = (
            component_id.strip()
            if isinstance(component_id, str)
            else ""
        )

        for descriptor in self.descriptors:
            if descriptor.component_id == normalized:
                return descriptor

        raise ComponentNotFoundError(
            f"component is not present in catalog: {component_id}"
        )

    def get_by_kind(self, kind: ComponentKind) -> ComponentDescriptor:
        """Return the component registered for one kind."""

        if not isinstance(kind, ComponentKind):
            raise ComponentRegistrationError(
                "kind must be a ComponentKind"
            )

        for descriptor in self.descriptors:
            if descriptor.kind is kind:
                return descriptor

        raise ComponentNotFoundError(
            f"component kind is not present in catalog: {kind.value}"
        )

    def providers(
        self,
        requirement: CapabilityRequirement,
    ) -> tuple[ComponentDescriptor, ...]:
        """Return deterministic providers of a capability."""

        matches = (
            descriptor
            for descriptor in self.descriptors
            if descriptor.provides(requirement)
        )
        return tuple(
            sorted(matches, key=lambda item: item.component_id)
        )

    def require(
        self,
        requirements: Iterable[CapabilityRequirement],
    ) -> dict[CapabilityRequirement, ComponentDescriptor]:
        """Resolve every required capability to one provider."""

        resolved: dict[
            CapabilityRequirement,
            ComponentDescriptor,
        ] = {}

        for requirement in tuple(requirements):
            providers = self.providers(requirement)
            if not providers:
                raise ComponentNotFoundError(
                    "no component provides capability: "
                    f"{requirement.name}@{requirement.version}"
                )
            resolved[requirement] = providers[0]

        return resolved
