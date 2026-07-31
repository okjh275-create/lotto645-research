from __future__ import annotations

from dataclasses import dataclass

from lrp.evolution.contracts.models import (
    AdaptiveWeightProfile,
)
from lrp.evolution.policies.policy import (
    AdaptivePolicyDecision,
)
from lrp.evolution.storage.snapshot import (
    EvolutionSnapshot,
)


@dataclass(frozen=True, slots=True)
class EvolutionRunResult:
    """Outcome of one EvolutionEngine execution."""

    decision: AdaptivePolicyDecision
    previous_profile: AdaptiveWeightProfile | None
    snapshot: EvolutionSnapshot | None

    def __post_init__(self) -> None:
        if not isinstance(
            self.decision,
            AdaptivePolicyDecision,
        ):
            raise TypeError(
                "decision must be an "
                "AdaptivePolicyDecision"
            )

        if (
            self.previous_profile is not None
            and not isinstance(
                self.previous_profile,
                AdaptiveWeightProfile,
            )
        ):
            raise TypeError(
                "previous_profile must be an "
                "AdaptiveWeightProfile or None"
            )

        if (
            self.snapshot is not None
            and not isinstance(
                self.snapshot,
                EvolutionSnapshot,
            )
        ):
            raise TypeError(
                "snapshot must be an "
                "EvolutionSnapshot or None"
            )

        if self.decision.applied and self.snapshot is None:
            raise ValueError(
                "applied decisions require a snapshot"
            )

        if (
            not self.decision.applied
            and self.snapshot is not None
        ):
            raise ValueError(
                "rejected decisions must not "
                "contain a snapshot"
            )

        if (
            self.snapshot is not None
            and self.snapshot.profile
            != self.decision.profile
        ):
            raise ValueError(
                "snapshot profile must match "
                "decision profile"
            )

    @property
    def applied(self) -> bool:
        return self.decision.applied

    @property
    def rejected(self) -> bool:
        return self.decision.rejected

    @property
    def persisted(self) -> bool:
        return self.snapshot is not None

    @property
    def profile(self) -> AdaptiveWeightProfile:
        return self.decision.profile

    @property
    def revision(self) -> int:
        return self.profile.revision

    @property
    def previous_revision(self) -> int | None:
        if self.previous_profile is None:
            return None

        return self.previous_profile.revision

    @property
    def reasons(self) -> tuple[str, ...]:
        return self.decision.reasons