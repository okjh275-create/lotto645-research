"""Plan the next adaptive-weight profile revision."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from lrp.evolution.contracts import (
    AdaptiveWeightProfile,
)
from lrp.evolution.feedback.safety import (
    AdaptiveSafetyResult,
)


@dataclass(frozen=True, slots=True)
class AdaptiveProfileUpdatePlan:
    """Planned adaptive profile revision."""

    approved: bool
    source_revision: int
    target_revision: int
    profile: AdaptiveWeightProfile
    violations: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "source_revision": (
                self.source_revision
            ),
            "target_revision": (
                self.target_revision
            ),
            "profile": {
                "hot_weight": (
                    self.profile.hot_weight
                ),
                "cold_weight": (
                    self.profile.cold_weight
                ),
                "gap_weight": (
                    self.profile.gap_weight
                ),
                "trend_weight": (
                    self.profile.trend_weight
                ),
                "transition_weight": (
                    self.profile.transition_weight
                ),
                "learning_weight": (
                    self.profile.learning_weight
                ),
                "adaptive_weight": (
                    self.profile.adaptive_weight
                ),
                "confidence": (
                    self.profile.confidence
                ),
                "sample_size": (
                    self.profile.sample_size
                ),
                "revision": (
                    self.profile.revision
                ),
                "generated_at": (
                    self.profile.generated_at
                    .astimezone(timezone.utc)
                    .isoformat()
                ),
            },
            "violations": list(
                self.violations
            ),
        }


class AdaptiveProfileUpdatePlanner:
    """Create a candidate next revision from safety-approved weights."""

    def plan(
        self,
        *,
        current_profile: AdaptiveWeightProfile,
        safety_result: AdaptiveSafetyResult,
        confidence: float | None = None,
        sample_size: int | None = None,
        generated_at: datetime | None = None,
    ) -> AdaptiveProfileUpdatePlan:
        if not isinstance(
            current_profile,
            AdaptiveWeightProfile,
        ):
            raise TypeError(
                "current_profile must be an "
                "AdaptiveWeightProfile"
            )

        if not isinstance(
            safety_result,
            AdaptiveSafetyResult,
        ):
            raise TypeError(
                "safety_result must be an "
                "AdaptiveSafetyResult"
            )

        timestamp = (
            generated_at
            if generated_at is not None
            else datetime.now(timezone.utc)
        )

        if not isinstance(
            timestamp,
            datetime,
        ):
            raise TypeError(
                "generated_at must be a datetime "
                "or None"
            )

        if timestamp.tzinfo is None:
            raise ValueError(
                "generated_at must be "
                "timezone-aware"
            )

        target_confidence = (
            current_profile.confidence
            if confidence is None
            else self._confidence(
                confidence
            )
        )

        target_sample_size = (
            current_profile.sample_size
            if sample_size is None
            else self._sample_size(
                sample_size
            )
        )

        source_revision = (
            current_profile.revision
        )

        target_revision = (
            source_revision + 1
            if safety_result.approved
            else source_revision
        )

        weights = (
            safety_result.safe_weights
        )

        profile = AdaptiveWeightProfile(
            hot_weight=weights[
                "hot_weight"
            ],
            cold_weight=weights[
                "cold_weight"
            ],
            gap_weight=weights[
                "gap_weight"
            ],
            trend_weight=weights[
                "trend_weight"
            ],
            transition_weight=weights[
                "transition_weight"
            ],
            learning_weight=weights[
                "learning_weight"
            ],
            adaptive_weight=weights[
                "adaptive_weight"
            ],
            confidence=target_confidence,
            sample_size=target_sample_size,
            revision=target_revision,
            generated_at=timestamp,
        )

        return AdaptiveProfileUpdatePlan(
            approved=safety_result.approved,
            source_revision=source_revision,
            target_revision=target_revision,
            profile=profile,
            violations=(
                safety_result.violations
            ),
        )

    @staticmethod
    def _confidence(
        value: object,
    ) -> float:
        if isinstance(value, bool):
            raise TypeError(
                "confidence must be numeric"
            )

        try:
            normalized = float(value)
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise TypeError(
                "confidence must be numeric"
            ) from exc

        if not 0.0 <= normalized <= 1.0:
            raise ValueError(
                "confidence must be between "
                "0.0 and 1.0"
            )

        return normalized

    @staticmethod
    def _sample_size(
        value: object,
    ) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
        ):
            raise TypeError(
                "sample_size must be an integer"
            )

        if value < 0:
            raise ValueError(
                "sample_size must be greater "
                "than or equal to 0"
            )

        return value
