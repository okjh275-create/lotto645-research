"""Plan safe adaptive profile rollbacks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from typing import Any

from lrp.evolution.contracts import (
    AdaptiveWeightProfile,
)
from lrp.evolution.feedback.repository import (
    AdaptiveAutomationRepository,
)


@dataclass(frozen=True, slots=True)
class AdaptiveRollbackDiff:
    """Difference between current and rollback-target weights."""

    component: str
    current_value: float
    target_value: float

    @property
    def delta(self) -> float:
        return self.target_value - self.current_value

    def as_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "current_value": self.current_value,
            "target_value": self.target_value,
            "delta": self.delta,
        }


@dataclass(frozen=True, slots=True)
class AdaptiveRollbackPlan:
    """Immutable plan for restoring a historical profile."""

    source_revision: int
    rollback_revision: int
    target_revision: int
    profile: AdaptiveWeightProfile
    differences: tuple[
        AdaptiveRollbackDiff,
        ...,
    ]
    changed_component_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_revision": self.source_revision,
            "rollback_revision": (
                self.rollback_revision
            ),
            "target_revision": self.target_revision,
            "profile": {
                "hot_weight": self.profile.hot_weight,
                "cold_weight": self.profile.cold_weight,
                "gap_weight": self.profile.gap_weight,
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
                "confidence": self.profile.confidence,
                "sample_size": self.profile.sample_size,
                "revision": self.profile.revision,
                "generated_at": (
                    self.profile.generated_at
                    .astimezone(timezone.utc)
                    .isoformat()
                ),
            },
            "differences": [
                item.as_dict()
                for item in self.differences
            ],
            "changed_component_count": (
                self.changed_component_count
            ),
        }


class AdaptiveRollbackManager:
    """Create a new profile revision from historical weights."""

    WEIGHT_FIELDS = (
        "hot_weight",
        "cold_weight",
        "gap_weight",
        "trend_weight",
        "transition_weight",
        "learning_weight",
        "adaptive_weight",
    )

    def __init__(
        self,
        *,
        repository: AdaptiveAutomationRepository,
    ) -> None:
        if not isinstance(
            repository,
            AdaptiveAutomationRepository,
        ):
            raise TypeError(
                "repository must be an "
                "AdaptiveAutomationRepository"
            )

        self._repository = repository

    @property
    def repository(
        self,
    ) -> AdaptiveAutomationRepository:
        return self._repository

    def plan(
        self,
        *,
        current_profile: AdaptiveWeightProfile,
        rollback_revision: int,
        generated_at: datetime | None = None,
        confidence: float | None = None,
        sample_size: int | None = None,
    ) -> AdaptiveRollbackPlan:
        if not isinstance(
            current_profile,
            AdaptiveWeightProfile,
        ):
            raise TypeError(
                "current_profile must be an "
                "AdaptiveWeightProfile"
            )

        rollback_revision = self._revision(
            rollback_revision,
            "rollback_revision",
        )

        latest = self.repository.latest_profile()

        if latest is None:
            raise RuntimeError(
                "adaptive profile repository is empty"
            )

        repository_revision = self._integer(
            latest,
            "target_revision",
        )

        if (
            repository_revision
            != current_profile.revision
        ):
            raise RuntimeError(
                "current profile revision does not "
                "match repository head: "
                f"current={current_profile.revision}, "
                f"repository={repository_revision}"
            )

        if rollback_revision >= (
            current_profile.revision
        ):
            raise ValueError(
                "rollback_revision must be less "
                "than the current revision"
            )

        historical = (
            self.repository
            .load_profile_revision(
                rollback_revision
            )
        )

        historical_profile = historical.get(
            "profile"
        )

        if not isinstance(
            historical_profile,
            Mapping,
        ):
            raise TypeError(
                "historical profile must be an object"
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
                "generated_at must be timezone-aware"
            )

        target_confidence = (
            current_profile.confidence
            if confidence is None
            else self._bounded_number(
                confidence,
                "confidence",
            )
        )

        target_sample_size = (
            current_profile.sample_size
            if sample_size is None
            else self._sample_size(
                sample_size
            )
        )

        target_revision = (
            current_profile.revision + 1
        )

        restored_weights = {
            field: self._number(
                historical_profile,
                field,
            )
            for field in self.WEIGHT_FIELDS
        }

        total = sum(
            restored_weights.values()
        )

        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                "historical weights must sum to 1.0"
            )

        profile = AdaptiveWeightProfile(
            **restored_weights,
            confidence=target_confidence,
            sample_size=target_sample_size,
            revision=target_revision,
            generated_at=timestamp,
        )

        differences = tuple(
            AdaptiveRollbackDiff(
                component=field,
                current_value=float(
                    getattr(
                        current_profile,
                        field,
                    )
                ),
                target_value=float(
                    restored_weights[field]
                ),
            )
            for field in self.WEIGHT_FIELDS
        )

        changed_component_count = sum(
            abs(item.delta) > 1e-12
            for item in differences
        )

        return AdaptiveRollbackPlan(
            source_revision=(
                current_profile.revision
            ),
            rollback_revision=rollback_revision,
            target_revision=target_revision,
            profile=profile,
            differences=differences,
            changed_component_count=(
                changed_component_count
            ),
        )

    @staticmethod
    def _revision(
        value: object,
        name: str,
    ) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
        ):
            raise TypeError(
                f"{name} must be an integer"
            )

        if value < 0:
            raise ValueError(
                f"{name} must be greater than "
                "or equal to 0"
            )

        return value

    @classmethod
    def _integer(
        cls,
        values: Mapping[str, Any],
        key: str,
    ) -> int:
        return cls._revision(
            values.get(key),
            key,
        )

    @staticmethod
    def _number(
        values: Mapping[str, Any],
        key: str,
    ) -> float:
        value = values.get(key)

        if (
            isinstance(value, bool)
            or not isinstance(
                value,
                (int, float),
            )
        ):
            raise TypeError(
                f"{key} must be numeric"
            )

        normalized = float(value)

        if not isfinite(normalized):
            raise ValueError(
                f"{key} must be finite"
            )

        return normalized

    @classmethod
    def _bounded_number(
        cls,
        value: object,
        name: str,
    ) -> float:
        if isinstance(value, bool):
            raise TypeError(
                f"{name} must be numeric"
            )

        try:
            normalized = float(value)
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise TypeError(
                f"{name} must be numeric"
            ) from exc

        if (
            not isfinite(normalized)
            or not 0.0 <= normalized <= 1.0
        ):
            raise ValueError(
                f"{name} must be between "
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
