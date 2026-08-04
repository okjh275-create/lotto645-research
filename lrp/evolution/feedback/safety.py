"""Safety validation for adaptive recommendations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Any

from lrp.evolution.feedback.contracts import (
    AdaptiveAction,
    AdaptiveRecommendation,
)


@dataclass(frozen=True, slots=True)
class AdaptiveSafetyResult:
    """Result of validating an adaptive recommendation."""

    approved: bool
    proposed_weights: Mapping[str, float]
    safe_weights: Mapping[str, float]
    violations: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(
            self.approved,
            bool,
        ):
            raise TypeError(
                "approved must be boolean"
            )

        object.__setattr__(
            self,
            "proposed_weights",
            MappingProxyType(
                dict(self.proposed_weights)
            ),
        )
        object.__setattr__(
            self,
            "safe_weights",
            MappingProxyType(
                dict(self.safe_weights)
            ),
        )
        object.__setattr__(
            self,
            "violations",
            tuple(self.violations),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "proposed_weights": dict(
                self.proposed_weights
            ),
            "safe_weights": dict(
                self.safe_weights
            ),
            "violations": list(
                self.violations
            ),
        }


class AdaptiveSafetyGuard:
    """Validate and normalize adaptive weight changes."""

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
        max_delta: float = 0.02,
        minimum_weight: float = 0.03,
        maximum_weight: float = 0.50,
        allow_rollback: bool = False,
    ) -> None:
        self._max_delta = self._positive_number(
            max_delta,
            "max_delta",
        )
        self._minimum_weight = self._bounded_number(
            minimum_weight,
            "minimum_weight",
        )
        self._maximum_weight = self._bounded_number(
            maximum_weight,
            "maximum_weight",
        )

        if (
            self._minimum_weight
            >= self._maximum_weight
        ):
            raise ValueError(
                "minimum_weight must be less "
                "than maximum_weight"
            )

        if (
            self._minimum_weight
            * len(self.WEIGHT_FIELDS)
            >= 1.0
        ):
            raise ValueError(
                "minimum_weight is too large "
                "for the weight count"
            )

        if not isinstance(
            allow_rollback,
            bool,
        ):
            raise TypeError(
                "allow_rollback must be boolean"
            )

        self._allow_rollback = allow_rollback

    @property
    def max_delta(self) -> float:
        return self._max_delta

    @property
    def minimum_weight(self) -> float:
        return self._minimum_weight

    @property
    def maximum_weight(self) -> float:
        return self._maximum_weight

    @property
    def allow_rollback(self) -> bool:
        return self._allow_rollback

    def validate(
        self,
        *,
        recommendation: AdaptiveRecommendation,
        current_weights: Mapping[str, float],
    ) -> AdaptiveSafetyResult:
        if not isinstance(
            recommendation,
            AdaptiveRecommendation,
        ):
            raise TypeError(
                "recommendation must be an "
                "AdaptiveRecommendation"
            )

        current = self._normalize_input_weights(
            current_weights
        )
        raw_proposed = dict(current)
        violations: list[str] = []

        decisions_by_field = {
            decision.component: decision
            for decision in recommendation.decisions
        }

        unknown = sorted(
            set(decisions_by_field)
            - set(self.WEIGHT_FIELDS)
        )

        if unknown:
            raise ValueError(
                "unknown decision components: "
                + ", ".join(unknown)
            )

        for field, decision in (
            decisions_by_field.items()
        ):
            current_value = current[field]

            if abs(
                decision.current_value
                - current_value
            ) > 1e-9:
                violations.append(
                    f"{field}: current value mismatch"
                )

            if (
                decision.action
                is AdaptiveAction.ROLLBACK
                and not self.allow_rollback
            ):
                violations.append(
                    f"{field}: rollback is not allowed"
                )

            delta = (
                decision.proposed_value
                - current_value
            )

            if abs(delta) > (
                self.max_delta + 1e-12
            ):
                violations.append(
                    f"{field}: delta exceeds "
                    f"maximum {self.max_delta}"
                )

            raw_proposed[field] = (
                decision.proposed_value
            )

        self._check_bounds(
            raw_proposed,
            violations,
            stage="raw",
        )

        normalized = self._normalize_total(
            raw_proposed
        )

        self._check_bounds(
            normalized,
            violations,
            stage="normalized",
        )

        for field in self.WEIGHT_FIELDS:
            normalized_delta = (
                normalized[field]
                - current[field]
            )

            if abs(normalized_delta) > (
                self.max_delta + 1e-12
            ):
                violations.append(
                    f"{field}: normalized delta "
                    f"exceeds maximum "
                    f"{self.max_delta}"
                )

        unique_violations = tuple(
            dict.fromkeys(violations)
        )

        approved = not unique_violations

        return AdaptiveSafetyResult(
            approved=approved,
            proposed_weights=normalized,
            safe_weights=(
                normalized
                if approved
                else current
            ),
            violations=unique_violations,
        )

    def _check_bounds(
        self,
        weights: Mapping[str, float],
        violations: list[str],
        *,
        stage: str,
    ) -> None:
        for field in self.WEIGHT_FIELDS:
            value = weights[field]

            if value < (
                self.minimum_weight - 1e-12
            ):
                violations.append(
                    f"{field}: {stage} value "
                    "is below minimum weight"
                )

            if value > (
                self.maximum_weight + 1e-12
            ):
                violations.append(
                    f"{field}: {stage} value "
                    "exceeds maximum weight"
                )

    def _normalize_input_weights(
        self,
        values: Mapping[str, float],
    ) -> dict[str, float]:
        if not isinstance(
            values,
            Mapping,
        ):
            raise TypeError(
                "current_weights must be a mapping"
            )

        expected = set(
            self.WEIGHT_FIELDS
        )
        actual = set(values)

        missing = sorted(
            expected - actual
        )
        unknown = sorted(
            actual - expected
        )

        if missing:
            raise ValueError(
                "missing current weights: "
                + ", ".join(missing)
            )

        if unknown:
            raise ValueError(
                "unknown current weights: "
                + ", ".join(unknown)
            )

        normalized = {
            field: self._finite_number(
                values[field],
                field,
            )
            for field in self.WEIGHT_FIELDS
        }

        if any(
            value < 0.0
            for value in normalized.values()
        ):
            raise ValueError(
                "current weights must be "
                "greater than or equal to 0"
            )

        total = sum(
            normalized.values()
        )

        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                "current weights must sum to 1.0"
            )

        return normalized

    def _normalize_total(
        self,
        weights: Mapping[str, float],
    ) -> dict[str, float]:
        total = sum(
            weights[field]
            for field in self.WEIGHT_FIELDS
        )

        if (
            not isfinite(total)
            or total <= 0.0
        ):
            raise ValueError(
                "proposed weight total must be "
                "finite and greater than 0"
            )

        return {
            field: weights[field] / total
            for field in self.WEIGHT_FIELDS
        }

    @staticmethod
    def _finite_number(
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

        if not isfinite(normalized):
            raise ValueError(
                f"{name} must be finite"
            )

        return normalized

    @classmethod
    def _positive_number(
        cls,
        value: object,
        name: str,
    ) -> float:
        normalized = cls._finite_number(
            value,
            name,
        )

        if normalized <= 0.0:
            raise ValueError(
                f"{name} must be greater than 0"
            )

        return normalized

    @classmethod
    def _bounded_number(
        cls,
        value: object,
        name: str,
    ) -> float:
        normalized = cls._finite_number(
            value,
            name,
        )

        if not 0.0 <= normalized <= 1.0:
            raise ValueError(
                f"{name} must be between "
                "0.0 and 1.0"
            )

        return normalized
