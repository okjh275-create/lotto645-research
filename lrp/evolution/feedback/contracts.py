"""Contracts for automated adaptive-learning feedback."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping


class AdaptiveAction(str, Enum):
    """Supported adaptive-learning actions."""

    KEEP = "keep"
    INCREASE = "increase"
    DECREASE = "decrease"
    ROLLBACK = "rollback"


@dataclass(frozen=True, slots=True)
class AdaptiveFeedback:
    """Observed validation evidence for one policy component."""

    policy_name: str
    component: str
    window_count: int
    total_round_count: int
    direction: str
    p_value: float
    significant: bool
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "policy_name",
            self._required_text(
                self.policy_name,
                "policy_name",
            ),
        )
        object.__setattr__(
            self,
            "component",
            self._required_text(
                self.component,
                "component",
            ),
        )
        object.__setattr__(
            self,
            "direction",
            self._required_text(
                self.direction,
                "direction",
            ),
        )

        if (
            isinstance(self.window_count, bool)
            or not isinstance(
                self.window_count,
                int,
            )
            or self.window_count < 1
        ):
            raise ValueError(
                "window_count must be an integer "
                "greater than or equal to 1"
            )

        if (
            isinstance(
                self.total_round_count,
                bool,
            )
            or not isinstance(
                self.total_round_count,
                int,
            )
            or self.total_round_count < 1
        ):
            raise ValueError(
                "total_round_count must be an integer "
                "greater than or equal to 1"
            )

        p_value = self._finite_number(
            self.p_value,
            "p_value",
        )

        if not 0.0 <= p_value <= 1.0:
            raise ValueError(
                "p_value must be between 0.0 and 1.0"
            )

        if not isinstance(
            self.significant,
            bool,
        ):
            raise TypeError(
                "significant must be boolean"
            )

        if not isinstance(
            self.metrics,
            Mapping,
        ):
            raise TypeError(
                "metrics must be a mapping"
            )

        normalized_metrics: dict[str, float] = {}

        for raw_key, raw_value in self.metrics.items():
            key = self._required_text(
                raw_key,
                "metric name",
            )
            normalized_metrics[key] = (
                self._finite_number(
                    raw_value,
                    f"metric '{key}'",
                )
            )

        object.__setattr__(
            self,
            "p_value",
            p_value,
        )
        object.__setattr__(
            self,
            "metrics",
            MappingProxyType(
                normalized_metrics
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "component": self.component,
            "window_count": self.window_count,
            "total_round_count": (
                self.total_round_count
            ),
            "direction": self.direction,
            "p_value": self.p_value,
            "significant": self.significant,
            "metrics": dict(self.metrics),
        }

    @staticmethod
    def _required_text(
        value: object,
        name: str,
    ) -> str:
        if not isinstance(value, str):
            raise TypeError(
                f"{name} must be a string"
            )

        normalized = value.strip()

        if not normalized:
            raise ValueError(
                f"{name} must not be empty"
            )

        return normalized

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


@dataclass(frozen=True, slots=True)
class AdaptiveDecision:
    """One proposed adaptive parameter decision."""

    component: str
    action: AdaptiveAction
    current_value: float
    proposed_value: float
    confidence: float
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "component",
            AdaptiveFeedback._required_text(
                self.component,
                "component",
            ),
        )
        object.__setattr__(
            self,
            "reason",
            AdaptiveFeedback._required_text(
                self.reason,
                "reason",
            ),
        )

        if not isinstance(
            self.action,
            AdaptiveAction,
        ):
            raise TypeError(
                "action must be an AdaptiveAction"
            )

        current_value = (
            AdaptiveFeedback._finite_number(
                self.current_value,
                "current_value",
            )
        )
        proposed_value = (
            AdaptiveFeedback._finite_number(
                self.proposed_value,
                "proposed_value",
            )
        )
        confidence = (
            AdaptiveFeedback._finite_number(
                self.confidence,
                "confidence",
            )
        )

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "confidence must be between "
                "0.0 and 1.0"
            )

        object.__setattr__(
            self,
            "current_value",
            current_value,
        )
        object.__setattr__(
            self,
            "proposed_value",
            proposed_value,
        )
        object.__setattr__(
            self,
            "confidence",
            confidence,
        )

    @property
    def delta(self) -> float:
        return (
            self.proposed_value
            - self.current_value
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "action": self.action.value,
            "current_value": (
                self.current_value
            ),
            "proposed_value": (
                self.proposed_value
            ),
            "delta": self.delta,
            "confidence": self.confidence,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class AdaptiveRecommendation:
    """Complete automated feedback recommendation."""

    recommendation_id: str
    created_at_utc: datetime
    feedback: AdaptiveFeedback
    decisions: tuple[
        AdaptiveDecision,
        ...,
    ]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "recommendation_id",
            AdaptiveFeedback._required_text(
                self.recommendation_id,
                "recommendation_id",
            ),
        )

        if not isinstance(
            self.created_at_utc,
            datetime,
        ):
            raise TypeError(
                "created_at_utc must be a datetime"
            )

        if self.created_at_utc.tzinfo is None:
            raise ValueError(
                "created_at_utc must be "
                "timezone-aware"
            )

        if not isinstance(
            self.feedback,
            AdaptiveFeedback,
        ):
            raise TypeError(
                "feedback must be an "
                "AdaptiveFeedback"
            )

        normalized_decisions = tuple(
            self.decisions
        )

        if not normalized_decisions:
            raise ValueError(
                "decisions must not be empty"
            )

        if any(
            not isinstance(
                decision,
                AdaptiveDecision,
            )
            for decision in normalized_decisions
        ):
            raise TypeError(
                "decisions must contain only "
                "AdaptiveDecision values"
            )

        components = [
            decision.component
            for decision in normalized_decisions
        ]

        if len(components) != len(
            set(components)
        ):
            raise ValueError(
                "decision components must be unique"
            )

        object.__setattr__(
            self,
            "created_at_utc",
            self.created_at_utc.astimezone(
                timezone.utc
            ),
        )
        object.__setattr__(
            self,
            "decisions",
            normalized_decisions,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "recommendation_id": (
                self.recommendation_id
            ),
            "created_at_utc": (
                self.created_at_utc
                .isoformat()
            ),
            "feedback": (
                self.feedback.as_dict()
            ),
            "decisions": [
                decision.as_dict()
                for decision in self.decisions
            ],
        }
