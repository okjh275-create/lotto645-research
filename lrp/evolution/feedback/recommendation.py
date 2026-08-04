"""Generate conservative adaptive recommendations from feedback."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from math import isfinite
from math import isfinite
from typing import Any

from lrp.evolution.feedback.contracts import (
    AdaptiveAction,
    AdaptiveDecision,
    AdaptiveFeedback,
    AdaptiveRecommendation,
)


class AdaptiveRecommendationEngine:
    """Convert validated feedback into parameter recommendations."""

    COMPONENT_TO_FIELD = {
        "hot": "hot_weight",
        "cold": "cold_weight",
        "gap": "gap_weight",
        "trend": "trend_weight",
        "transition": "transition_weight",
        "learning": "learning_weight",
        "adaptive": "adaptive_weight",
    }

    def __init__(
        self,
        *,
        step_size: float = 0.01,
        rollback_threshold: float = -0.05,
    ) -> None:
        self._step_size = self._positive_number(
            step_size,
            "step_size",
        )
        self._rollback_threshold = self._finite_number(
            rollback_threshold,
            "rollback_threshold",
        )

        if self._rollback_threshold >= 0.0:
            raise ValueError(
                "rollback_threshold must be less than 0"
            )

    @property
    def step_size(self) -> float:
        return self._step_size

    @property
    def rollback_threshold(self) -> float:
        return self._rollback_threshold

    def recommend(
        self,
        *,
        recommendation_id: str,
        feedback: Sequence[AdaptiveFeedback],
        current_weights: Mapping[str, float],
        created_at_utc: datetime | None = None,
    ) -> AdaptiveRecommendation:
        normalized_feedback = tuple(feedback)

        if not normalized_feedback:
            raise ValueError(
                "feedback must not be empty"
            )

        if any(
            not isinstance(
                item,
                AdaptiveFeedback,
            )
            for item in normalized_feedback
        ):
            raise TypeError(
                "feedback must contain only "
                "AdaptiveFeedback values"
            )

        policy_names = {
            item.policy_name
            for item in normalized_feedback
        }

        if len(policy_names) != 1:
            raise ValueError(
                "feedback must belong to one policy"
            )

        components = [
            item.component
            for item in normalized_feedback
        ]

        if len(components) != len(
            set(components)
        ):
            raise ValueError(
                "feedback components must be unique"
            )

        weights = self._normalize_weights(
            current_weights
        )

        decisions = tuple(
            self._decision(
                item=item,
                current_weights=weights,
            )
            for item in normalized_feedback
        )

        timestamp = (
            created_at_utc
            if created_at_utc is not None
            else datetime.now(timezone.utc)
        )

        return AdaptiveRecommendation(
            recommendation_id=(
                recommendation_id
            ),
            created_at_utc=timestamp,
            feedback=self._summary_feedback(
                normalized_feedback
            ),
            decisions=decisions,
        )

    def _decision(
        self,
        *,
        item: AdaptiveFeedback,
        current_weights: Mapping[str, float],
    ) -> AdaptiveDecision:
        field = self.COMPONENT_TO_FIELD.get(
            item.component
        )

        if field is None:
            raise ValueError(
                f"unknown feedback component: "
                f"{item.component}"
            )

        current_value = current_weights[field]

        practical_code = item.metrics.get(
            "practical_direction_code",
            0.0,
        )
        best_code = item.metrics.get(
            "best_direction_code",
            0.0,
        )
        practical_delta = item.metrics.get(
            "practical_hit_mean_delta",
            0.0,
        )
        best_delta = item.metrics.get(
            "best_hit_mean_delta",
            0.0,
        )

        combined_delta = (
            practical_delta + best_delta
        ) / 2.0

        if (
            practical_delta
            <= self.rollback_threshold
            and best_delta
            <= self.rollback_threshold
        ):
            return AdaptiveDecision(
                component=field,
                action=AdaptiveAction.ROLLBACK,
                current_value=current_value,
                proposed_value=current_value,
                confidence=self._confidence(item),
                reason=(
                    "Both practical and best hit deltas "
                    "crossed the rollback threshold."
                ),
            )

        if not item.significant:
            return AdaptiveDecision(
                component=field,
                action=AdaptiveAction.KEEP,
                current_value=current_value,
                proposed_value=current_value,
                confidence=self._confidence(item),
                reason=(
                    "Cross-window evidence was not "
                    "statistically significant."
                ),
            )

        evidence_code = practical_code + best_code

        if evidence_code > 0.0:
            action = self._positive_action(
                item.direction
            )
        elif evidence_code < 0.0:
            action = self._negative_action(
                item.direction
            )
        elif combined_delta > 0.0:
            action = self._positive_action(
                item.direction
            )
        elif combined_delta < 0.0:
            action = self._negative_action(
                item.direction
            )
        else:
            action = AdaptiveAction.KEEP

        proposed_value = self._proposed_value(
            current_value=current_value,
            action=action,
        )

        return AdaptiveDecision(
            component=field,
            action=action,
            current_value=current_value,
            proposed_value=proposed_value,
            confidence=self._confidence(item),
            reason=self._reason(
                item=item,
                action=action,
            ),
        )

    def _positive_action(
        self,
        trend_direction: str,
    ) -> AdaptiveAction:
        if trend_direction == "decreasing":
            return AdaptiveAction.DECREASE

        if trend_direction in {
            "increasing",
            "stable",
            "insufficient_data",
        }:
            return AdaptiveAction.INCREASE

        raise ValueError(
            f"unknown trend direction: "
            f"{trend_direction}"
        )

    def _negative_action(
        self,
        trend_direction: str,
    ) -> AdaptiveAction:
        if trend_direction == "increasing":
            return AdaptiveAction.DECREASE

        if trend_direction in {
            "decreasing",
            "stable",
            "insufficient_data",
        }:
            return AdaptiveAction.DECREASE

        raise ValueError(
            f"unknown trend direction: "
            f"{trend_direction}"
        )

    def _proposed_value(
        self,
        *,
        current_value: float,
        action: AdaptiveAction,
    ) -> float:
        if action is AdaptiveAction.INCREASE:
            return min(
                1.0,
                current_value + self.step_size,
            )

        if action is AdaptiveAction.DECREASE:
            return max(
                0.0,
                current_value - self.step_size,
            )

        return current_value

    @staticmethod
    def _confidence(
        feedback: AdaptiveFeedback,
    ) -> float:
        significance_confidence = (
            1.0 - feedback.p_value
        )

        window_factor = min(
            1.0,
            feedback.window_count / 3.0,
        )

        round_factor = min(
            1.0,
            feedback.total_round_count / 300.0,
        )

        return max(
            0.0,
            min(
                1.0,
                significance_confidence
                * window_factor
                * round_factor,
            ),
        )

    @staticmethod
    def _reason(
        *,
        item: AdaptiveFeedback,
        action: AdaptiveAction,
    ) -> str:
        return (
            f"Action {action.value} selected from "
            f"{item.direction} weight trend with "
            f"p_value={item.p_value:.6f}."
        )

    @staticmethod
    def _summary_feedback(
        feedback: tuple[
            AdaptiveFeedback,
            ...,
        ],
    ) -> AdaptiveFeedback:
        first = feedback[0]

        return AdaptiveFeedback(
            policy_name=first.policy_name,
            component="portfolio",
            window_count=first.window_count,
            total_round_count=(
                first.total_round_count
            ),
            direction="mixed",
            p_value=min(
                item.p_value
                for item in feedback
            ),
            significant=any(
                item.significant
                for item in feedback
            ),
            metrics={
                "component_count": float(
                    len(feedback)
                ),
                "significant_component_count": float(
                    sum(
                        item.significant
                        for item in feedback
                    )
                ),
            },
        )

    def _normalize_weights(
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
            self.COMPONENT_TO_FIELD.values()
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
            key: self._finite_number(
                values[key],
                key,
            )
            for key in sorted(expected)
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
