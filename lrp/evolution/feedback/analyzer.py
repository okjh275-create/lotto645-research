"""Convert cross-window validation reports into adaptive feedback."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from lrp.evolution.feedback.contracts import (
    AdaptiveFeedback,
)


class AdaptiveFeedbackAnalyzer:
    """Build component feedback from cross-window report evidence."""

    COMPONENT_MAP = {
        "hot_weight": "hot",
        "cold_weight": "cold",
        "gap_weight": "gap",
        "trend_weight": "trend",
        "transition_weight": "transition",
        "learning_weight": "learning",
        "adaptive_weight": "adaptive",
    }

    def analyze(
        self,
        report: Mapping[str, Any],
        *,
        policy_name: str,
    ) -> tuple[AdaptiveFeedback, ...]:
        if not isinstance(report, Mapping):
            raise TypeError(
                "report must be a mapping"
            )

        normalized_policy = self._required_text(
            policy_name,
            "policy_name",
        )

        policies = self._mapping(
            report,
            "policies",
        )
        trends = self._mapping(
            report,
            "weight_trends",
        )
        significance = self._mapping(
            report,
            "significance",
        )

        policy = policies.get(
            normalized_policy
        )

        if not isinstance(policy, Mapping):
            raise ValueError(
                f"unknown policy: {normalized_policy}"
            )

        trend_policies = self._mapping(
            trends,
            "policies",
        )
        significance_policies = self._mapping(
            significance,
            "policies",
        )

        policy_trends = trend_policies.get(
            normalized_policy
        )
        policy_significance = (
            significance_policies.get(
                normalized_policy
            )
        )

        if not isinstance(
            policy_trends,
            Mapping,
        ):
            raise ValueError(
                "policy trend data is missing"
            )

        if not isinstance(
            policy_significance,
            Mapping,
        ):
            raise ValueError(
                "policy significance data is missing"
            )

        weight_trends = self._mapping(
            policy_trends,
            "weights",
        )
        practical = self._mapping(
            policy_significance,
            "practical",
        )
        best = self._mapping(
            policy_significance,
            "best",
        )

        window_count = self._integer(
            policy,
            "window_count",
        )
        total_round_count = self._integer(
            policy,
            "total_round_count",
        )

        practical_p = self._number(
            practical,
            "p_value",
        )
        best_p = self._number(
            best,
            "p_value",
        )

        practical_significant = self._boolean(
            practical,
            "significant",
        )
        best_significant = self._boolean(
            best,
            "significant",
        )

        practical_direction = self._text(
            practical,
            "direction",
        )
        best_direction = self._text(
            best,
            "direction",
        )

        feedback_items = []

        for weight_field, component in (
            self.COMPONENT_MAP.items()
        ):
            trend = weight_trends.get(
                weight_field
            )

            if not isinstance(
                trend,
                Mapping,
            ):
                raise ValueError(
                    f"missing trend data: {weight_field}"
                )

            trend_direction = self._text(
                trend,
                "direction",
            )

            feedback_items.append(
                AdaptiveFeedback(
                    policy_name=normalized_policy,
                    component=component,
                    window_count=window_count,
                    total_round_count=(
                        total_round_count
                    ),
                    direction=trend_direction,
                    p_value=min(
                        practical_p,
                        best_p,
                    ),
                    significant=(
                        practical_significant
                        or best_significant
                    ),
                    metrics={
                        "best_hit_mean_delta": (
                            self._number(
                                policy,
                                "best_hit_mean_delta",
                            )
                        ),
                        "practical_hit_mean_delta": (
                            self._number(
                                policy,
                                "practical_hit_mean_delta",
                            )
                        ),
                        "average_probability_l1_delta": (
                            self._number(
                                policy,
                                "average_probability_l1_delta",
                            )
                        ),
                        "average_changed_set_count": (
                            self._number(
                                policy,
                                "average_changed_set_count",
                            )
                        ),
                        "trend_first": (
                            self._optional_number(
                                trend,
                                "first",
                            )
                        ),
                        "trend_last": (
                            self._optional_number(
                                trend,
                                "last",
                            )
                        ),
                        "trend_net_change": (
                            self._optional_number(
                                trend,
                                "net_change",
                            )
                        ),
                        "practical_adaptive_wins": (
                            float(
                                self._integer(
                                    practical,
                                    "adaptive_wins",
                                )
                            )
                        ),
                        "practical_noop_wins": (
                            float(
                                self._integer(
                                    practical,
                                    "noop_wins",
                                )
                            )
                        ),
                        "best_adaptive_wins": (
                            float(
                                self._integer(
                                    best,
                                    "adaptive_wins",
                                )
                            )
                        ),
                        "best_noop_wins": (
                            float(
                                self._integer(
                                    best,
                                    "noop_wins",
                                )
                            )
                        ),
                        "practical_direction_code": (
                            self._direction_code(
                                practical_direction
                            )
                        ),
                        "best_direction_code": (
                            self._direction_code(
                                best_direction
                            )
                        ),
                    },
                )
            )

        return tuple(feedback_items)

    @staticmethod
    def _direction_code(
        direction: str,
    ) -> float:
        if direction == "adaptive_better":
            return 1.0

        if direction == "noop_better":
            return -1.0

        if direction == "tie":
            return 0.0

        raise ValueError(
            f"unknown significance direction: {direction}"
        )

    @staticmethod
    def _mapping(
        values: Mapping[str, Any],
        key: str,
    ) -> Mapping[str, Any]:
        value = values.get(key)

        if not isinstance(value, Mapping):
            raise TypeError(
                f"{key} must be an object"
            )

        return value

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

    @classmethod
    def _text(
        cls,
        values: Mapping[str, Any],
        key: str,
    ) -> str:
        return cls._required_text(
            values.get(key),
            key,
        )

    @staticmethod
    def _integer(
        values: Mapping[str, Any],
        key: str,
    ) -> int:
        value = values.get(key)

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
        ):
            raise TypeError(
                f"{key} must be an integer"
            )

        return value

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

        return float(value)

    @classmethod
    def _optional_number(
        cls,
        values: Mapping[str, Any],
        key: str,
    ) -> float:
        value = values.get(key)

        if value is None:
            return 0.0

        return cls._number(
            values,
            key,
        )

    @staticmethod
    def _boolean(
        values: Mapping[str, Any],
        key: str,
    ) -> bool:
        value = values.get(key)

        if not isinstance(value, bool):
            raise TypeError(
                f"{key} must be boolean"
            )

        return value
