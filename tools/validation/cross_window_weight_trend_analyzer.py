"""Analyze policy weight trends across validation windows."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from statistics import fmean
from typing import Any


class CrossWindowWeightTrendAnalyzer:
    """Analyze ordered final-weight changes across windows."""

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
        stable_tolerance: float = 0.005,
    ) -> None:
        if isinstance(
            stable_tolerance,
            bool,
        ):
            raise TypeError(
                "stable_tolerance must be numeric"
            )

        try:
            normalized = float(
                stable_tolerance
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise TypeError(
                "stable_tolerance must be numeric"
            ) from exc

        if not isfinite(normalized):
            raise ValueError(
                "stable_tolerance must be finite"
            )

        if normalized < 0.0:
            raise ValueError(
                "stable_tolerance must be greater "
                "than or equal to 0"
            )

        self._stable_tolerance = normalized

    @property
    def stable_tolerance(
        self,
    ) -> float:
        return self._stable_tolerance

    def analyze(
        self,
        report: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(
            report,
            Mapping,
        ):
            raise TypeError(
                "report must be a mapping"
            )

        windows = report.get(
            "windows"
        )

        if not isinstance(
            windows,
            list,
        ):
            raise TypeError(
                "windows must be a list"
            )

        policies = report.get(
            "policies"
        )

        if not isinstance(
            policies,
            Mapping,
        ):
            raise TypeError(
                "policies must be an object"
            )

        ordered_windows = sorted(
            windows,
            key=lambda item: (
                self._integer(
                    item,
                    "start_round",
                ),
                self._integer(
                    item,
                    "end_round",
                ),
            ),
        )

        analyzed_policies = {}

        for policy_name, policy in sorted(
            policies.items()
        ):
            if not isinstance(
                policy_name,
                str,
            ):
                raise TypeError(
                    "policy names must be strings"
                )

            if not isinstance(
                policy,
                Mapping,
            ):
                raise TypeError(
                    "policy values must be objects"
                )

            series = self._policy_series(
                policy=policy,
                ordered_windows=ordered_windows,
            )

            analyzed_policies[
                policy_name
            ] = {
                "window_count": len(series),
                "weights": {
                    field: self._analyze_values(
                        [
                            item["weights"][field]
                            for item in series
                        ]
                    )
                    for field in self.WEIGHT_FIELDS
                },
                "windows": series,
            }

        return {
            "schema_version": 1,
            "stable_tolerance": (
                self.stable_tolerance
            ),
            "window_count": len(
                ordered_windows
            ),
            "policy_count": len(
                analyzed_policies
            ),
            "policies": analyzed_policies,
        }

    def _policy_series(
        self,
        *,
        policy: Mapping[str, Any],
        ordered_windows: list[Any],
    ) -> list[dict[str, Any]]:
        policy_windows = policy.get(
            "windows"
        )

        if not isinstance(
            policy_windows,
            list,
        ):
            raise TypeError(
                "policy windows must be a list"
            )

        by_window = {}

        for item in policy_windows:
            if not isinstance(
                item,
                Mapping,
            ):
                raise TypeError(
                    "policy window rows must "
                    "be objects"
                )

            start_round = self._integer(
                item,
                "start_round",
            )
            end_round = self._integer(
                item,
                "end_round",
            )

            weights = item.get(
                "final_weights"
            )

            if not isinstance(
                weights,
                Mapping,
            ):
                raise TypeError(
                    "final_weights must be an object"
                )

            normalized_weights = {
                field: self._number(
                    weights,
                    field,
                )
                for field in self.WEIGHT_FIELDS
            }

            total = sum(
                normalized_weights.values()
            )

            if abs(total - 1.0) > 1e-9:
                raise ValueError(
                    "final weights must sum to 1.0"
                )

            by_window[
                (
                    start_round,
                    end_round,
                )
            ] = normalized_weights

        series = []

        for window in ordered_windows:
            start_round = self._integer(
                window,
                "start_round",
            )
            end_round = self._integer(
                window,
                "end_round",
            )

            weights = by_window.get(
                (
                    start_round,
                    end_round,
                )
            )

            if weights is None:
                continue

            series.append(
                {
                    "start_round": start_round,
                    "end_round": end_round,
                    "weights": weights,
                }
            )

        return series

    def _analyze_values(
        self,
        values: list[float],
    ) -> dict[str, Any]:
        if not values:
            return {
                "values": [],
                "first": None,
                "last": None,
                "net_change": None,
                "mean": None,
                "direction": "insufficient_data",
                "increase_steps": 0,
                "decrease_steps": 0,
                "stable_steps": 0,
            }

        changes = [
            current - previous
            for previous, current in zip(
                values,
                values[1:],
            )
        ]

        increase_steps = sum(
            change > self.stable_tolerance
            for change in changes
        )
        decrease_steps = sum(
            change < -self.stable_tolerance
            for change in changes
        )
        stable_steps = (
            len(changes)
            - increase_steps
            - decrease_steps
        )

        net_change = (
            values[-1] - values[0]
        )

        if len(values) < 2:
            direction = "insufficient_data"
        elif abs(
            net_change
        ) <= self.stable_tolerance:
            direction = "stable"
        elif net_change > 0.0:
            direction = "increasing"
        else:
            direction = "decreasing"

        return {
            "values": values,
            "first": values[0],
            "last": values[-1],
            "net_change": net_change,
            "mean": fmean(values),
            "direction": direction,
            "increase_steps": (
                increase_steps
            ),
            "decrease_steps": (
                decrease_steps
            ),
            "stable_steps": stable_steps,
        }

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

        normalized = float(value)

        if not isfinite(normalized):
            raise ValueError(
                f"{key} must be finite"
            )

        return normalized
