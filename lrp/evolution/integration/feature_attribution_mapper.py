from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any


class FeatureAttributionMapper:
    """Extract feature attribution signals from a prediction payload."""

    COMPONENTS = (
        "hot",
        "cold",
        "gap",
        "trend",
        "transition",
    )

    def map(
        self,
        prediction_payload: Mapping[str, Any],
        winning_numbers: tuple[int, ...],
    ) -> dict[str, float]:

        probabilities = (
            prediction_payload[
                "probability_vector"
            ]["probabilities"]
        )

        lookup = {
            item["number"]: item
            for item in probabilities
        }

        result: dict[str, float] = {}

        for component in self.COMPONENTS:

            winner_values = [
                lookup[number]["components"][component]
                for number in winning_numbers
            ]

            all_values = [
                item["components"][component]
                for item in probabilities
            ]

            winner_mean = (
                sum(winner_values)
                / len(winner_values)
            )

            overall_mean = (
                sum(all_values)
                / len(all_values)
            )

            delta = (
                winner_mean
                - overall_mean
            )

            if not isfinite(delta):
                raise ValueError(
                    "non-finite attribution"
                )

            result[component] = max(
                -1.0,
                min(1.0, delta),
            )

        return result
