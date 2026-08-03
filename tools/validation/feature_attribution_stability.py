"""Block stability analysis for lagged feature attribution."""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path
from statistics import fmean
from typing import Any


COMPONENTS = (
    "hot",
    "cold",
    "gap",
    "trend",
    "transition",
)

OUTCOMES = (
    "best_hit_delta",
    "practical_hit_delta",
)


def pearson(
    left: list[float],
    right: list[float],
) -> float:
    if len(left) != len(right):
        raise ValueError(
            "correlation inputs must have equal length"
        )

    if len(left) < 2:
        return 0.0

    left_mean = fmean(left)
    right_mean = fmean(right)

    left_delta = [
        value - left_mean
        for value in left
    ]
    right_delta = [
        value - right_mean
        for value in right
    ]

    numerator = sum(
        first * second
        for first, second in zip(
            left_delta,
            right_delta,
        )
    )

    denominator = sqrt(
        sum(value * value for value in left_delta)
        * sum(value * value for value in right_delta)
    )

    if denominator == 0.0:
        return 0.0

    return numerator / denominator


def load_rows(
    *,
    rounds_path: Path,
    learning_root: Path,
) -> list[dict[str, Any]]:
    replay_rows = [
        json.loads(line)
        for line in rounds_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    signals = {}

    for path in learning_root.glob(
        "review-*.json"
    ):
        payload = json.loads(
            path.read_text(encoding="utf-8")
        )
        metadata = payload["metadata"]

        signals[payload["round_no"]] = {
            component: float(
                metadata[
                    f"feature_signal_{component}"
                ]
            )
            for component in COMPONENTS
        }

    replay_rows.sort(
        key=lambda row: row["round_no"]
    )

    lagged = []

    for previous, current in zip(
        replay_rows,
        replay_rows[1:],
    ):
        previous_round = previous["round_no"]
        current_round = current["round_no"]

        if current_round != previous_round + 1:
            continue

        lagged.append(
            {
                "round_no": current_round,
                **signals[previous_round],
                **{
                    outcome: float(
                        current[outcome]
                    )
                    for outcome in OUTCOMES
                },
            }
        )

    return lagged


def analyze_blocks(
    rows: list[dict[str, Any]],
    *,
    block_size: int = 25,
) -> dict[str, Any]:
    if block_size < 2:
        raise ValueError(
            "block_size must be at least 2"
        )

    blocks = []

    for start in range(
        0,
        len(rows),
        block_size,
    ):
        block = rows[
            start:start + block_size
        ]

        if len(block) < 2:
            continue

        correlations = {}

        for component in COMPONENTS:
            correlations[component] = {}

            feature_values = [
                row[component]
                for row in block
            ]

            for outcome in OUTCOMES:
                correlations[component][
                    outcome
                ] = pearson(
                    feature_values,
                    [
                        row[outcome]
                        for row in block
                    ],
                )

        blocks.append(
            {
                "block": len(blocks) + 1,
                "start_round": (
                    block[0]["round_no"]
                ),
                "end_round": (
                    block[-1]["round_no"]
                ),
                "observation_count": len(block),
                "correlations": correlations,
            }
        )

    stability = {}

    for component in COMPONENTS:
        stability[component] = {}

        for outcome in OUTCOMES:
            values = [
                block["correlations"][
                    component
                ][outcome]
                for block in blocks
            ]

            positive_count = sum(
                value > 0.0
                for value in values
            )
            negative_count = sum(
                value < 0.0
                for value in values
            )

            stability[component][outcome] = {
                "block_correlations": values,
                "mean_correlation": (
                    fmean(values)
                    if values
                    else 0.0
                ),
                "positive_blocks": (
                    positive_count
                ),
                "negative_blocks": (
                    negative_count
                ),
                "consistent_direction": (
                    positive_count == len(values)
                    or negative_count == len(values)
                ),
            }

    return {
        "schema_version": 1,
        "lagged_observation_count": len(rows),
        "block_size": block_size,
        "block_count": len(blocks),
        "blocks": blocks,
        "stability": stability,
    }
