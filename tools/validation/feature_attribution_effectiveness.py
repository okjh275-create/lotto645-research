"""Feature-attribution effectiveness analysis for historical replay."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from math import isfinite, sqrt
from pathlib import Path
from statistics import fmean, pstdev
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
    "probability_l1_delta",
    "changed_set_count",
)


def analyze_feature_attribution(
    *,
    rounds_path: Path,
    learning_root: Path,
) -> dict[str, Any]:
    """Join replay rows and learning snapshots, then analyze signals."""

    rows = _load_rounds(rounds_path)
    signals_by_round = _load_signals(
        learning_root
    )

    joined = []

    for row in rows:
        round_no = _integer(
            row,
            "round_no",
        )

        if round_no not in signals_by_round:
            raise ValueError(
                "missing feature signals for "
                f"round {round_no}"
            )

        joined.append(
            {
                **row,
                **signals_by_round[round_no],
            }
        )

    contemporaneous = {
        component: _component_analysis(
            rows=joined,
            component=component,
        )
        for component in COMPONENTS
    }

    lagged_rows = []

    for previous, current in zip(
        joined,
        joined[1:],
    ):
        previous_round = _integer(
            previous,
            "round_no",
        )
        current_round = _integer(
            current,
            "round_no",
        )

        if current_round != previous_round + 1:
            continue

        lagged_rows.append(
            {
                "round_no": current_round,
                **{
                    (
                        f"feature_signal_"
                        f"{component}"
                    ): previous[
                        f"feature_signal_"
                        f"{component}"
                    ]
                    for component in COMPONENTS
                },
                **{
                    outcome: current[outcome]
                    for outcome in OUTCOMES
                },
            }
        )

    lagged = {
        component: _component_analysis(
            rows=lagged_rows,
            component=component,
        )
        for component in COMPONENTS
    }

    return {
        "schema_version": 1,
        "round_count": len(joined),
        "lagged_round_count": len(
            lagged_rows
        ),
        "components": list(COMPONENTS),
        "outcomes": list(OUTCOMES),
        "contemporaneous": contemporaneous,
        "lagged_one_round": lagged,
        "interpretation": {
            "contemporaneous": (
                "Descriptive attribution only; "
                "winning numbers from the same "
                "round were used to construct "
                "the feature signal."
            ),
            "lagged_one_round": (
                "Associates the previous round's "
                "feature signal with the next "
                "round's outcome. This is more "
                "relevant to adaptive-policy "
                "validation, but is not proof "
                "of causality."
            ),
        },
    }


def write_feature_attribution_report(
    *,
    report: Mapping[str, Any],
    output: Path,
) -> Path:
    """Write a deterministic JSON report."""

    if not isinstance(report, Mapping):
        raise TypeError(
            "report must be a mapping"
        )

    output = Path(output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            dict(report),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return output


def _load_rounds(
    path: Path,
) -> list[dict[str, Any]]:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    rows = []

    for line_number, raw_line in enumerate(
        path.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):
        line = raw_line.strip()

        if not line:
            continue

        payload = json.loads(line)

        if not isinstance(payload, dict):
            raise TypeError(
                "replay row must be an object; "
                f"line={line_number}"
            )

        for field_name in (
            "round_no",
            *OUTCOMES,
        ):
            if field_name not in payload:
                raise ValueError(
                    "missing replay field: "
                    f"{field_name}; "
                    f"line={line_number}"
                )

        rows.append(payload)

    if not rows:
        raise ValueError(
            "replay rounds must not be empty"
        )

    rows.sort(
        key=lambda row: _integer(
            row,
            "round_no",
        )
    )

    return rows


def _load_signals(
    root: Path,
) -> dict[int, dict[str, float]]:
    root = Path(root)

    if not root.is_dir():
        raise FileNotFoundError(root)

    result: dict[
        int,
        dict[str, float],
    ] = {}

    for path in sorted(
        root.glob("review-*.json")
    ):
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(payload, Mapping):
            raise TypeError(
                f"{path.name} must contain "
                "a JSON object"
            )

        round_no = _integer(
            payload,
            "round_no",
        )

        metadata = payload.get(
            "metadata"
        )

        if not isinstance(
            metadata,
            Mapping,
        ):
            raise TypeError(
                f"{path.name} metadata must "
                "be a mapping"
            )

        signals = {}

        for component in COMPONENTS:
            key = (
                f"feature_signal_{component}"
            )
            signals[key] = _finite_number(
                metadata,
                key,
                minimum=-1.0,
                maximum=1.0,
            )

        if round_no in result:
            raise ValueError(
                "duplicate learning snapshot "
                f"for round {round_no}"
            )

        result[round_no] = signals

    if not result:
        raise ValueError(
            "no learning snapshots found"
        )

    return result


def _component_analysis(
    *,
    rows: Sequence[Mapping[str, Any]],
    component: str,
) -> dict[str, Any]:
    signal_key = (
        f"feature_signal_{component}"
    )

    signals = [
        _finite_number(
            row,
            signal_key,
            minimum=-1.0,
            maximum=1.0,
        )
        for row in rows
    ]

    positive_indexes = [
        index
        for index, value in enumerate(
            signals
        )
        if value > 0.0
    ]
    negative_indexes = [
        index
        for index, value in enumerate(
            signals
        )
        if value < 0.0
    ]
    zero_indexes = [
        index
        for index, value in enumerate(
            signals
        )
        if value == 0.0
    ]

    correlations = {}
    grouped_outcomes = {}

    for outcome in OUTCOMES:
        outcome_values = [
            _finite_number(
                row,
                outcome,
            )
            for row in rows
        ]

        correlations[outcome] = (
            _pearson(
                signals,
                outcome_values,
            )
        )

        grouped_outcomes[outcome] = {
            "positive_signal_mean": (
                _indexed_mean(
                    outcome_values,
                    positive_indexes,
                )
            ),
            "negative_signal_mean": (
                _indexed_mean(
                    outcome_values,
                    negative_indexes,
                )
            ),
            "zero_signal_mean": (
                _indexed_mean(
                    outcome_values,
                    zero_indexes,
                )
            ),
            "positive_minus_negative": (
                _mean_difference(
                    outcome_values,
                    positive_indexes,
                    negative_indexes,
                )
            ),
        }

    return {
        "observation_count": len(signals),
        "signal_mean": (
            fmean(signals)
            if signals
            else 0.0
        ),
        "signal_stddev": (
            pstdev(signals)
            if len(signals) > 1
            else 0.0
        ),
        "signal_min": (
            min(signals)
            if signals
            else 0.0
        ),
        "signal_max": (
            max(signals)
            if signals
            else 0.0
        ),
        "positive_count": len(
            positive_indexes
        ),
        "negative_count": len(
            negative_indexes
        ),
        "zero_count": len(
            zero_indexes
        ),
        "non_zero_count": (
            len(positive_indexes)
            + len(negative_indexes)
        ),
        "correlations": correlations,
        "grouped_outcomes": (
            grouped_outcomes
        ),
    }


def _pearson(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right):
        raise ValueError(
            "correlation inputs must have "
            "equal length"
        )

    if len(left) < 2:
        return 0.0

    left_mean = fmean(left)
    right_mean = fmean(right)

    left_deviation = [
        value - left_mean
        for value in left
    ]
    right_deviation = [
        value - right_mean
        for value in right
    ]

    numerator = sum(
        first * second
        for first, second in zip(
            left_deviation,
            right_deviation,
        )
    )

    denominator = sqrt(
        sum(
            value * value
            for value in left_deviation
        )
        * sum(
            value * value
            for value in right_deviation
        )
    )

    if denominator == 0.0:
        return 0.0

    result = numerator / denominator

    return max(
        -1.0,
        min(1.0, result),
    )


def _indexed_mean(
    values: Sequence[float],
    indexes: Sequence[int],
) -> float | None:
    if not indexes:
        return None

    return fmean(
        values[index]
        for index in indexes
    )


def _mean_difference(
    values: Sequence[float],
    positive_indexes: Sequence[int],
    negative_indexes: Sequence[int],
) -> float | None:
    positive = _indexed_mean(
        values,
        positive_indexes,
    )
    negative = _indexed_mean(
        values,
        negative_indexes,
    )

    if positive is None or negative is None:
        return None

    return positive - negative


def _integer(
    values: Mapping[str, Any],
    key: str,
) -> int:
    if key not in values:
        raise ValueError(
            f"missing field: {key}"
        )

    value = values[key]

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
    ):
        raise TypeError(
            f"{key} must be an integer"
        )

    return value


def _finite_number(
    values: Mapping[str, Any],
    key: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if key not in values:
        raise ValueError(
            f"missing field: {key}"
        )

    value = values[key]

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

    if (
        minimum is not None
        and normalized < minimum
    ):
        raise ValueError(
            f"{key} must be greater than "
            f"or equal to {minimum}"
        )

    if (
        maximum is not None
        and normalized > maximum
    ):
        raise ValueError(
            f"{key} must be less than "
            f"or equal to {maximum}"
        )

    return normalized
