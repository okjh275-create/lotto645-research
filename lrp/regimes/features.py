"""Aggregate number-level statistics signals into regime features."""

from __future__ import annotations

import math
from collections.abc import Mapping
from statistics import fmean, pvariance
from typing import Any

from lrp.contracts import ContractError

from .contracts import RegimeFeatureSnapshot


_NUMBERS = tuple(range(1, 46))
_LOW = tuple(range(1, 23))
_HIGH = tuple(range(23, 46))
_FIELDS = (
    "recency",
    "frequency",
    "gap_reversion",
    "pair_graph",
)


def _finite_unit(
    value: object,
    *,
    field: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise ContractError(
            f"{field} must be numeric"
        )

    result = float(value)

    if not math.isfinite(result):
        raise ContractError(
            f"{field} must be finite"
        )

    if not 0.0 <= result <= 1.0:
        raise ContractError(
            f"{field} must be between 0 and 1"
        )

    return result


def _signals_from_snapshot(
    snapshot: object,
) -> Mapping[Any, Any]:
    if isinstance(snapshot, Mapping):
        signals = snapshot.get("signals")
    else:
        signals = getattr(
            snapshot,
            "signals",
            None,
        )

    if not isinstance(signals, Mapping):
        raise ContractError(
            "snapshot.signals must be a mapping"
        )

    return signals


def _row(
    signals: Mapping[Any, Any],
    number: int,
) -> Mapping[Any, Any]:
    value = signals.get(number)

    if value is None:
        value = signals.get(str(number))

    if not isinstance(value, Mapping):
        raise ContractError(
            f"signals are missing number {number}"
        )

    return value


def _field_values(
    signals: Mapping[Any, Any],
    field: str,
) -> dict[int, float]:
    values: dict[int, float] = {}

    for number in _NUMBERS:
        row = _row(signals, number)

        if field not in row:
            raise ContractError(
                f"signals[{number}] is missing {field}"
            )

        values[number] = _finite_unit(
            row[field],
            field=f"signals[{number}].{field}",
        )

    return values


def _relative_band_ratio(
    frequency: Mapping[int, float],
    band: tuple[int, ...],
) -> float:
    low_mean = fmean(
        frequency[number]
        for number in _LOW
    )
    high_mean = fmean(
        frequency[number]
        for number in _HIGH
    )

    total = low_mean + high_mean

    if math.isclose(total, 0.0):
        return 0.5

    selected = (
        low_mean
        if band is _LOW
        else high_mean
    )

    return selected / total


class RegimeFeatureExtractor:
    """Build global regime features from normalized number signals."""

    def extract(
        self,
        snapshot: object,
    ) -> RegimeFeatureSnapshot:
        signals = _signals_from_snapshot(snapshot)

        values = {
            field: _field_values(
                signals,
                field,
            )
            for field in _FIELDS
        }

        recency = values["recency"]
        frequency = values["frequency"]
        gaps = values["gap_reversion"]
        pairs = values["pair_graph"]

        average_recency = fmean(
            recency.values()
        )
        average_frequency = fmean(
            frequency.values()
        )
        average_gap_reversion = fmean(
            gaps.values()
        )
        pair_density = fmean(
            pairs.values()
        )

        frequency_dispersion = pvariance(
            frequency.values()
        )
        recency_variance = pvariance(
            recency.values()
        )
        pair_variance = pvariance(
            pairs.values()
        )

        low_band_ratio = _relative_band_ratio(
            frequency,
            _LOW,
        )
        high_band_ratio = _relative_band_ratio(
            frequency,
            _HIGH,
        )

        return RegimeFeatureSnapshot(
            average_recency=average_recency,
            average_frequency=average_frequency,
            average_gap_reversion=(
                average_gap_reversion
            ),
            pair_density=pair_density,
            frequency_dispersion=(
                frequency_dispersion
            ),
            recency_variance=recency_variance,
            pair_variance=pair_variance,
            low_band_ratio=low_band_ratio,
            high_band_ratio=high_band_ratio,
        )
