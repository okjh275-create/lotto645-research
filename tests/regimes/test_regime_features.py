from __future__ import annotations

import pytest

from lrp.contracts import ContractError
from lrp.regimes import (
    RegimeFeatureExtractor,
    RegimeFeatureSnapshot,
)


def snapshot(
    *,
    low_frequency: float = 0.4,
    high_frequency: float = 0.6,
) -> dict[str, object]:
    signals: dict[int, dict[str, float]] = {}

    for number in range(1, 46):
        frequency = (
            low_frequency
            if number <= 22
            else high_frequency
        )

        signals[number] = {
            "recency": 0.6,
            "frequency": frequency,
            "gap_reversion": 0.4,
            "pair_graph": (number - 1) / 44,
        }

    return {"signals": signals}


def test_extracts_global_features() -> None:
    result = RegimeFeatureExtractor().extract(
        snapshot()
    )

    assert isinstance(result, RegimeFeatureSnapshot)
    assert result.average_recency == pytest.approx(0.6)
    assert result.average_gap_reversion == pytest.approx(0.4)
    assert result.pair_density == pytest.approx(0.5)
    assert result.low_band_ratio == pytest.approx(0.4)
    assert result.high_band_ratio == pytest.approx(0.6)


def test_band_ratios_sum_to_one() -> None:
    result = RegimeFeatureExtractor().extract(
        snapshot(
            low_frequency=0.8,
            high_frequency=0.2,
        )
    )

    assert result.low_band_ratio == pytest.approx(0.8)
    assert result.high_band_ratio == pytest.approx(0.2)
    assert (
        result.low_band_ratio
        + result.high_band_ratio
    ) == pytest.approx(1.0)


def test_zero_frequency_bands_are_balanced() -> None:
    result = RegimeFeatureExtractor().extract(
        snapshot(
            low_frequency=0.0,
            high_frequency=0.0,
        )
    )

    assert result.low_band_ratio == 0.5
    assert result.high_band_ratio == 0.5


def test_rejects_missing_number() -> None:
    payload = snapshot()
    del payload["signals"][45]

    with pytest.raises(ContractError):
        RegimeFeatureExtractor().extract(payload)


def test_rejects_missing_signal_field() -> None:
    payload = snapshot()
    del payload["signals"][10]["pair_graph"]

    with pytest.raises(ContractError):
        RegimeFeatureExtractor().extract(payload)


def test_rejects_out_of_range_signal() -> None:
    payload = snapshot()
    payload["signals"][10]["recency"] = 1.2

    with pytest.raises(ContractError):
        RegimeFeatureExtractor().extract(payload)
