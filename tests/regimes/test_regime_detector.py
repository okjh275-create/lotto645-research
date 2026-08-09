from __future__ import annotations

import pytest

from lrp.contracts import ContractError
from lrp.regimes import (
    RegimeDetector,
    RegimeDecision,
    RegimeFeatureSnapshot,
)


def make_features(**overrides: float) -> RegimeFeatureSnapshot:
    values = {
        "average_recency": 0.50,
        "average_frequency": 0.50,
        "average_gap_reversion": 0.50,
        "pair_density": 0.50,
        "frequency_dispersion": 0.20,
        "recency_variance": 0.20,
        "pair_variance": 0.20,
        "low_band_ratio": 0.50,
        "high_band_ratio": 0.50,
    }
    values.update(overrides)
    return RegimeFeatureSnapshot(**values)


def test_detect_returns_regime_decision() -> None:
    result = RegimeDetector().detect(
        make_features()
    )

    assert isinstance(result, RegimeDecision)
    assert 0.0 <= result.confidence <= 1.0
    assert result.primary in result.scores


def test_detects_gap_recovery_pressure() -> None:
    result = RegimeDetector().detect(
        make_features(
            average_recency=0.20,
            average_gap_reversion=0.90,
            frequency_dispersion=0.70,
            recency_variance=0.65,
            pair_density=0.15,
            pair_variance=0.10,
        )
    )

    assert result.scores["gap_recovery"] > 0.65


def test_detects_high_band_expansion_score() -> None:
    result = RegimeDetector().detect(
        make_features(
            low_band_ratio=0.20,
            high_band_ratio=0.80,
            average_frequency=0.70,
            pair_density=0.60,
        )
    )

    assert (
        result.scores["high_band_expansion"]
        > result.scores["low_band_expansion"]
    )


def test_detects_low_band_expansion_score() -> None:
    result = RegimeDetector().detect(
        make_features(
            low_band_ratio=0.80,
            high_band_ratio=0.20,
            average_frequency=0.70,
            pair_density=0.60,
        )
    )

    assert (
        result.scores["low_band_expansion"]
        > result.scores["high_band_expansion"]
    )


def test_balanced_state_has_strong_neutral_score() -> None:
    result = RegimeDetector().detect(
        make_features(
            low_band_ratio=0.50,
            high_band_ratio=0.50,
            average_recency=0.50,
            average_gap_reversion=0.45,
            pair_variance=0.10,
        )
    )

    assert result.scores["neutral"] >= 0.65


def test_ambiguous_state_exposes_mixed_score() -> None:
    result = RegimeDetector().detect(
        make_features(
            average_recency=0.50,
            average_frequency=0.50,
            average_gap_reversion=0.50,
            pair_density=0.50,
            frequency_dispersion=0.30,
            recency_variance=0.30,
            pair_variance=0.30,
            low_band_ratio=0.50,
            high_band_ratio=0.50,
        )
    )

    assert result.scores["mixed"] >= 0.0
    assert result.scores["mixed"] <= 1.0


def test_rejects_invalid_input() -> None:
    with pytest.raises(ContractError):
        RegimeDetector().detect(object())
