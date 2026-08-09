from __future__ import annotations

import pytest

from lrp.contracts import ContractError
from lrp.regimes import (
    RegimeDecision,
    RegimeFeatureSnapshot,
    RegimeStabilityConfig,
    RegimeStabilityPolicy,
)


def features(**overrides: float) -> RegimeFeatureSnapshot:
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


def previous_decision(
    primary: str,
) -> RegimeDecision:
    snapshot = features()

    return RegimeDecision(
        primary=primary,
        confidence=0.60,
        scores={
            primary: 0.60,
            "neutral": (
                0.60
                if primary == "neutral"
                else 0.40
            ),
        },
        features=snapshot,
    )


def test_without_previous_matches_detector() -> None:
    policy = RegimeStabilityPolicy()

    result = policy.decide(
        features(
            low_band_ratio=0.20,
            high_band_ratio=0.80,
            average_frequency=0.70,
            pair_density=0.60,
        )
    )

    assert result.primary in result.scores
    assert result.confidence == pytest.approx(
        result.scores[result.primary]
    )


def test_small_transition_can_retain_previous_regime() -> None:
    policy = RegimeStabilityPolicy(
        config=RegimeStabilityConfig(
            hysteresis_margin=1.0,
            minimum_retained_score=0.0,
        )
    )

    previous = previous_decision(
        "high_band_expansion"
    )

    result = policy.decide(
        features(
            low_band_ratio=0.55,
            high_band_ratio=0.45,
        ),
        previous=previous,
    )

    assert result.primary == "high_band_expansion"
    assert result.confidence == pytest.approx(
        result.scores["high_band_expansion"]
    )


def test_large_transition_is_not_suppressed() -> None:
    policy = RegimeStabilityPolicy(
        config=RegimeStabilityConfig(
            hysteresis_margin=0.01,
            minimum_retained_score=0.0,
        )
    )

    previous = previous_decision(
        "low_band_expansion"
    )

    result = policy.decide(
        features(
            low_band_ratio=0.10,
            high_band_ratio=0.90,
            average_frequency=0.80,
            pair_density=0.70,
        ),
        previous=previous,
    )

    assert result.primary != "low_band_expansion"


def test_secondary_confidence_matches_secondary_score() -> None:
    result = RegimeStabilityPolicy().decide(
        features(
            low_band_ratio=0.35,
            high_band_ratio=0.65,
            pair_density=0.55,
        )
    )

    if result.secondary is not None:
        assert result.secondary_confidence == pytest.approx(
            result.scores[result.secondary]
        )
        assert result.secondary_confidence <= result.confidence


def test_rejects_invalid_previous() -> None:
    with pytest.raises(ContractError):
        RegimeStabilityPolicy().decide(
            features(),
            previous=object(),
        )


def test_config_rejects_invalid_margin() -> None:
    with pytest.raises(ContractError):
        RegimeStabilityConfig(
            hysteresis_margin=1.1
        )
