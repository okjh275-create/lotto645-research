from __future__ import annotations

import pytest

from lrp.contracts import ContractError
from lrp.regimes import (
    SUPPORTED_REGIMES,
    RegimeDecision,
    RegimeFeatureSnapshot,
)


def features() -> RegimeFeatureSnapshot:
    return RegimeFeatureSnapshot(
        average_recency=0.55,
        average_frequency=0.48,
        average_gap_reversion=0.45,
        pair_density=0.52,
        frequency_dispersion=0.31,
        recency_variance=0.18,
        pair_variance=0.22,
        low_band_ratio=0.49,
        high_band_ratio=0.51,
    )


def test_feature_snapshot_serializes() -> None:
    snapshot = features()

    payload = snapshot.as_dict()

    assert payload["average_recency"] == 0.55
    assert payload["pair_density"] == 0.52
    assert payload["high_band_ratio"] == 0.51


def test_feature_snapshot_rejects_out_of_range() -> None:
    with pytest.raises(ContractError):
        RegimeFeatureSnapshot(
            average_recency=1.1,
            average_frequency=0.5,
            average_gap_reversion=0.5,
            pair_density=0.5,
            frequency_dispersion=0.5,
            recency_variance=0.5,
            pair_variance=0.5,
            low_band_ratio=0.5,
            high_band_ratio=0.5,
        )


def test_regime_decision_serializes() -> None:
    decision = RegimeDecision(
        primary="gap_recovery",
        confidence=0.78,
        secondary="cluster_rotation",
        secondary_confidence=0.54,
        scores={
            "gap_recovery": 0.78,
            "cluster_rotation": 0.54,
            "neutral": 0.20,
        },
        features=features(),
    )

    payload = decision.as_dict()

    assert payload["primary"] == "gap_recovery"
    assert payload["confidence"] == 0.78
    assert payload["secondary"] == "cluster_rotation"
    assert payload["features"]["pair_density"] == 0.52


def test_decision_requires_primary_score() -> None:
    with pytest.raises(ContractError):
        RegimeDecision(
            primary="gap_recovery",
            confidence=0.7,
            scores={"neutral": 0.3},
            features=features(),
        )


def test_decision_rejects_unknown_regime() -> None:
    with pytest.raises(ContractError):
        RegimeDecision(
            primary="unknown",
            confidence=0.7,
            scores={"neutral": 0.3},
            features=features(),
        )


def test_secondary_requires_confidence() -> None:
    with pytest.raises(ContractError):
        RegimeDecision(
            primary="neutral",
            confidence=0.6,
            secondary="mixed",
            scores={
                "neutral": 0.6,
                "mixed": 0.5,
            },
            features=features(),
        )


def test_supported_regimes_include_neutral_and_mixed() -> None:
    assert "neutral" in SUPPORTED_REGIMES
    assert "mixed" in SUPPORTED_REGIMES
