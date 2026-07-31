"""Tests for F-001/F-002 prediction-pipeline wiring."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from lrp.contracts import ContractError
from lrp.pipelines.prediction import _regime_features
from lrp.prediction import (
    ProbabilityFusionEngine,
    RegimeDetector,
)


@dataclass(frozen=True)
class Feature:
    number: int
    freq_all: int
    freq10: int
    freq20: int
    freq50: int
    gap: int


class Snapshot:
    def __init__(self, numbers: object) -> None:
        self.numbers = numbers


def _features() -> tuple[Feature, ...]:
    return tuple(
        Feature(
            number=number,
            freq_all=30 + number,
            freq10=number % 4,
            freq20=(number * 2) % 7,
            freq50=(number * 3) % 12,
            gap=number % 16,
        )
        for number in range(1, 46)
    )


def test_regime_features_accepts_sequence() -> None:
    features = _regime_features(
        Snapshot(_features())
    )

    assert len(features) == 45
    assert features[0].number == 1
    assert features[-1].number == 45


def test_regime_features_accepts_mapping() -> None:
    source = {
        item.number: item
        for item in _features()
    }

    features = _regime_features(
        Snapshot(source)
    )

    assert len(features) == 45


def test_regime_features_rejects_incomplete_data() -> None:
    with pytest.raises(ContractError):
        _regime_features(
            Snapshot(_features()[:-1])
        )


def test_f001_f002_build_probability_vector() -> None:
    profile = RegimeDetector().detect(
        _regime_features(
            Snapshot(_features())
        ),
        round_no=1220,
        generated_at_kst=(
            "2026-07-31T19:00:00+09:00"
        ),
    )

    vector = ProbabilityFusionEngine().build(
        profile
    )

    assert vector.round_no == 1220
    assert len(vector.probabilities) == 45

    total = sum(
        item.probability
        for item in vector.probabilities
    )

    assert abs(total - 1.0) < 1e-9
    assert vector.metadata["engine"] == "F-002"
    assert vector.metadata["source_engine"] == "F-001"
