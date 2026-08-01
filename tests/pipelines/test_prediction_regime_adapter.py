from __future__ import annotations

from dataclasses import dataclass

import pytest

from lrp.contracts import ContractError
from lrp.pipelines.prediction import (
    _regime_features,
)


@dataclass(frozen=True)
class NumberStatisticsStub:
    number: int
    total_frequency: int
    short_frequency: int
    mid_frequency: int
    long_frequency: int
    gap: int


@dataclass(frozen=True)
class SnapshotStub:
    numbers: dict[int, NumberStatisticsStub]


def make_snapshot() -> SnapshotStub:
    return SnapshotStub(
        numbers={
            number: NumberStatisticsStub(
                number=number,
                total_frequency=70 + number,
                short_frequency=number % 4,
                mid_frequency=number % 6,
                long_frequency=number % 10,
                gap=number % 8,
            )
            for number in range(1, 46)
        }
    )


def test_regime_features_adapt_project_c_fields() -> None:
    features = _regime_features(
        make_snapshot()
    )

    assert len(features) == 45

    first = features[0]

    assert first.number == 1
    assert first.freq_all == 71.0
    assert first.freq10 == 1.0
    assert first.freq20 == 1.0
    assert first.freq50 == 1.0
    assert first.gap == 1.0


def test_regime_features_are_sorted() -> None:
    original = make_snapshot()

    snapshot = SnapshotStub(
        numbers=dict(
            reversed(
                tuple(original.numbers.items())
            )
        )
    )

    features = _regime_features(snapshot)

    assert tuple(
        feature.number
        for feature in features
    ) == tuple(range(1, 46))


def test_regime_features_require_45_numbers() -> None:
    snapshot = make_snapshot()
    snapshot.numbers.pop(45)

    with pytest.raises(
        ContractError,
        match="exactly 45",
    ):
        _regime_features(snapshot)


def test_regime_features_require_source_fields() -> None:
    snapshot = make_snapshot()

    snapshot.numbers[1] = {
        "number": 1,
        "total_frequency": 79,
        "short_frequency": 1,
        "mid_frequency": 2,
        "gap": 4,
    }

    with pytest.raises(
        ContractError,
        match="long_frequency",
    ):
        _regime_features(snapshot)


@dataclass(frozen=True)
class NormalizedFeatureStub:
    number: int
    freq_all: int
    freq10: int
    freq20: int
    freq50: int
    gap: int


def test_regime_features_accept_normalized_fields() -> None:
    snapshot = SnapshotStub(
        numbers={
            number: NormalizedFeatureStub(
                number=number,
                freq_all=70 + number,
                freq10=number % 4,
                freq20=number % 6,
                freq50=number % 10,
                gap=number % 8,
            )
            for number in range(1, 46)
        }
    )

    features = _regime_features(snapshot)

    assert features[0].freq_all == 71.0
    assert features[0].freq10 == 1.0
    assert features[0].freq20 == 1.0
    assert features[0].freq50 == 1.0
