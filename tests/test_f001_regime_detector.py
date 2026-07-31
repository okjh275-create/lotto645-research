"""Tests for Project F-001 number regime detection."""

from __future__ import annotations

from dataclasses import dataclass

from lrp.prediction import (
    RegimeDetector,
    RegimeDetectorConfig,
)


@dataclass(frozen=True)
class SampleFeature:
    number: int
    freq_all: int
    freq10: int
    freq20: int
    freq50: int
    gap: int


def _build_features() -> list[SampleFeature]:
    features: list[SampleFeature] = []

    for number in range(1, 46):
        features.append(
            SampleFeature(
                number=number,
                freq_all=20 + number,
                freq10=number % 4,
                freq20=(number * 2) % 6,
                freq50=(number * 3) % 10,
                gap=number % 16,
            )
        )

    return features


def test_regime_detector_builds_complete_profile() -> None:
    detector = RegimeDetector()

    profile = detector.detect(
        _build_features(),
        round_no=1220,
        generated_at_kst="2026-07-31T18:30:00+09:00",
        metadata={"source": "unit-test"},
    )

    assert profile.round_no == 1220
    assert profile.generated_at_kst == (
        "2026-07-31T18:30:00+09:00"
    )
    assert len(profile.regimes) == 45
    assert profile.regimes[0].number == 1
    assert profile.regimes[-1].number == 45
    assert profile.metadata["engine"] == "F-001"
    assert profile.metadata["source"] == "unit-test"


def test_regime_scores_are_normalized() -> None:
    detector = RegimeDetector()
    profile = detector.detect(
        _build_features(),
        generated_at_kst="2026-07-31T18:30:00+09:00",
    )

    for regime in profile.regimes:
        assert 0.0 <= regime.hot_score <= 1.0
        assert 0.0 <= regime.cold_score <= 1.0
        assert 0.0 <= regime.gap_score <= 1.0
        assert 0.0 <= regime.trend_score <= 1.0
        assert 0.0 <= regime.transition_score <= 1.0
        assert 0.0 <= regime.confidence <= 1.0


def test_profile_lookup_and_top_ranking() -> None:
    detector = RegimeDetector()
    profile = detector.detect(
        _build_features(),
        generated_at_kst="2026-07-31T18:30:00+09:00",
    )

    assert profile.get(1).number == 1
    assert profile.get(45).number == 45

    top_gap = profile.top(
        metric="gap_score",
        limit=5,
    )

    assert len(top_gap) == 5
    assert all(
        top_gap[index].gap_score
        >= top_gap[index + 1].gap_score
        for index in range(len(top_gap) - 1)
    )


def test_detector_accepts_custom_configuration() -> None:
    config = RegimeDetectorConfig(
        short_window=10,
        mid_window=20,
        long_window=50,
        gap_saturation=20,
        confidence_saturation=60,
    )
    detector = RegimeDetector(config)

    profile = detector.detect(
        _build_features(),
        generated_at_kst="2026-07-31T18:30:00+09:00",
    )

    assert profile.metadata["config"]["gap_saturation"] == 20
    assert (
        profile.metadata["config"]["confidence_saturation"]
        == 60
    )


def main() -> None:
    test_regime_detector_builds_complete_profile()
    test_regime_scores_are_normalized()
    test_profile_lookup_and_top_ranking()
    test_detector_accepts_custom_configuration()
    print("OK: F-001 regime detector tests")


if __name__ == "__main__":
    main()
