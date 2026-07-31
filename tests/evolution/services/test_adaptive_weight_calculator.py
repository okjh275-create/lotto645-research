from __future__ import annotations

from datetime import datetime, timezone
from math import isclose

import pytest

from lrp.evolution import (
    AdaptiveWeightCalculator,
    AdaptiveWeightProfile,
)


FIXED_TIME = datetime(
    2026,
    7,
    31,
    10,
    0,
    0,
    tzinfo=timezone.utc,
)


def test_zero_confidence_returns_baseline_weights() -> None:
    calculator = AdaptiveWeightCalculator()

    profile = calculator.calculate(
        {"hot": 1.0, "cold": -1.0},
        confidence=0.0,
        sample_size=10,
        revision=2,
        generated_at=FIXED_TIME,
    )

    assert profile.to_probability_weights() == pytest.approx(
        AdaptiveWeightProfile.default(
            generated_at=FIXED_TIME,
        ).to_probability_weights()
    )


def test_empty_signals_keep_baseline_weights() -> None:
    calculator = AdaptiveWeightCalculator()

    profile = calculator.calculate(
        {},
        confidence=1.0,
        sample_size=10,
        revision=2,
        generated_at=FIXED_TIME,
    )

    assert profile.to_probability_weights() == pytest.approx(
        AdaptiveWeightProfile.default(
            generated_at=FIXED_TIME,
        ).to_probability_weights()
    )


def test_positive_signal_increases_relative_weight() -> None:
    calculator = AdaptiveWeightCalculator()

    baseline = AdaptiveWeightProfile.default(
        generated_at=FIXED_TIME,
    )

    profile = calculator.calculate(
        {"hot": 1.0},
        confidence=1.0,
        sample_size=20,
        revision=2,
        generated_at=FIXED_TIME,
        baseline=baseline,
    )

    assert profile.hot_weight > baseline.hot_weight


def test_negative_signal_decreases_relative_weight() -> None:
    calculator = AdaptiveWeightCalculator()

    baseline = AdaptiveWeightProfile.default(
        generated_at=FIXED_TIME,
    )

    profile = calculator.calculate(
        {"hot": -1.0},
        confidence=1.0,
        sample_size=20,
        revision=2,
        generated_at=FIXED_TIME,
        baseline=baseline,
    )

    assert profile.hot_weight < baseline.hot_weight


def test_partial_confidence_blends_adjusted_and_baseline() -> None:
    calculator = AdaptiveWeightCalculator()

    baseline = AdaptiveWeightProfile.default(
        generated_at=FIXED_TIME,
    )

    full_profile = calculator.calculate(
        {"hot": 1.0},
        confidence=1.0,
        sample_size=20,
        revision=2,
        generated_at=FIXED_TIME,
        baseline=baseline,
    )

    partial_profile = calculator.calculate(
        {"hot": 1.0},
        confidence=0.5,
        sample_size=20,
        revision=2,
        generated_at=FIXED_TIME,
        baseline=baseline,
    )

    assert (
        baseline.hot_weight
        < partial_profile.hot_weight
        < full_profile.hot_weight
    )


def test_calculated_weights_sum_to_one() -> None:
    calculator = AdaptiveWeightCalculator()

    profile = calculator.calculate(
        {
            "hot": 1.0,
            "cold": -0.8,
            "gap": 0.5,
            "trend": -0.3,
            "transition": 0.7,
            "learning": 0.2,
            "adaptive": -0.4,
        },
        confidence=0.85,
        sample_size=50,
        revision=4,
        generated_at=FIXED_TIME,
    )

    assert isclose(
        profile.total_weight,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    )


def test_metadata_is_preserved() -> None:
    calculator = AdaptiveWeightCalculator()

    profile = calculator.calculate(
        {"trend": 0.5},
        confidence=0.75,
        sample_size=36,
        revision=5,
        generated_at=FIXED_TIME,
    )

    assert profile.confidence == 0.75
    assert profile.sample_size == 36
    assert profile.revision == 5
    assert profile.generated_at == FIXED_TIME


def test_custom_baseline_is_supported() -> None:
    baseline = AdaptiveWeightProfile(
        hot_weight=0.20,
        cold_weight=0.20,
        gap_weight=0.20,
        trend_weight=0.10,
        transition_weight=0.10,
        learning_weight=0.10,
        adaptive_weight=0.10,
        confidence=0.4,
        sample_size=12,
        revision=3,
        generated_at=FIXED_TIME,
    )

    calculator = AdaptiveWeightCalculator()

    profile = calculator.calculate(
        {},
        confidence=0.0,
        sample_size=13,
        revision=4,
        generated_at=FIXED_TIME,
        baseline=baseline,
    )

    assert profile.to_probability_weights() == pytest.approx(
        baseline.to_probability_weights()
    )


def test_unknown_signal_is_rejected() -> None:
    calculator = AdaptiveWeightCalculator()

    with pytest.raises(
        ValueError,
        match="unknown adaptive signal components",
    ):
        calculator.calculate(
            {"unknown": 0.5},
            confidence=0.5,
            sample_size=1,
            revision=1,
            generated_at=FIXED_TIME,
        )


@pytest.mark.parametrize(
    "signal",
    [-1.01, 1.01],
)
def test_out_of_range_signal_is_rejected(
    signal: float,
) -> None:
    calculator = AdaptiveWeightCalculator()

    with pytest.raises(
        ValueError,
        match="between -1.0 and 1.0",
    ):
        calculator.calculate(
            {"hot": signal},
            confidence=0.5,
            sample_size=1,
            revision=1,
            generated_at=FIXED_TIME,
        )


@pytest.mark.parametrize(
    "confidence",
    [-0.01, 1.01],
)
def test_invalid_confidence_is_rejected(
    confidence: float,
) -> None:
    calculator = AdaptiveWeightCalculator()

    with pytest.raises(
        ValueError,
        match="between 0.0 and 1.0",
    ):
        calculator.calculate(
            {},
            confidence=confidence,
            sample_size=1,
            revision=1,
            generated_at=FIXED_TIME,
        )


def test_non_finite_signal_is_rejected() -> None:
    calculator = AdaptiveWeightCalculator()

    with pytest.raises(
        ValueError,
        match="must be finite",
    ):
        calculator.calculate(
            {"hot": float("nan")},
            confidence=0.5,
            sample_size=1,
            revision=1,
            generated_at=FIXED_TIME,
        )


def test_negative_adjustment_scale_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="greater than or equal to 0",
    ):
        AdaptiveWeightCalculator(
            adjustment_scale=-0.1,
        )


def test_invalid_minimum_weight_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="too large",
    ):
        AdaptiveWeightCalculator(
            minimum_weight=0.15,
        )


def test_public_evolution_api_exports_calculator() -> None:
    calculator = AdaptiveWeightCalculator()

    assert calculator.adjustment_scale == 0.25
    assert calculator.minimum_weight == 0.01