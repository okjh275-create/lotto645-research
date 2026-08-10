from __future__ import annotations

import math
import pytest

from lrp.evolution.contracts.review_reward_vector import (
    ReviewRewardVector,
)
from lrp.regimes.reward_calculator import (
    RegimeRewardCalculator,
)


def make_vector(
    *,
    portfolio: float = 0.5,
    practical: float = 0.2,
    rank_quality: float = 0.1,
    coverage: float = -0.2,
    sample_size: int = 10,
) -> ReviewRewardVector:
    return ReviewRewardVector(
        portfolio_hit=portfolio,
        practical_hit=practical,
        rank_quality=rank_quality,
        coverage=coverage,
        diversity=0.0,
        stability=0.0,
        sample_size=sample_size,
        metadata={},
    )


def test_calculate_builds_regime_reward() -> None:
    calculator = RegimeRewardCalculator()

    result = calculator.calculate(
        make_vector(),
        global_regime={
            "primary": "gap_recovery",
            "confidence": 0.8,
        },
    )

    expected = (
        0.45 * 0.5
        + 0.25 * 0.2
        + 0.20 * 0.1
        + 0.10 * (-0.2)
    )

    assert math.isclose(
        result.reward,
        expected,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )

    assert result.regime == "gap_recovery"
    assert result.confidence == 0.8
    assert result.sample_weight == 1.0


def test_sample_weight_is_normalized() -> None:
    calculator = RegimeRewardCalculator()

    reward = calculator.calculate(
        make_vector(sample_size=4),
        global_regime={
            "primary": "cluster_rotation",
            "confidence": 1.0,
        },
    )

    assert math.isclose(
        reward.sample_weight,
        0.4,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def test_sample_weight_is_capped_at_one() -> None:
    calculator = RegimeRewardCalculator()

    reward = calculator.calculate(
        make_vector(sample_size=50),
        global_regime={
            "primary": "high_band_expansion",
            "confidence": 0.5,
        },
    )

    assert reward.sample_weight == 1.0


def test_reward_is_clamped() -> None:
    calculator = RegimeRewardCalculator()

    reward = calculator.calculate(
        make_vector(
            portfolio=1.0,
            practical=1.0,
            rank_quality=1.0,
            coverage=1.0,
        ),
        global_regime={
            "primary": "gap_recovery",
            "confidence": 1.0,
        },
    )

    assert math.isclose(
        reward.reward,
        1.0,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"primary": 1, "confidence": 0.5},
        {"primary": "gap_recovery"},
        {"primary": "gap_recovery", "confidence": "0.5"},
    ],
)
def test_invalid_global_regime_is_rejected(
    payload: dict[str, object],
) -> None:
    calculator = RegimeRewardCalculator()

    with pytest.raises(TypeError):
        calculator.calculate(
            make_vector(),
            global_regime=payload,
        )
