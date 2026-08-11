from __future__ import annotations

import pytest

from lrp.evolution.contracts.regime_calibration import (
    RegimeCalibration,
)
from lrp.regimes.calibration_updater import (
    RegimeCalibrationUpdater,
)
from lrp.regimes.learning_rate import (
    AdaptiveLearningRatePolicy,
)
from lrp.regimes.reward import RegimeReward


def make_reward(
    *,
    regime: str = "gap_recovery",
    reward: float = 1.0,
) -> RegimeReward:
    return RegimeReward(
        regime=regime,
        reward=reward,
        confidence=1.0,
        sample_weight=1.0,
    )


def test_default_updater_preserves_fixed_rate() -> None:
    updater = RegimeCalibrationUpdater(
        learning_rate=0.10
    )

    result = updater.update(
        RegimeCalibration.neutral(),
        make_reward(),
        revision=100,
        sample_size=1000,
    )

    assert result.gap_recovery == pytest.approx(
        1.10
    )


def test_adaptive_policy_uses_base_rate_when_fresh() -> None:
    updater = RegimeCalibrationUpdater(
        learning_rate_policy=(
            AdaptiveLearningRatePolicy()
        )
    )

    result = updater.update(
        RegimeCalibration.neutral(),
        make_reward(),
        revision=0,
        sample_size=0,
    )

    assert result.gap_recovery == pytest.approx(
        1.10
    )


def test_adaptive_policy_reduces_update_when_mature() -> None:
    updater = RegimeCalibrationUpdater(
        learning_rate_policy=(
            AdaptiveLearningRatePolicy()
        )
    )

    calibration = RegimeCalibration.neutral()
    reward = make_reward()

    fresh = updater.update(
        calibration,
        reward,
        revision=1,
        sample_size=0,
    )
    mature = updater.update(
        calibration,
        reward,
        revision=25,
        sample_size=100,
    )

    fresh_delta = (
        fresh.gap_recovery
        - calibration.gap_recovery
    )
    mature_delta = (
        mature.gap_recovery
        - calibration.gap_recovery
    )

    assert mature_delta < fresh_delta
    assert mature_delta > 0.0


def test_adaptive_policy_respects_minimum_rate() -> None:
    updater = RegimeCalibrationUpdater(
        learning_rate_policy=(
            AdaptiveLearningRatePolicy(
                base_rate=0.10,
                min_rate=0.02,
            )
        )
    )

    result = updater.update(
        RegimeCalibration.neutral(),
        make_reward(),
        revision=1_000_000,
        sample_size=1_000_000,
    )

    assert result.gap_recovery == pytest.approx(
        1.02
    )


def test_negative_reward_uses_adaptive_rate() -> None:
    updater = RegimeCalibrationUpdater(
        learning_rate_policy=(
            AdaptiveLearningRatePolicy()
        )
    )

    result = updater.update(
        RegimeCalibration.neutral(),
        make_reward(reward=-1.0),
        revision=25,
        sample_size=100,
    )

    assert result.gap_recovery == pytest.approx(
        0.98
    )


def test_unrelated_regime_remains_unchanged() -> None:
    updater = RegimeCalibrationUpdater(
        learning_rate_policy=(
            AdaptiveLearningRatePolicy()
        )
    )

    calibration = RegimeCalibration.neutral()

    result = updater.update(
        calibration,
        make_reward(
            regime="gap_recovery",
        ),
        revision=4,
        sample_size=60,
    )

    assert result.cluster_rotation == (
        calibration.cluster_rotation
    )
    assert result.high_band_expansion == (
        calibration.high_band_expansion
    )
    assert result.low_band_expansion == (
        calibration.low_band_expansion
    )


def test_invalid_learning_rate_policy_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="learning_rate_policy",
    ):
        RegimeCalibrationUpdater(
            learning_rate_policy=object()
        )