from __future__ import annotations

import math

import pytest

from lrp.evolution.contracts.regime_calibration import (
    RegimeCalibration,
)
from lrp.regimes.calibration_updater import (
    RegimeCalibrationUpdater,
)
from lrp.regimes.reward import RegimeReward


def make_reward(
    regime: str,
    *,
    reward: float,
    confidence: float = 1.0,
    sample_weight: float = 1.0,
) -> RegimeReward:
    return RegimeReward(
        regime=regime,
        reward=reward,
        confidence=confidence,
        sample_weight=sample_weight,
    )


def test_positive_reward_increases_target_regime() -> None:
    updater = RegimeCalibrationUpdater(
        learning_rate=0.10
    )
    calibration = RegimeCalibration()

    updated = updater.update(
        calibration,
        make_reward(
            "gap_recovery",
            reward=0.24,
        ),
    )

    assert math.isclose(
        updated.gap_recovery,
        1.024,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def test_negative_reward_decreases_target_regime() -> None:
    updater = RegimeCalibrationUpdater(
        learning_rate=0.10
    )
    calibration = RegimeCalibration()

    updated = updater.update(
        calibration,
        make_reward(
            "gap_recovery",
            reward=-0.50,
        ),
    )

    assert math.isclose(
        updated.gap_recovery,
        0.95,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def test_update_uses_effective_reward() -> None:
    updater = RegimeCalibrationUpdater(
        learning_rate=0.10
    )

    updated = updater.update(
        RegimeCalibration(),
        make_reward(
            "gap_recovery",
            reward=1.0,
            confidence=0.5,
            sample_weight=0.4,
        ),
    )

    assert math.isclose(
        updated.gap_recovery,
        1.02,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def test_upper_bound_is_clamped() -> None:
    updater = RegimeCalibrationUpdater(
        learning_rate=1.0
    )

    updated = updater.update(
        RegimeCalibration(
            high_band_expansion=1.49
        ),
        make_reward(
            "high_band_expansion",
            reward=1.0,
        ),
    )

    assert updated.high_band_expansion == 1.50


def test_lower_bound_is_clamped() -> None:
    updater = RegimeCalibrationUpdater(
        learning_rate=1.0
    )

    updated = updater.update(
        RegimeCalibration(
            low_band_expansion=0.51
        ),
        make_reward(
            "low_band_expansion",
            reward=-1.0,
        ),
    )

    assert updated.low_band_expansion == 0.50


def test_only_target_regime_changes() -> None:
    calibration = RegimeCalibration(
        gap_recovery=1.00,
        cluster_rotation=0.90,
        high_band_expansion=1.20,
        low_band_expansion=0.80,
    )

    updated = RegimeCalibrationUpdater().update(
        calibration,
        make_reward(
            "gap_recovery",
            reward=0.5,
        ),
    )

    assert updated.gap_recovery != calibration.gap_recovery
    assert updated.cluster_rotation == calibration.cluster_rotation
    assert updated.high_band_expansion == calibration.high_band_expansion
    assert updated.low_band_expansion == calibration.low_band_expansion


@pytest.mark.parametrize(
    "regime",
    ["neutral", "mixed"],
)
def test_neutral_and_mixed_are_noop(
    regime: str,
) -> None:
    calibration = RegimeCalibration()

    updated = RegimeCalibrationUpdater().update(
        calibration,
        make_reward(
            regime,
            reward=1.0,
        ),
    )

    assert updated is calibration


@pytest.mark.parametrize(
    "value",
    [0.0, -0.1, 1.1],
)
def test_invalid_learning_rate_range_is_rejected(
    value: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="learning_rate",
    ):
        RegimeCalibrationUpdater(
            learning_rate=value
        )


@pytest.mark.parametrize(
    "value",
    [True, "0.1", None],
)
def test_invalid_learning_rate_type_is_rejected(
    value: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="learning_rate must be numeric",
    ):
        RegimeCalibrationUpdater(
            learning_rate=value
        )


def test_invalid_calibration_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="calibration must be a RegimeCalibration",
    ):
        RegimeCalibrationUpdater().update(
            object(),
            make_reward(
                "gap_recovery",
                reward=0.0,
            ),
        )


def test_invalid_reward_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="reward must be a RegimeReward",
    ):
        RegimeCalibrationUpdater().update(
            RegimeCalibration(),
            object(),
        )
