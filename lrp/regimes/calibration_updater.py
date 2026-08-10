from __future__ import annotations

from dataclasses import replace

from lrp.evolution.contracts.regime_calibration import (
    RegimeCalibration,
)
from lrp.regimes.reward import RegimeReward


class RegimeCalibrationUpdater:
    """Update one regime calibration from an effective reward."""

    def __init__(
        self,
        *,
        learning_rate: float = 0.10,
    ) -> None:
        if isinstance(learning_rate, bool):
            raise TypeError(
                "learning_rate must be numeric"
            )

        if not isinstance(learning_rate, (int, float)):
            raise TypeError(
                "learning_rate must be numeric"
            )

        normalized = float(learning_rate)

        if not 0.0 < normalized <= 1.0:
            raise ValueError(
                "learning_rate must be greater than 0 and less than or equal to 1"
            )

        self._learning_rate = normalized

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    def update(
        self,
        calibration: RegimeCalibration,
        reward: RegimeReward,
    ) -> RegimeCalibration:
        if not isinstance(
            calibration,
            RegimeCalibration,
        ):
            raise TypeError(
                "calibration must be a RegimeCalibration"
            )

        if not isinstance(reward, RegimeReward):
            raise TypeError(
                "reward must be a RegimeReward"
            )

        regime = reward.regime

        if regime in {"neutral", "mixed"}:
            return calibration

        current = calibration.get(regime)

        updated = (
            current
            + self.learning_rate
            * reward.effective_reward
        )

        updated = max(
            0.50,
            min(1.50, updated),
        )

        if regime == "gap_recovery":
            return replace(
                calibration,
                gap_recovery=updated,
            )

        if regime == "cluster_rotation":
            return replace(
                calibration,
                cluster_rotation=updated,
            )

        if regime == "high_band_expansion":
            return replace(
                calibration,
                high_band_expansion=updated,
            )

        if regime == "low_band_expansion":
            return replace(
                calibration,
                low_band_expansion=updated,
            )

        return calibration
