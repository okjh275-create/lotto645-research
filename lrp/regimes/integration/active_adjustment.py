from __future__ import annotations

from dataclasses import dataclass
from lrp.regimes.integration.bayesian_provider import (
    RegimeBayesianProvider,
)
from lrp.regimes.integration.calibration_provider import (
    RegimeCalibrationProvider,
)



@dataclass(frozen=True, slots=True)
class RegimeAdjustmentConfig:
    """Conservative probability adjustment limits by regime."""

    gap_recovery_max_boost: float = 0.02
    cluster_rotation_max_boost: float = 0.01
    high_band_max_boost: float = 0.02
    low_band_max_boost: float = 0.02
    bayesian_signal_limit: float = 0.25

    def __post_init__(self) -> None:
        for field_name in (
            "gap_recovery_max_boost",
            "cluster_rotation_max_boost",
            "high_band_max_boost",
            "low_band_max_boost",
            "bayesian_signal_limit",
        ):
            value = getattr(self, field_name)

            if isinstance(value, bool):
                raise TypeError(
                    f"{field_name} must be numeric"
                )

            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"{field_name} must be numeric"
                )

            normalized = float(value)

            upper_bound = (
                1.0
                if field_name == "bayesian_signal_limit"
                else 0.10
            )

            if not 0.0 <= normalized <= upper_bound:
                raise ValueError(
                    f"{field_name} must be between "
                    f"0 and {upper_bound:.2f}"
                )

            object.__setattr__(
                self,
                field_name,
                normalized,
            )


class ProbabilityVectorAdjuster:
    """Rebuild a normalized probability vector from score multipliers."""

    @staticmethod
    def adjust(
        probability_vector: object,
        *,
        multipliers: dict[int, float],
    ) -> object:
        from lrp.prediction.probability import (
            NumberProbability,
            ProbabilityVector,
        )

        if not isinstance(
            probability_vector,
            ProbabilityVector,
        ):
            raise TypeError(
                "probability_vector must be a ProbabilityVector"
            )

        adjusted_scores: dict[int, float] = {}

        for item in probability_vector.probabilities:
            multiplier = float(
                multipliers.get(
                    item.number,
                    1.0,
                )
            )

            if multiplier < 0.0:
                raise ValueError(
                    "probability multiplier must be non-negative"
                )

            adjusted_scores[item.number] = (
                item.raw_score * multiplier
            )

        total_score = sum(
            adjusted_scores.values()
        )

        if total_score <= 0.0:
            raise ValueError(
                "adjusted raw scores must have positive total"
            )

        ranked_numbers = sorted(
            adjusted_scores,
            key=lambda number: (
                -adjusted_scores[number],
                number,
            ),
        )

        rank_by_number = {
            number: index
            for index, number
            in enumerate(
                ranked_numbers,
                start=1,
            )
        }

        probabilities = tuple(
            NumberProbability(
                number=item.number,
                probability=(
                    adjusted_scores[item.number]
                    / total_score
                ),
                raw_score=adjusted_scores[item.number],
                rank=rank_by_number[item.number],
                components=item.components,
                metadata=item.metadata,
            )
            for item
            in probability_vector.probabilities
        )

        metadata = dict(
            probability_vector.metadata
        )
        metadata["global_regime_adjusted"] = True

        return ProbabilityVector(
            round_no=probability_vector.round_no,
            generated_at_kst=(
                probability_vector.generated_at_kst
            ),
            probabilities=probabilities,
            metadata=metadata,
        )

class ActiveGlobalRegimeAdjustmentAdapter:
    """Apply conservative global-regime probability adjustments."""

    def __init__(
        self,
        config: RegimeAdjustmentConfig | None = None,
        calibration_provider: (
            RegimeCalibrationProvider | None
        ) = None,
        bayesian_provider: (
            RegimeBayesianProvider | None
        ) = None,
    ) -> None:
        self._config = (
            config
            if config is not None
            else RegimeAdjustmentConfig()
        )

        if not isinstance(
            self._config,
            RegimeAdjustmentConfig,
        ):
            raise TypeError(
                "config must be a RegimeAdjustmentConfig"
            )

        if (
            calibration_provider is not None
            and not isinstance(
                calibration_provider,
                RegimeCalibrationProvider,
            )
        ):
            raise TypeError(
                "calibration_provider must implement "
                "RegimeCalibrationProvider"
            )

        if (
            bayesian_provider is not None
            and not isinstance(
                bayesian_provider,
                RegimeBayesianProvider,
            )
        ):
            raise TypeError(
                "bayesian_provider must implement "
                "RegimeBayesianProvider"
            )

        self._calibration_provider = (
            calibration_provider
        )
        self._bayesian_provider = (
            bayesian_provider
        )

    @property
    def config(self) -> RegimeAdjustmentConfig:
        return self._config

    @property
    def calibration_provider(
        self,
    ) -> RegimeCalibrationProvider | None:
        return self._calibration_provider

    @property
    def bayesian_provider(
        self,
    ) -> RegimeBayesianProvider | None:
        return self._bayesian_provider

    def adjust(
        self,
        probability_vector: object,
        *,
        global_regime: object | None,
        round_no: int,
        seed: int,
    ) -> object:
        from lrp.prediction.probability import (
            ProbabilityVector,
        )
        from lrp.regimes.contracts import (
            RegimeDecision,
        )

        self._validate_round_no(round_no)
        self._validate_seed(seed)

        if not isinstance(
            probability_vector,
            ProbabilityVector,
        ):
            raise TypeError(
                "probability_vector must be a ProbabilityVector"
            )

        if global_regime is None:
            return probability_vector

        if not isinstance(
            global_regime,
            RegimeDecision,
        ):
            raise TypeError(
                "global_regime must be a RegimeDecision or None"
            )

        regime = global_regime.primary

        if regime in {
            "neutral",
            "mixed",
        }:
            return probability_vector

        multipliers = self._multipliers(
            probability_vector,
            global_regime,
            round_no=round_no,
        )

        return ProbabilityVectorAdjuster.adjust(
            probability_vector,
            multipliers=multipliers,
        )

    def _multipliers(
        self,
        probability_vector: object,
        global_regime: object,
        *,
        round_no: int,
    ) -> dict[int, float]:
        from lrp.prediction.probability import (
            ProbabilityVector,
        )
        from lrp.regimes.contracts import (
            RegimeDecision,
        )

        if not isinstance(
            probability_vector,
            ProbabilityVector,
        ):
            raise TypeError(
                "probability_vector must be a ProbabilityVector"
            )

        if not isinstance(
            global_regime,
            RegimeDecision,
        ):
            raise TypeError(
                "global_regime must be a RegimeDecision"
            )

        regime = global_regime.primary
        confidence = global_regime.confidence

        calibration_factor = 1.0

        if self.calibration_provider is not None:
            calibration = (
                self.calibration_provider
                .get_calibration(
                    round_no=round_no
                )
            )

            if calibration is not None:
                calibration_factor = (
                    calibration.get(regime)
                )

        bayesian_factor = 1.0

        if self.bayesian_provider is not None:
            bayesian_state = (
                self.bayesian_provider
                .get_bayesian_state(
                    round_no=round_no
                )
            )

            if bayesian_state is not None:
                signal = bayesian_state.to_signals().get(
                    regime,
                    0.0,
                )

                limit = (
                    self._config
                    .bayesian_signal_limit
                )

                bounded_signal = max(
                    -limit,
                    min(limit, float(signal)),
                )

                bayesian_factor = (
                    1.0 + bounded_signal
                )

        multipliers: dict[int, float] = {}

        for item in probability_vector.probabilities:
            boost = 0.0

            if regime == "gap_recovery":
                strength = float(
                    item.components.get(
                        "gap",
                        0.0,
                    )
                )
                boost = (
                    self._config
                    .gap_recovery_max_boost
                    * confidence
                    * strength
                    * calibration_factor
                    * bayesian_factor
                )

            elif regime == "cluster_rotation":
                strength = float(
                    item.components.get(
                        "transition",
                        0.0,
                    )
                )
                boost = (
                    self._config
                    .cluster_rotation_max_boost
                    * confidence
                    * strength
                    * calibration_factor
                    * bayesian_factor
                )

            elif (
                regime == "high_band_expansion"
                and 31 <= item.number <= 45
            ):
                boost = (
                    self._config
                    .high_band_max_boost
                    * confidence
                    * calibration_factor
                    * bayesian_factor
                )

            elif (
                regime == "low_band_expansion"
                and 1 <= item.number <= 15
            ):
                boost = (
                    self._config
                    .low_band_max_boost
                    * confidence
                    * calibration_factor
                    * bayesian_factor
                )

            multipliers[item.number] = 1.0 + boost

        return multipliers

    @staticmethod
    def _validate_round_no(
        round_no: int,
    ) -> None:
        if (
            isinstance(round_no, bool)
            or not isinstance(round_no, int)
        ):
            raise TypeError(
                "round_no must be an integer"
            )

        if round_no < 1:
            raise ValueError(
                "round_no must be greater than or equal to 1"
            )

    @staticmethod
    def _validate_seed(
        seed: int,
    ) -> None:
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
        ):
            raise TypeError(
                "seed must be an integer"
            )