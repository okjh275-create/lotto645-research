from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RegimeAdjustmentConfig:
    """Conservative probability adjustment limits by regime."""

    gap_recovery_max_boost: float = 0.02
    cluster_rotation_max_boost: float = 0.01
    high_band_max_boost: float = 0.02
    low_band_max_boost: float = 0.02

    def __post_init__(self) -> None:
        for field_name in (
            "gap_recovery_max_boost",
            "cluster_rotation_max_boost",
            "high_band_max_boost",
            "low_band_max_boost",
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

            if not 0.0 <= normalized <= 0.10:
                raise ValueError(
                    f"{field_name} must be between 0 and 0.10"
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

    @property
    def config(self) -> RegimeAdjustmentConfig:
        return self._config

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
        )

        return ProbabilityVectorAdjuster.adjust(
            probability_vector,
            multipliers=multipliers,
        )

    def _multipliers(
        self,
        probability_vector: object,
        global_regime: object,
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
                )

            elif (
                regime == "high_band_expansion"
                and 31 <= item.number <= 45
            ):
                boost = (
                    self._config
                    .high_band_max_boost
                    * confidence
                )

            elif (
                regime == "low_band_expansion"
                and 1 <= item.number <= 15
            ):
                boost = (
                    self._config
                    .low_band_max_boost
                    * confidence
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