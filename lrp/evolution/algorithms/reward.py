from __future__ import annotations

from math import isfinite

from lrp.evolution.contracts.reward import (
    RewardObservation,
)
from lrp.evolution.contracts.signals import (
    SignalFrame,
)


class RewardCalculator:
    """Convert observed component performance into signals."""

    def __init__(
        self,
        *,
        prior_strength: float = 2.0,
        target_sample_size: int = 20,
        source_prefix: str = "reward",
    ) -> None:
        self._prior_strength = (
            self._validate_prior_strength(
                prior_strength
            )
        )
        self._target_sample_size = (
            self._validate_target_sample_size(
                target_sample_size
            )
        )
        self._source_prefix = (
            self._validate_source_prefix(
                source_prefix
            )
        )

    @property
    def prior_strength(self) -> float:
        return self._prior_strength

    @property
    def target_sample_size(self) -> int:
        return self._target_sample_size

    @property
    def source_prefix(self) -> str:
        return self._source_prefix

    def calculate(
        self,
        observation: RewardObservation,
    ) -> SignalFrame:
        """Create a baseline-centred reward signal frame."""

        if not isinstance(
            observation,
            RewardObservation,
        ):
            raise TypeError(
                "observation must be a "
                "RewardObservation"
            )

        signals = {
            component: self._calculate_component_signal(
                successes=observation.successes[
                    component
                ],
                trials=observation.trials[
                    component
                ],
                baseline=observation.baseline_rate,
            )
            for component in (
                observation.observed_components
            )
        }

        reliability = min(
            1.0,
            observation.total_trials
            / self.target_sample_size,
        )

        return SignalFrame(
            source=(
                f"{self.source_prefix}:"
                f"{observation.source}"
            ),
            signals=signals,
            reliability=reliability,
        )

    def _calculate_component_signal(
        self,
        *,
        successes: int,
        trials: int,
        baseline: float,
    ) -> float:
        posterior_rate = (
            successes
            + baseline * self.prior_strength
        ) / (
            trials
            + self.prior_strength
        )

        if posterior_rate >= baseline:
            signal = (
                posterior_rate - baseline
            ) / (
                1.0 - baseline
            )
        else:
            signal = (
                posterior_rate - baseline
            ) / baseline

        return min(
            1.0,
            max(-1.0, signal),
        )

    @staticmethod
    def _validate_prior_strength(
        prior_strength: float,
    ) -> float:
        if isinstance(prior_strength, bool):
            raise TypeError(
                "prior_strength must be numeric"
            )

        if not isinstance(
            prior_strength,
            (int, float),
        ):
            raise TypeError(
                "prior_strength must be numeric"
            )

        value = float(prior_strength)

        if not isfinite(value):
            raise ValueError(
                "prior_strength must be finite"
            )

        if value < 0.0:
            raise ValueError(
                "prior_strength must be greater "
                "than or equal to 0"
            )

        return value

    @staticmethod
    def _validate_target_sample_size(
        target_sample_size: int,
    ) -> int:
        if isinstance(target_sample_size, bool):
            raise TypeError(
                "target_sample_size must be an integer"
            )

        if not isinstance(
            target_sample_size,
            int,
        ):
            raise TypeError(
                "target_sample_size must be an integer"
            )

        if target_sample_size < 1:
            raise ValueError(
                "target_sample_size must be greater "
                "than or equal to 1"
            )

        return target_sample_size

    @staticmethod
    def _validate_source_prefix(
        source_prefix: str,
    ) -> str:
        if not isinstance(source_prefix, str):
            raise TypeError(
                "source_prefix must be a string"
            )

        normalized = source_prefix.strip()

        if not normalized:
            raise ValueError(
                "source_prefix must not be empty"
            )

        return normalized
