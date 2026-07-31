from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class BayesianEvidence:
    """Binary outcome evidence for a Bayesian component."""

    successes: int
    failures: int

    def __post_init__(self) -> None:
        self._validate_count(
            "successes",
            self.successes,
        )
        self._validate_count(
            "failures",
            self.failures,
        )

    @property
    def observations(self) -> int:
        """Return the total number of observed outcomes."""

        return self.successes + self.failures

    @property
    def success_rate(self) -> float:
        """Return the empirical success rate.

        An empty evidence set has no empirical success rate and
        therefore returns 0.0.
        """

        if self.observations == 0:
            return 0.0

        return self.successes / self.observations

    @staticmethod
    def _validate_count(
        name: str,
        value: int,
    ) -> None:
        if isinstance(value, bool):
            raise TypeError(
                f"{name} must be an integer"
            )

        if not isinstance(value, int):
            raise TypeError(
                f"{name} must be an integer"
            )

        if value < 0:
            raise ValueError(
                f"{name} must be greater than "
                "or equal to 0"
            )


@dataclass(frozen=True, slots=True)
class BayesianPosterior:
    """Immutable Beta posterior for a binary outcome model."""

    alpha: float
    beta: float

    def __post_init__(self) -> None:
        self._validate_parameter(
            "alpha",
            self.alpha,
        )
        self._validate_parameter(
            "beta",
            self.beta,
        )

    @property
    def concentration(self) -> float:
        """Return the total Beta distribution concentration."""

        return self.alpha + self.beta

    @property
    def mean(self) -> float:
        """Return the posterior expected success probability."""

        return self.alpha / self.concentration

    @property
    def variance(self) -> float:
        """Return the posterior variance."""

        concentration = self.concentration

        return (
            self.alpha
            * self.beta
            / (
                concentration
                * concentration
                * (concentration + 1.0)
            )
        )

    @property
    def adaptive_signal(self) -> float:
        """Map posterior mean from [0, 1] to [-1, 1]."""

        return (2.0 * self.mean) - 1.0

    def updated(
        self,
        evidence: BayesianEvidence,
    ) -> BayesianPosterior:
        """Return a posterior updated with new evidence."""

        if not isinstance(
            evidence,
            BayesianEvidence,
        ):
            raise TypeError(
                "evidence must be BayesianEvidence"
            )

        return BayesianPosterior(
            alpha=self.alpha + evidence.successes,
            beta=self.beta + evidence.failures,
        )

    @staticmethod
    def _validate_parameter(
        name: str,
        value: float,
    ) -> None:
        if isinstance(value, bool):
            raise TypeError(
                f"{name} must be numeric"
            )

        if not isinstance(value, (int, float)):
            raise TypeError(
                f"{name} must be numeric"
            )

        numeric_value = float(value)

        if not isfinite(numeric_value):
            raise ValueError(
                f"{name} must be finite"
            )

        if numeric_value <= 0.0:
            raise ValueError(
                f"{name} must be greater than 0"
            )
