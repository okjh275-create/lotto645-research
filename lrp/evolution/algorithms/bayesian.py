from __future__ import annotations

from math import isfinite

from lrp.evolution.contracts.bayesian import (
    BayesianEvidence,
    BayesianPosterior,
)


class BayesianPosteriorUpdater:
    """Update Beta posteriors from binary outcome evidence."""

    def __init__(
        self,
        *,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ) -> None:
        self._prior_alpha = self._validate_prior(
            "prior_alpha",
            prior_alpha,
        )
        self._prior_beta = self._validate_prior(
            "prior_beta",
            prior_beta,
        )

    @property
    def prior_alpha(self) -> float:
        return self._prior_alpha

    @property
    def prior_beta(self) -> float:
        return self._prior_beta

    def prior(self) -> BayesianPosterior:
        """Return a fresh posterior representing the configured prior."""

        return BayesianPosterior(
            alpha=self.prior_alpha,
            beta=self.prior_beta,
        )

    def update(
        self,
        evidence: BayesianEvidence,
        *,
        previous: BayesianPosterior | None = None,
    ) -> BayesianPosterior:
        """Update the configured prior or a previous posterior."""

        if not isinstance(
            evidence,
            BayesianEvidence,
        ):
            raise TypeError(
                "evidence must be BayesianEvidence"
            )

        if (
            previous is not None
            and not isinstance(
                previous,
                BayesianPosterior,
            )
        ):
            raise TypeError(
                "previous must be BayesianPosterior "
                "or None"
            )

        baseline = (
            previous
            if previous is not None
            else self.prior()
        )

        return baseline.updated(evidence)

    @staticmethod
    def _validate_prior(
        name: str,
        value: float,
    ) -> float:
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

        return numeric_value
