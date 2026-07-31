from __future__ import annotations

from math import isfinite
from typing import Mapping

from lrp.evolution.contracts.bayesian import (
    BayesianEvidence,
    BayesianPosterior,
    BayesianState,
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
        """Return a fresh posterior representing the prior."""

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


class BayesianSignalCalculator:
    """Update component posteriors and produce adaptive signals."""

    COMPONENTS = BayesianState.COMPONENTS

    def __init__(
        self,
        updater: BayesianPosteriorUpdater | None = None,
    ) -> None:
        if (
            updater is not None
            and not isinstance(
                updater,
                BayesianPosteriorUpdater,
            )
        ):
            raise TypeError(
                "updater must be a "
                "BayesianPosteriorUpdater or None"
            )

        self._updater = (
            updater
            if updater is not None
            else BayesianPosteriorUpdater()
        )

    @property
    def updater(self) -> BayesianPosteriorUpdater:
        return self._updater

    def initial_state(self) -> BayesianState:
        """Return an initial state from the updater prior."""

        return BayesianState.default(
            alpha=self.updater.prior_alpha,
            beta=self.updater.prior_beta,
        )

    def update(
        self,
        evidence: Mapping[str, BayesianEvidence],
        *,
        previous: BayesianState | None = None,
    ) -> BayesianState:
        """Update all Bayesian components from exact evidence."""

        normalized_evidence = self._validate_evidence(
            evidence
        )

        if (
            previous is not None
            and not isinstance(
                previous,
                BayesianState,
            )
        ):
            raise TypeError(
                "previous must be BayesianState or None"
            )

        baseline = (
            previous
            if previous is not None
            else self.initial_state()
        )

        posteriors = {
            name: self.updater.update(
                normalized_evidence[name],
                previous=baseline.posteriors[name],
            )
            for name in self.COMPONENTS
        }

        return BayesianState.from_posteriors(
            posteriors
        )

    def calculate(
        self,
        state: BayesianState,
    ) -> dict[str, float]:
        """Return AdaptiveWeightCalculator-compatible signals."""

        if not isinstance(
            state,
            BayesianState,
        ):
            raise TypeError(
                "state must be BayesianState"
            )

        return state.to_signals()

    @classmethod
    def _validate_evidence(
        cls,
        evidence: Mapping[str, BayesianEvidence],
    ) -> dict[str, BayesianEvidence]:
        if not isinstance(evidence, Mapping):
            raise TypeError(
                "evidence must be a mapping"
            )

        provided = set(evidence)
        required = set(cls.COMPONENTS)

        missing = sorted(required - provided)
        unknown = sorted(provided - required)

        if missing:
            raise ValueError(
                "missing Bayesian evidence: "
                + ", ".join(missing)
            )

        if unknown:
            raise ValueError(
                "unknown Bayesian evidence: "
                + ", ".join(unknown)
            )

        normalized: dict[str, BayesianEvidence] = {}

        for name in cls.COMPONENTS:
            value = evidence[name]

            if not isinstance(
                value,
                BayesianEvidence,
            ):
                raise TypeError(
                    f"evidence for '{name}' must be "
                    "BayesianEvidence"
                )

            normalized[name] = value

        return normalized
