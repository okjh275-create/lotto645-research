from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Mapping

from lrp.evolution.contracts.bayesian import (
    BayesianPosterior,
)


@dataclass(frozen=True, slots=True)
class RegimeBayesianState:
    """Bayesian posterior state for adaptive global regimes."""

    gap_recovery: BayesianPosterior
    cluster_rotation: BayesianPosterior
    high_band_expansion: BayesianPosterior
    low_band_expansion: BayesianPosterior

    REGIMES: ClassVar[tuple[str, ...]] = (
        "gap_recovery",
        "cluster_rotation",
        "high_band_expansion",
        "low_band_expansion",
    )

    def __post_init__(self) -> None:
        for regime in self.REGIMES:
            posterior = getattr(
                self,
                regime,
            )

            if not isinstance(
                posterior,
                BayesianPosterior,
            ):
                raise TypeError(
                    f"posterior for '{regime}' must be "
                    "BayesianPosterior"
                )

    @classmethod
    def default(
        cls,
        *,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> "RegimeBayesianState":
        posterior = BayesianPosterior(
            alpha=alpha,
            beta=beta,
        )

        return cls.from_posteriors(
            {
                regime: posterior
                for regime in cls.REGIMES
            }
        )

    @classmethod
    def from_posteriors(
        cls,
        posteriors: Mapping[
            str,
            BayesianPosterior,
        ],
    ) -> "RegimeBayesianState":
        if not isinstance(
            posteriors,
            Mapping,
        ):
            raise TypeError(
                "posteriors must be a mapping"
            )

        provided = set(posteriors)
        required = set(cls.REGIMES)

        missing = sorted(
            required - provided
        )
        unknown = sorted(
            provided - required
        )

        if missing:
            raise ValueError(
                "missing regime posteriors: "
                + ", ".join(missing)
            )

        if unknown:
            raise ValueError(
                "unknown regime posteriors: "
                + ", ".join(unknown)
            )

        for regime in cls.REGIMES:
            if not isinstance(
                posteriors[regime],
                BayesianPosterior,
            ):
                raise TypeError(
                    f"posterior for '{regime}' must be "
                    "BayesianPosterior"
                )

        return cls(
            gap_recovery=(
                posteriors["gap_recovery"]
            ),
            cluster_rotation=(
                posteriors["cluster_rotation"]
            ),
            high_band_expansion=(
                posteriors["high_band_expansion"]
            ),
            low_band_expansion=(
                posteriors["low_band_expansion"]
            ),
        )

    @property
    def posteriors(
        self,
    ) -> dict[str, BayesianPosterior]:
        return {
            regime: getattr(
                self,
                regime,
            )
            for regime in self.REGIMES
        }

    def to_signals(
        self,
    ) -> dict[str, float]:
        return {
            regime: posterior.adaptive_signal
            for regime, posterior
            in self.posteriors.items()
        }