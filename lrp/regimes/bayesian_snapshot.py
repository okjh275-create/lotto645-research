from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, ClassVar, Mapping

from lrp.evolution.contracts.bayesian import (
    BayesianPosterior,
)
from lrp.regimes.bayesian_state import (
    RegimeBayesianState,
)


@dataclass(frozen=True, slots=True)
class RegimeBayesianSnapshot:
    """Persisted Bayesian posterior state for global regimes."""

    state: RegimeBayesianState
    saved_at: datetime
    revision: int
    sample_size: int
    schema_version: int = 1

    CURRENT_SCHEMA_VERSION: ClassVar[int] = 1

    def __post_init__(self) -> None:
        if not isinstance(
            self.state,
            RegimeBayesianState,
        ):
            raise TypeError(
                "state must be a RegimeBayesianState"
            )

        if not isinstance(self.saved_at, datetime):
            raise TypeError(
                "saved_at must be a datetime"
            )

        if (
            self.saved_at.tzinfo is None
            or self.saved_at.utcoffset() is None
        ):
            raise ValueError(
                "saved_at must be timezone-aware"
            )

        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
        ):
            raise TypeError(
                "revision must be an integer"
            )

        if self.revision < 1:
            raise ValueError(
                "revision must be greater than or equal to 1"
            )

        if (
            isinstance(self.sample_size, bool)
            or not isinstance(self.sample_size, int)
        ):
            raise TypeError(
                "sample_size must be an integer"
            )

        if self.sample_size < 0:
            raise ValueError(
                "sample_size must be greater than or equal to 0"
            )

        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
        ):
            raise TypeError(
                "schema_version must be an integer"
            )

        if self.schema_version != self.CURRENT_SCHEMA_VERSION:
            raise ValueError(
                "unsupported regime Bayesian schema version: "
                f"{self.schema_version}"
            )

    @classmethod
    def create(
        cls,
        state: RegimeBayesianState,
        *,
        revision: int = 1,
        sample_size: int = 0,
        saved_at: datetime | None = None,
    ) -> "RegimeBayesianSnapshot":
        return cls(
            state=state,
            saved_at=saved_at or datetime.now(timezone.utc),
            revision=revision,
            sample_size=sample_size,
            schema_version=cls.CURRENT_SCHEMA_VERSION,
        )

    def to_dict(self) -> dict[str, Any]:
        posteriors = {
            regime: {
                "alpha": posterior.alpha,
                "beta": posterior.beta,
            }
            for regime, posterior
            in self.state.posteriors.items()
        }

        return {
            "schema_version": self.schema_version,
            "saved_at": self.saved_at.isoformat(),
            "revision": self.revision,
            "sample_size": self.sample_size,
            "posteriors": posteriors,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "RegimeBayesianSnapshot":
        if not isinstance(payload, Mapping):
            raise TypeError(
                "payload must be a mapping"
            )

        required = (
            "schema_version",
            "saved_at",
            "revision",
            "sample_size",
            "posteriors",
        )

        missing = [
            name
            for name in required
            if name not in payload
        ]

        if missing:
            raise ValueError(
                "missing regime Bayesian snapshot fields: "
                + ", ".join(missing)
            )

        saved_at_raw = payload["saved_at"]

        if not isinstance(saved_at_raw, str):
            raise ValueError(
                "saved_at must be an ISO-8601 string"
            )

        try:
            saved_at = datetime.fromisoformat(
                saved_at_raw
            )
        except ValueError as exc:
            raise ValueError(
                "saved_at must be a valid ISO-8601 string"
            ) from exc

        posteriors_raw = payload["posteriors"]

        if not isinstance(posteriors_raw, Mapping):
            raise ValueError(
                "posteriors must be a mapping"
            )

        posteriors: dict[str, BayesianPosterior] = {}

        for regime, posterior_raw in posteriors_raw.items():
            if not isinstance(regime, str):
                raise ValueError(
                    "posterior regime names must be strings"
                )

            if not isinstance(posterior_raw, Mapping):
                raise ValueError(
                    f"posterior for '{regime}' must be a mapping"
                )

            if (
                "alpha" not in posterior_raw
                or "beta" not in posterior_raw
            ):
                raise ValueError(
                    f"posterior for '{regime}' must contain "
                    "alpha and beta"
                )

            posteriors[regime] = BayesianPosterior(
                alpha=float(posterior_raw["alpha"]),
                beta=float(posterior_raw["beta"]),
            )

        return cls(
            state=RegimeBayesianState.from_posteriors(
                posteriors
            ),
            saved_at=saved_at,
            revision=int(payload["revision"]),
            sample_size=int(payload["sample_size"]),
            schema_version=int(payload["schema_version"]),
        )