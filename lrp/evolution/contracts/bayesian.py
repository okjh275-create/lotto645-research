from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import ClassVar, Mapping


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
        """Return the empirical success rate."""

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


@dataclass(frozen=True, slots=True)
class BayesianComponentState:
    """Posterior state assigned to one evolution component."""

    name: str
    posterior: BayesianPosterior

    COMPONENTS: ClassVar[tuple[str, ...]] = (
        "hot",
        "cold",
        "gap",
        "trend",
        "transition",
        "learning",
        "adaptive",
    )

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError(
                "name must be a string"
            )

        normalized_name = self.name.strip()

        if normalized_name not in self.COMPONENTS:
            raise ValueError(
                f"unsupported Bayesian component: "
                f"{normalized_name}"
            )

        if not isinstance(
            self.posterior,
            BayesianPosterior,
        ):
            raise TypeError(
                "posterior must be BayesianPosterior"
            )

        object.__setattr__(
            self,
            "name",
            normalized_name,
        )

    @property
    def signal(self) -> float:
        """Return the component adaptive signal."""

        return self.posterior.adaptive_signal


@dataclass(frozen=True, slots=True)
class BayesianState:
    """Complete Bayesian state for all evolution components."""

    hot: BayesianComponentState
    cold: BayesianComponentState
    gap: BayesianComponentState
    trend: BayesianComponentState
    transition: BayesianComponentState
    learning: BayesianComponentState
    adaptive: BayesianComponentState

    COMPONENTS: ClassVar[tuple[str, ...]] = (
        "hot",
        "cold",
        "gap",
        "trend",
        "transition",
        "learning",
        "adaptive",
    )

    def __post_init__(self) -> None:
        for name in self.COMPONENTS:
            component = getattr(self, name)

            if not isinstance(
                component,
                BayesianComponentState,
            ):
                raise TypeError(
                    f"{name} must be a "
                    "BayesianComponentState"
                )

            if component.name != name:
                raise ValueError(
                    f"{name} component must have "
                    f"name '{name}'"
                )

    @classmethod
    def default(
        cls,
        *,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> BayesianState:
        """Create a state using an identical prior for every component."""

        posterior = BayesianPosterior(
            alpha=alpha,
            beta=beta,
        )

        return cls.from_posteriors(
            {
                name: posterior
                for name in cls.COMPONENTS
            }
        )

    @classmethod
    def from_posteriors(
        cls,
        posteriors: Mapping[str, BayesianPosterior],
    ) -> BayesianState:
        """Create a state from an exact component-posterior mapping."""

        cls._validate_posteriors(posteriors)

        components = {
            name: BayesianComponentState(
                name=name,
                posterior=posteriors[name],
            )
            for name in cls.COMPONENTS
        }

        return cls(
            hot=components["hot"],
            cold=components["cold"],
            gap=components["gap"],
            trend=components["trend"],
            transition=components["transition"],
            learning=components["learning"],
            adaptive=components["adaptive"],
        )

    @property
    def components(
        self,
    ) -> dict[str, BayesianComponentState]:
        """Return all component states by canonical name."""

        return {
            name: getattr(self, name)
            for name in self.COMPONENTS
        }

    @property
    def posteriors(
        self,
    ) -> dict[str, BayesianPosterior]:
        """Return all posterior values by component name."""

        return {
            name: component.posterior
            for name, component in self.components.items()
        }

    def to_signals(self) -> dict[str, float]:
        """Return signals compatible with AdaptiveWeightCalculator."""

        return {
            name: component.signal
            for name, component in self.components.items()
        }

    @classmethod
    def _validate_posteriors(
        cls,
        posteriors: Mapping[str, BayesianPosterior],
    ) -> None:
        if not isinstance(posteriors, Mapping):
            raise TypeError(
                "posteriors must be a mapping"
            )

        provided = set(posteriors)
        required = set(cls.COMPONENTS)

        missing = sorted(required - provided)
        unknown = sorted(provided - required)

        if missing:
            raise ValueError(
                "missing Bayesian components: "
                + ", ".join(missing)
            )

        if unknown:
            raise ValueError(
                "unknown Bayesian components: "
                + ", ".join(unknown)
            )

        for name in cls.COMPONENTS:
            if not isinstance(
                posteriors[name],
                BayesianPosterior,
            ):
                raise TypeError(
                    f"posterior for '{name}' must be "
                    "BayesianPosterior"
                )
