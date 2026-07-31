from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import ClassVar, Mapping

from lrp.evolution.contracts.signals import (
    SIGNAL_COMPONENTS,
)


@dataclass(frozen=True, slots=True)
class RewardObservation:
    """Observed performance for evolution components.

    Each component records successful outcomes and the number
    of evaluated opportunities. The calculator converts these
    observations into normalized signals.
    """

    source: str
    successes: Mapping[str, int]
    trials: Mapping[str, int]
    baseline_rate: float = 0.5

    COMPONENTS: ClassVar[tuple[str, ...]] = (
        SIGNAL_COMPONENTS
    )

    def __post_init__(self) -> None:
        normalized_source = self._normalize_source(
            self.source
        )
        normalized_successes = self._normalize_counts(
            self.successes,
            field_name="successes",
            allow_zero=True,
        )
        normalized_trials = self._normalize_counts(
            self.trials,
            field_name="trials",
            allow_zero=False,
        )
        normalized_baseline = self._normalize_baseline(
            self.baseline_rate
        )

        self._validate_component_sets(
            successes=normalized_successes,
            trials=normalized_trials,
        )
        self._validate_success_bounds(
            successes=normalized_successes,
            trials=normalized_trials,
        )

        object.__setattr__(
            self,
            "source",
            normalized_source,
        )
        object.__setattr__(
            self,
            "successes",
            MappingProxyType(normalized_successes),
        )
        object.__setattr__(
            self,
            "trials",
            MappingProxyType(normalized_trials),
        )
        object.__setattr__(
            self,
            "baseline_rate",
            normalized_baseline,
        )

    @property
    def total_trials(self) -> int:
        return sum(self.trials.values())

    @property
    def observed_components(self) -> tuple[str, ...]:
        return tuple(
            component
            for component in self.COMPONENTS
            if component in self.trials
        )

    def success_rate(
        self,
        component: str,
    ) -> float:
        if component not in self.trials:
            raise ValueError(
                f"component was not observed: {component}"
            )

        return (
            self.successes[component]
            / self.trials[component]
        )

    @staticmethod
    def _normalize_source(
        source: str,
    ) -> str:
        if not isinstance(source, str):
            raise TypeError(
                "source must be a string"
            )

        normalized = source.strip()

        if not normalized:
            raise ValueError(
                "source must not be empty"
            )

        return normalized

    @classmethod
    def _normalize_counts(
        cls,
        values: Mapping[str, int],
        *,
        field_name: str,
        allow_zero: bool,
    ) -> dict[str, int]:
        if not isinstance(values, Mapping):
            raise TypeError(
                f"{field_name} must be a mapping"
            )

        if not values:
            raise ValueError(
                f"{field_name} must contain at least "
                "one item"
            )

        normalized: dict[str, int] = {}

        for raw_component, raw_value in values.items():
            if not isinstance(raw_component, str):
                raise TypeError(
                    f"{field_name} component names "
                    "must be strings"
                )

            component = raw_component.strip()

            if not component:
                raise ValueError(
                    f"{field_name} component names "
                    "must not be empty"
                )

            if component not in cls.COMPONENTS:
                raise ValueError(
                    f"unknown reward component: "
                    f"{component}"
                )

            if component in normalized:
                raise ValueError(
                    f"duplicate reward component: "
                    f"{component}"
                )

            if isinstance(raw_value, bool):
                raise TypeError(
                    f"{field_name} for '{component}' "
                    "must be an integer"
                )

            if not isinstance(raw_value, int):
                raise TypeError(
                    f"{field_name} for '{component}' "
                    "must be an integer"
                )

            minimum = 0 if allow_zero else 1

            if raw_value < minimum:
                comparison = (
                    "greater than or equal to 0"
                    if allow_zero
                    else "greater than or equal to 1"
                )
                raise ValueError(
                    f"{field_name} for '{component}' "
                    f"must be {comparison}"
                )

            normalized[component] = raw_value

        return normalized

    @staticmethod
    def _normalize_baseline(
        baseline_rate: float,
    ) -> float:
        if isinstance(baseline_rate, bool):
            raise TypeError(
                "baseline_rate must be numeric"
            )

        if not isinstance(
            baseline_rate,
            (int, float),
        ):
            raise TypeError(
                "baseline_rate must be numeric"
            )

        value = float(baseline_rate)

        if not isfinite(value):
            raise ValueError(
                "baseline_rate must be finite"
            )

        if not 0.0 < value < 1.0:
            raise ValueError(
                "baseline_rate must be between "
                "0.0 and 1.0 exclusive"
            )

        return value

    @staticmethod
    def _validate_component_sets(
        *,
        successes: Mapping[str, int],
        trials: Mapping[str, int],
    ) -> None:
        success_components = set(successes)
        trial_components = set(trials)

        missing_successes = sorted(
            trial_components - success_components
        )
        missing_trials = sorted(
            success_components - trial_components
        )

        if missing_successes:
            raise ValueError(
                "missing success counts for: "
                + ", ".join(missing_successes)
            )

        if missing_trials:
            raise ValueError(
                "missing trial counts for: "
                + ", ".join(missing_trials)
            )

    @staticmethod
    def _validate_success_bounds(
        *,
        successes: Mapping[str, int],
        trials: Mapping[str, int],
    ) -> None:
        for component, success_count in (
            successes.items()
        ):
            if success_count > trials[component]:
                raise ValueError(
                    f"successes for '{component}' "
                    "must not exceed trials"
                )
