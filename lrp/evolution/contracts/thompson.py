from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class BetaArmStatistics:
    """Beta posterior statistics for one bandit arm."""

    arm: str
    successes: float = 0.0
    failures: float = 0.0
    prior_alpha: float = 1.0
    prior_beta: float = 1.0

    def __post_init__(self) -> None:
        arm = self._normalize_arm(self.arm)
        successes = self._normalize_non_negative(
            self.successes,
            field_name="successes",
        )
        failures = self._normalize_non_negative(
            self.failures,
            field_name="failures",
        )
        prior_alpha = self._normalize_positive(
            self.prior_alpha,
            field_name="prior_alpha",
        )
        prior_beta = self._normalize_positive(
            self.prior_beta,
            field_name="prior_beta",
        )

        object.__setattr__(self, "arm", arm)
        object.__setattr__(self, "successes", successes)
        object.__setattr__(self, "failures", failures)
        object.__setattr__(self, "prior_alpha", prior_alpha)
        object.__setattr__(self, "prior_beta", prior_beta)

    @property
    def alpha(self) -> float:
        return self.prior_alpha + self.successes

    @property
    def beta(self) -> float:
        return self.prior_beta + self.failures

    @property
    def observations(self) -> float:
        return self.successes + self.failures

    @property
    def posterior_mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def record(self, reward: float) -> BetaArmStatistics:
        value = self._normalize_reward(reward)

        return BetaArmStatistics(
            arm=self.arm,
            successes=self.successes + value,
            failures=self.failures + (1.0 - value),
            prior_alpha=self.prior_alpha,
            prior_beta=self.prior_beta,
        )

    @staticmethod
    def _normalize_arm(arm: str) -> str:
        if not isinstance(arm, str):
            raise TypeError("arm must be a string")

        normalized = arm.strip()

        if not normalized:
            raise ValueError("arm must not be empty")

        return normalized

    @staticmethod
    def _normalize_non_negative(
        value: float,
        *,
        field_name: str,
    ) -> float:
        if isinstance(value, bool):
            raise TypeError(f"{field_name} must be numeric")

        if not isinstance(value, (int, float)):
            raise TypeError(f"{field_name} must be numeric")

        normalized = float(value)

        if not isfinite(normalized):
            raise ValueError(f"{field_name} must be finite")

        if normalized < 0.0:
            raise ValueError(
                f"{field_name} must be greater than "
                "or equal to 0"
            )

        return normalized

    @staticmethod
    def _normalize_positive(
        value: float,
        *,
        field_name: str,
    ) -> float:
        normalized = BetaArmStatistics._normalize_non_negative(
            value,
            field_name=field_name,
        )

        if normalized <= 0.0:
            raise ValueError(
                f"{field_name} must be greater than 0"
            )

        return normalized

    @staticmethod
    def _normalize_reward(reward: float) -> float:
        if isinstance(reward, bool):
            raise TypeError("reward must be numeric")

        if not isinstance(reward, (int, float)):
            raise TypeError("reward must be numeric")

        value = float(reward)

        if not isfinite(value):
            raise ValueError("reward must be finite")

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "reward must be between 0.0 and 1.0 inclusive"
            )

        return value


@dataclass(frozen=True, slots=True)
class ThompsonDecision:
    """Selection result from Thompson Sampling."""

    arm: str
    sample: float
    seed: int | None
    observations: float

    def __post_init__(self) -> None:
        normalized_arm = BetaArmStatistics._normalize_arm(
            self.arm
        )

        if isinstance(self.sample, bool):
            raise TypeError("sample must be numeric")

        if not isinstance(self.sample, (int, float)):
            raise TypeError("sample must be numeric")

        sample = float(self.sample)

        if not isfinite(sample):
            raise ValueError("sample must be finite")

        if not 0.0 <= sample <= 1.0:
            raise ValueError(
                "sample must be between 0.0 and 1.0 inclusive"
            )

        if self.seed is not None:
            if isinstance(self.seed, bool):
                raise TypeError("seed must be an integer or None")

            if not isinstance(self.seed, int):
                raise TypeError("seed must be an integer or None")

        observations = (
            BetaArmStatistics._normalize_non_negative(
                self.observations,
                field_name="observations",
            )
        )

        object.__setattr__(self, "arm", normalized_arm)
        object.__setattr__(self, "sample", sample)
        object.__setattr__(
            self,
            "observations",
            observations,
        )
