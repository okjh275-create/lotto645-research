"""Prediction-model evaluation contracts for Project M."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Iterable

from lrp.contracts import ContractError


@dataclass(frozen=True, slots=True)
class EvaluationWindow:
    """One contiguous evaluation window."""

    name: str
    start_round: int
    end_round: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ContractError(
                "evaluation window name must not be empty"
            )

        for field_name in ("start_round", "end_round"):
            value = getattr(self, field_name)

            if isinstance(value, bool) or not isinstance(value, int):
                raise ContractError(
                    f"{field_name} must be an integer"
                )

        if self.start_round <= 1:
            raise ContractError(
                "start_round must be greater than one"
            )

        if self.end_round < self.start_round:
            raise ContractError(
                "end_round must be greater than or equal to start_round"
            )

    @property
    def round_count(self) -> int:
        return self.end_round - self.start_round + 1

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["round_count"] = self.round_count
        return payload


@dataclass(frozen=True, slots=True)
class WindowEvaluation:
    """Performance of one model in one evaluation window."""

    window: EvaluationWindow
    round_count: int

    average_best_hits: float
    average_practical_hits: float

    baseline_best_hit_delta: float
    baseline_practical_hit_delta: float

    best_hit_stddev: float
    practical_hit_stddev: float

    best_win_count: int
    best_loss_count: int
    best_tie_count: int
    best_sign_test_pvalue: float

    practical_win_count: int
    practical_loss_count: int
    practical_tie_count: int
    practical_sign_test_pvalue: float

    average_jaccard: float

    def __post_init__(self) -> None:
        if not isinstance(self.window, EvaluationWindow):
            raise TypeError(
                "window must be EvaluationWindow"
            )

        if (
            isinstance(self.round_count, bool)
            or not isinstance(self.round_count, int)
            or self.round_count < 1
        ):
            raise ContractError(
                "round_count must be a positive integer"
            )

        if self.round_count != self.window.round_count:
            raise ContractError(
                "round_count must match evaluation window"
            )

        for field_name in (
            "average_best_hits",
            "average_practical_hits",
            "baseline_best_hit_delta",
            "baseline_practical_hit_delta",
            "best_hit_stddev",
            "practical_hit_stddev",
            "best_sign_test_pvalue",
            "practical_sign_test_pvalue",
            "average_jaccard",
        ):
            value = getattr(self, field_name)

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
            ):
                raise ContractError(
                    f"{field_name} must be finite numeric"
                )

        for field_name in (
            "average_best_hits",
            "average_practical_hits",
        ):
            value = float(getattr(self, field_name))

            if not 0.0 <= value <= 6.0:
                raise ContractError(
                    f"{field_name} must be between 0 and 6"
                )

        for field_name in (
            "best_hit_stddev",
            "practical_hit_stddev",
            "average_jaccard",
        ):
            if float(getattr(self, field_name)) < 0.0:
                raise ContractError(
                    f"{field_name} must be non-negative"
                )

        if float(self.average_jaccard) > 1.0:
            raise ContractError(
                "average_jaccard must not exceed 1"
            )

        for field_name in (
            "best_sign_test_pvalue",
            "practical_sign_test_pvalue",
        ):
            value = float(getattr(self, field_name))

            if not 0.0 <= value <= 1.0:
                raise ContractError(
                    f"{field_name} must be between 0 and 1"
                )

        for field_name in (
            "best_win_count",
            "best_loss_count",
            "best_tie_count",
            "practical_win_count",
            "practical_loss_count",
            "practical_tie_count",
        ):
            value = getattr(self, field_name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ContractError(
                    f"{field_name} must be a non-negative integer"
                )

        if (
            self.best_win_count
            + self.best_loss_count
            + self.best_tie_count
            != self.round_count
        ):
            raise ContractError(
                "best outcome counts must equal round_count"
            )

        if (
            self.practical_win_count
            + self.practical_loss_count
            + self.practical_tie_count
            != self.round_count
        ):
            raise ContractError(
                "practical outcome counts must equal round_count"
            )

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["window"] = self.window.as_dict()
        return payload


@dataclass(frozen=True, slots=True)
class ModelEvaluation:
    """Cross-window evaluation for one prediction model."""

    model_name: str
    windows: tuple[WindowEvaluation, ...]

    mean_best_hit_delta: float
    mean_practical_hit_delta: float

    worst_best_hit_delta: float
    worst_practical_hit_delta: float

    significant_best_window_count: int
    significant_practical_window_count: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.model_name, str)
            or not self.model_name.strip()
        ):
            raise ContractError(
                "model_name must not be empty"
            )

        if not self.windows:
            raise ContractError(
                "model evaluation requires at least one window"
            )

        if any(
            not isinstance(window, WindowEvaluation)
            for window in self.windows
        ):
            raise TypeError(
                "windows must contain WindowEvaluation values"
            )

        names = tuple(
            window.window.name
            for window in self.windows
        )

        if len(names) != len(set(names)):
            raise ContractError(
                "evaluation window names must be unique"
            )

        for field_name in (
            "mean_best_hit_delta",
            "mean_practical_hit_delta",
            "worst_best_hit_delta",
            "worst_practical_hit_delta",
        ):
            value = getattr(self, field_name)

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
            ):
                raise ContractError(
                    f"{field_name} must be finite numeric"
                )

        for field_name in (
            "significant_best_window_count",
            "significant_practical_window_count",
        ):
            value = getattr(self, field_name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value > len(self.windows)
            ):
                raise ContractError(
                    f"{field_name} is outside valid range"
                )

    @property
    def total_round_count(self) -> int:
        return sum(
            window.round_count
            for window in self.windows
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "windows": [
                window.as_dict()
                for window in self.windows
            ],
            "mean_best_hit_delta": self.mean_best_hit_delta,
            "mean_practical_hit_delta": (
                self.mean_practical_hit_delta
            ),
            "worst_best_hit_delta": self.worst_best_hit_delta,
            "worst_practical_hit_delta": (
                self.worst_practical_hit_delta
            ),
            "significant_best_window_count": (
                self.significant_best_window_count
            ),
            "significant_practical_window_count": (
                self.significant_practical_window_count
            ),
            "total_round_count": self.total_round_count,
        }


def build_model_evaluation(
    *,
    model_name: str,
    windows: Iterable[WindowEvaluation],
    significance_level: float = 0.05,
) -> ModelEvaluation:
    """Aggregate immutable window evaluations."""

    normalized = tuple(windows)

    if not normalized:
        raise ContractError(
            "windows must not be empty"
        )

    if (
        isinstance(significance_level, bool)
        or not isinstance(significance_level, (int, float))
        or not isfinite(float(significance_level))
        or not 0.0 < float(significance_level) < 1.0
    ):
        raise ContractError(
            "significance_level must be between 0 and 1"
        )

    return ModelEvaluation(
        model_name=model_name,
        windows=normalized,
        mean_best_hit_delta=(
            sum(
                window.baseline_best_hit_delta
                for window in normalized
            )
            / len(normalized)
        ),
        mean_practical_hit_delta=(
            sum(
                window.baseline_practical_hit_delta
                for window in normalized
            )
            / len(normalized)
        ),
        worst_best_hit_delta=min(
            window.baseline_best_hit_delta
            for window in normalized
        ),
        worst_practical_hit_delta=min(
            window.baseline_practical_hit_delta
            for window in normalized
        ),
        significant_best_window_count=sum(
            window.baseline_best_hit_delta > 0.0
            and window.best_sign_test_pvalue
            < significance_level
            for window in normalized
        ),
        significant_practical_window_count=sum(
            window.baseline_practical_hit_delta > 0.0
            and window.practical_sign_test_pvalue
            < significance_level
            for window in normalized
        ),
    )
