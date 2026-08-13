"""Cross-window model evaluation matrix for Project M."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from lrp.contracts import ContractError
from lrp.evaluation import (
    ChampionRanking,
    EvaluationWindow,
    ModelEvaluation,
    WindowEvaluation,
    build_model_evaluation,
    rank_model_evaluations,
)


@dataclass(frozen=True, slots=True)
class HistoricalEvaluationMatrix:
    """Complete model-by-window evaluation matrix."""

    windows: tuple[EvaluationWindow, ...]
    evaluations: tuple[ModelEvaluation, ...]
    ranking: ChampionRanking

    @property
    def model_names(self) -> tuple[str, ...]:
        return tuple(
            item.model_name
            for item in self.evaluations
        )

    @property
    def window_names(self) -> tuple[str, ...]:
        return tuple(
            window.name
            for window in self.windows
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "windows": [
                window.as_dict()
                for window in self.windows
            ],
            "evaluations": [
                item.as_dict()
                for item in self.evaluations
            ],
            "ranking": self.ranking.as_dict(),
            "model_names": list(self.model_names),
            "window_names": list(self.window_names),
        }


def build_evaluation_matrix(
    *,
    windows: Iterable[EvaluationWindow],
    results: Iterable[
        tuple[str, WindowEvaluation]
    ],
) -> HistoricalEvaluationMatrix:
    """Aggregate complete model-by-window evaluation cells."""

    normalized_windows = tuple(windows)
    normalized_results = tuple(results)

    if not normalized_windows:
        raise ContractError(
            "windows must not be empty"
        )

    if any(
        not isinstance(window, EvaluationWindow)
        for window in normalized_windows
    ):
        raise TypeError(
            "windows must contain EvaluationWindow values"
        )

    window_names = tuple(
        window.name
        for window in normalized_windows
    )

    if len(window_names) != len(set(window_names)):
        raise ContractError(
            "evaluation window names must be unique"
        )

    windows_by_name = {
        window.name: window
        for window in normalized_windows
    }

    if not normalized_results:
        raise ContractError(
            "results must not be empty"
        )

    cells: dict[
        tuple[str, str],
        WindowEvaluation,
    ] = {}

    model_names: set[str] = set()

    for item in normalized_results:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
        ):
            raise TypeError(
                "each result must be "
                "(model_name, WindowEvaluation)"
            )

        model_name, evaluation = item

        if (
            not isinstance(model_name, str)
            or not model_name.strip()
        ):
            raise ContractError(
                "model_name must not be empty"
            )

        if not isinstance(
            evaluation,
            WindowEvaluation,
        ):
            raise TypeError(
                "result evaluation must be WindowEvaluation"
            )

        window_name = evaluation.window.name

        expected_window = windows_by_name.get(
            window_name
        )

        if expected_window is None:
            raise ContractError(
                "result references unknown evaluation window"
            )

        if evaluation.window != expected_window:
            raise ContractError(
                "result window does not match "
                "matrix evaluation window"
            )

        key = (
            model_name,
            window_name,
        )

        if key in cells:
            raise ContractError(
                "duplicate model-window cell"
            )

        cells[key] = evaluation
        model_names.add(model_name)

    ordered_models = tuple(
        sorted(model_names)
    )

    expected_cells = {
        (
            model_name,
            window.name,
        )
        for model_name in ordered_models
        for window in normalized_windows
    }

    if set(cells) != expected_cells:
        raise ContractError(
            "complete model-window coverage is required"
        )

    evaluations = tuple(
        build_model_evaluation(
            model_name=model_name,
            windows=(
                cells[
                    (
                        model_name,
                        window.name,
                    )
                ]
                for window in normalized_windows
            ),
        )
        for model_name in ordered_models
    )

    ranking = rank_model_evaluations(
        evaluations
    )

    return HistoricalEvaluationMatrix(
        windows=normalized_windows,
        evaluations=evaluations,
        ranking=ranking,
    )
