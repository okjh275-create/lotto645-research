"""Historical cross-window model evaluation runner for Project M."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from lrp.evaluation import (
    EvaluationWindow,
    WindowEvaluation,
)

from .model_evaluation_matrix import (
    HistoricalEvaluationMatrix,
    build_evaluation_matrix,
)


WindowEvaluator = Callable[
    [str, EvaluationWindow],
    WindowEvaluation,
]


class HistoricalModelEvaluationRunner:
    """Execute a model-by-window evaluation matrix."""

    def __init__(
        self,
        *,
        evaluator: WindowEvaluator,
    ) -> None:
        if not callable(evaluator):
            raise TypeError(
                "evaluator must be callable"
            )

        self._evaluator = evaluator

    def run(
        self,
        *,
        model_names: Iterable[str],
        windows: Iterable[EvaluationWindow],
    ) -> HistoricalEvaluationMatrix:
        """Evaluate every model-window pair and rank the models."""

        normalized_models = tuple(model_names)
        normalized_windows = tuple(windows)

        if not normalized_models:
            raise ValueError(
                "model_names must not be empty"
            )

        if not normalized_windows:
            raise ValueError(
                "windows must not be empty"
            )

        if any(
            not isinstance(model_name, str)
            or not model_name.strip()
            for model_name in normalized_models
        ):
            raise ValueError(
                "model_names must contain non-empty strings"
            )

        if len(normalized_models) != len(
            set(normalized_models)
        ):
            raise ValueError(
                "model_names must be unique"
            )

        if any(
            not isinstance(window, EvaluationWindow)
            for window in normalized_windows
        ):
            raise TypeError(
                "windows must contain EvaluationWindow values"
            )

        results: list[
            tuple[str, WindowEvaluation]
        ] = []

        for model_name in normalized_models:
            for window in normalized_windows:
                evaluation = self._evaluator(
                    model_name,
                    window,
                )

                if not isinstance(
                    evaluation,
                    WindowEvaluation,
                ):
                    raise TypeError(
                        "evaluator must return WindowEvaluation"
                    )

                if evaluation.window != window:
                    raise ValueError(
                        "evaluator result window must match "
                        "requested evaluation window"
                    )

                results.append(
                    (
                        model_name,
                        evaluation,
                    )
                )

        return build_evaluation_matrix(
            windows=normalized_windows,
            results=results,
        )
