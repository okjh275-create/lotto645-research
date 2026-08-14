"""Adapter from model/window replay rows to Project M evaluation."""

from __future__ import annotations

from collections.abc import Callable

from lrp.evaluation import (
    EvaluationWindow,
    WindowEvaluation,
)

from .historical_replay_models import (
    ReplayRoundResult,
)
from .model_evaluation_adapter import (
    window_evaluation_from_replay,
)


ReplayRowsProvider = Callable[
    [str, EvaluationWindow],
    tuple[ReplayRoundResult, ...],
]


class HistoricalModelReplayEvaluator:
    """Evaluate one model over one historical window."""

    def __init__(
        self,
        *,
        replay_rows: ReplayRowsProvider,
    ) -> None:
        if not callable(replay_rows):
            raise TypeError(
                "replay_rows must be callable"
            )

        self._replay_rows = replay_rows

    def __call__(
        self,
        model_name: str,
        window: EvaluationWindow,
    ) -> WindowEvaluation:
        if (
            not isinstance(model_name, str)
            or not model_name.strip()
        ):
            raise ValueError(
                "model_name must be a non-empty string"
            )

        if not isinstance(
            window,
            EvaluationWindow,
        ):
            raise TypeError(
                "window must be EvaluationWindow"
            )

        rows = self._replay_rows(
            model_name,
            window,
        )

        if not isinstance(rows, tuple):
            raise TypeError(
                "replay_rows must return a tuple"
            )

        if any(
            not isinstance(row, ReplayRoundResult)
            for row in rows
        ):
            raise TypeError(
                "replay rows must contain ReplayRoundResult values"
            )

        return window_evaluation_from_replay(
            window=window,
            rows=rows,
        )
