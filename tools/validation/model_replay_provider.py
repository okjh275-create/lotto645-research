"""Replay-row provider for Project M historical model evaluation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from lrp.evaluation import EvaluationWindow

from .historical_replay_models import (
    ReplayConfig,
    ReplayRoundResult,
)
from .historical_replay_runner import (
    HistoricalReplayResult,
)


ReplayExecutor = Callable[..., HistoricalReplayResult]


class HistoricalModelReplayProvider:
    """Execute one model over one evaluation window."""

    def __init__(
        self,
        *,
        execute: ReplayExecutor,
        output_root: str | Path,
        base_config: ReplayConfig | None = None,
    ) -> None:
        if not callable(execute):
            raise TypeError(
                "execute must be callable"
            )

        if (
            base_config is not None
            and not isinstance(
                base_config,
                ReplayConfig,
            )
        ):
            raise TypeError(
                "base_config must be ReplayConfig or None"
            )

        self._execute = execute
        self._output_root = Path(output_root)

        self._base_config = (
            base_config
            if base_config is not None
            else ReplayConfig(
                start_round=2,
                end_round=2,
            )
        )

    def __call__(
        self,
        model_name: str,
        window: EvaluationWindow,
    ) -> tuple[ReplayRoundResult, ...]:
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

        config = replace(
            self._base_config,
            start_round=window.start_round,
            end_round=window.end_round,
        )

        output_root = (
            self._output_root
            / model_name
            / window.name
        )

        result = self._execute(
            model_name=model_name,
            config=config,
            output_root=output_root,
        )

        if not isinstance(
            result,
            HistoricalReplayResult,
        ):
            raise TypeError(
                "execute must return HistoricalReplayResult"
            )

        return result.rounds
