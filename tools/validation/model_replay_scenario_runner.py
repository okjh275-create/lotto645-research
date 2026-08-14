"""Execute one Project M replay scenario with existing replay infrastructure."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .historical_replay_executor import (
    HistoricalReplayExecutor,
)
from .historical_replay_models import (
    ReplayConfig,
)
from .historical_replay_runner import (
    HistoricalReplayRunner,
)
from .regime_learning_comparison_runner import (
    RegimeLearningScenario,
)


ExecutorFactory = Callable[..., object]
RunnerFactory = Callable[..., object]


class HistoricalModelReplayScenarioRunner:
    """Build and execute one existing regime-learning replay scenario."""

    def __init__(
        self,
        *,
        history: tuple[object, ...],
        executor_factory: ExecutorFactory = HistoricalReplayExecutor,
        runner_factory: RunnerFactory = HistoricalReplayRunner,
    ) -> None:
        normalized_history = tuple(history)

        if not normalized_history:
            raise ValueError(
                "history must not be empty"
            )

        if not callable(executor_factory):
            raise TypeError(
                "executor_factory must be callable"
            )

        if not callable(runner_factory):
            raise TypeError(
                "runner_factory must be callable"
            )

        rounds = tuple(
            self._round_no(row)
            for row in normalized_history
        )

        if len(rounds) != len(set(rounds)):
            raise ValueError(
                "history round numbers must be unique"
            )

        self._history = normalized_history
        self._draw_by_round = {
            round_no: row
            for round_no, row in zip(
                rounds,
                normalized_history,
                strict=True,
            )
        }
        self._executor_factory = executor_factory
        self._runner_factory = runner_factory

    def __call__(
        self,
        *,
        scenario: RegimeLearningScenario,
        config: ReplayConfig,
        output_root: str | Path,
    ) -> object:
        if not isinstance(
            scenario,
            RegimeLearningScenario,
        ):
            raise TypeError(
                "scenario must be RegimeLearningScenario"
            )

        if not isinstance(
            config,
            ReplayConfig,
        ):
            raise TypeError(
                "config must be ReplayConfig"
            )

        root = Path(output_root)

        calibration_root = (
            root / "regime-calibration"
            if scenario.calibration_enabled
            else None
        )

        bayesian_root = (
            root / "regime-bayesian"
            if scenario.bayesian_enabled
            else None
        )

        executor = self._executor_factory(
            history=self._history,
            config=config,
            learning_root=(
                root / "learning"
            ),
            profile_root=(
                root / "profiles"
            ),
            regime_calibration_root=(
                calibration_root
            ),
            regime_bayesian_root=(
                bayesian_root
            ),
        )

        runner = self._runner_factory(
            executor=executor
        )

        return runner.run(
            config=config,
            draw_by_round=self._draw_by_round,
            output_root=root,
        )

    @staticmethod
    def _round_no(
        row: object,
    ) -> int:
        value = getattr(
            row,
            "round_no",
            None,
        )

        if value is None:
            value = getattr(
                row,
                "round",
                None,
            )

        if value is None and isinstance(
            row,
            dict,
        ):
            value = row.get(
                "round_no",
                row.get("round"),
            )

        if value is None:
            raise TypeError(
                "history rows must expose round_no or round"
            )

        return int(value)
