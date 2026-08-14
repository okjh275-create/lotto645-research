from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from tools.validation.historical_replay_models import (
    ReplayConfig,
)
from tools.validation.historical_replay_runner import (
    HistoricalReplayResult,
)
from tools.validation.regime_learning_comparison_runner import (
    RegimeLearningScenario,
)

from tools.validation.model_replay_scenario_runner import (
    HistoricalModelReplayScenarioRunner,
)


@dataclass(frozen=True)
class FakeDraw:
    round_no: int


class FakeReplayRunner:
    def __init__(
        self,
        *,
        executor: object,
        seen: list,
    ) -> None:
        self.executor = executor
        self.seen = seen

    def run(
        self,
        *,
        config: ReplayConfig,
        draw_by_round: dict[int, object],
        output_root: Path,
        initial_state: object | None = None,
        overwrite: bool = False,
    ) -> object:
        self.seen.append(
            {
                "executor": self.executor,
                "config": config,
                "draw_by_round": dict(draw_by_round),
                "output_root": Path(output_root),
                "initial_state": initial_state,
                "overwrite": overwrite,
            }
        )

        return object()


def test_scenario_runner_builds_baseline_executor(
    tmp_path: Path,
) -> None:
    executor_calls = []
    runner_calls = []

    def executor_factory(**kwargs):
        executor_calls.append(kwargs)
        return object()

    def runner_factory(*, executor):
        return FakeReplayRunner(
            executor=executor,
            seen=runner_calls,
        )

    runner = HistoricalModelReplayScenarioRunner(
        history=(
            FakeDraw(1219),
            FakeDraw(1220),
        ),
        executor_factory=executor_factory,
        runner_factory=runner_factory,
    )

    config = ReplayConfig(
        start_round=1220,
        end_round=1220,
    )

    result = runner(
        scenario=RegimeLearningScenario(
            name="baseline",
            calibration_enabled=False,
            bayesian_enabled=False,
        ),
        config=config,
        output_root=tmp_path,
    )

    assert result is not None
    assert len(executor_calls) == 1
    assert len(runner_calls) == 1

    executor_args = executor_calls[0]

    assert executor_args["config"] is config
    assert executor_args["history"] == (
        FakeDraw(1219),
        FakeDraw(1220),
    )

    assert executor_args["learning_root"] == (
        tmp_path / "learning"
    )
    assert executor_args["profile_root"] == (
        tmp_path / "profiles"
    )

    assert (
        executor_args["regime_calibration_root"]
        is None
    )
    assert (
        executor_args["regime_bayesian_root"]
        is None
    )

    run_args = runner_calls[0]

    assert run_args["config"] is config
    assert run_args["output_root"] == tmp_path
    assert run_args["draw_by_round"] == {
        1219: FakeDraw(1219),
        1220: FakeDraw(1220),
    }


def test_scenario_runner_builds_combined_roots(
    tmp_path: Path,
) -> None:
    executor_calls = []

    def executor_factory(**kwargs):
        executor_calls.append(kwargs)
        return object()

    class Runner:
        def __init__(self, *, executor):
            self.executor = executor

        def run(self, **kwargs):
            return object()

    runner = HistoricalModelReplayScenarioRunner(
        history=(
            FakeDraw(1219),
            FakeDraw(1220),
        ),
        executor_factory=executor_factory,
        runner_factory=Runner,
    )

    runner(
        scenario=RegimeLearningScenario(
            name="combined",
            calibration_enabled=True,
            bayesian_enabled=True,
        ),
        config=ReplayConfig(
            start_round=1220,
            end_round=1220,
        ),
        output_root=tmp_path,
    )

    args = executor_calls[0]

    assert args["regime_calibration_root"] == (
        tmp_path / "regime-calibration"
    )
    assert args["regime_bayesian_root"] == (
        tmp_path / "regime-bayesian"
    )


def test_scenario_runner_rejects_empty_history() -> None:
    with pytest.raises(
        ValueError,
        match="history must not be empty",
    ):
        HistoricalModelReplayScenarioRunner(
            history=(),
        )


def test_scenario_runner_requires_scenario(
    tmp_path: Path,
) -> None:
    runner = HistoricalModelReplayScenarioRunner(
        history=(FakeDraw(1220),),
    )

    with pytest.raises(
        TypeError,
        match="RegimeLearningScenario",
    ):
        runner(
            scenario=object(),
            config=ReplayConfig(
                start_round=1220,
                end_round=1220,
            ),
            output_root=tmp_path,
        )


def test_scenario_runner_requires_replay_config(
    tmp_path: Path,
) -> None:
    runner = HistoricalModelReplayScenarioRunner(
        history=(FakeDraw(1220),),
    )

    with pytest.raises(
        TypeError,
        match="ReplayConfig",
    ):
        runner(
            scenario=RegimeLearningScenario(
                name="baseline",
                calibration_enabled=False,
                bayesian_enabled=False,
            ),
            config=object(),
            output_root=tmp_path,
        )


def test_scenario_runner_rejects_duplicate_rounds() -> None:
    with pytest.raises(
        ValueError,
        match="history round numbers must be unique",
    ):
        HistoricalModelReplayScenarioRunner(
            history=(
                FakeDraw(1220),
                FakeDraw(1220),
            ),
        )
