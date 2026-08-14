"""Resolve Project M model names to replay scenarios."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .historical_replay_models import ReplayConfig
from .regime_learning_comparison_runner import (
    RegimeLearningScenario,
    default_regime_learning_scenarios,
)


ScenarioRunner = Callable[..., Any]


class HistoricalModelReplayExecuteAdapter:
    """Translate model names into existing replay scenarios."""

    def __init__(
        self,
        *,
        run_scenario: ScenarioRunner,
        scenarios: tuple[
            RegimeLearningScenario,
            ...,
        ] | None = None,
    ) -> None:
        if not callable(run_scenario):
            raise TypeError(
                "run_scenario must be callable"
            )

        normalized_scenarios = (
            default_regime_learning_scenarios()
            if scenarios is None
            else tuple(scenarios)
        )

        if not normalized_scenarios:
            raise ValueError(
                "scenarios must not be empty"
            )

        if any(
            not isinstance(
                scenario,
                RegimeLearningScenario,
            )
            for scenario in normalized_scenarios
        ):
            raise TypeError(
                "scenarios must contain "
                "RegimeLearningScenario values"
            )

        names = tuple(
            scenario.name
            for scenario in normalized_scenarios
        )

        if len(names) != len(set(names)):
            raise ValueError(
                "scenario names must be unique"
            )

        self._run_scenario = run_scenario
        self._scenarios = {
            scenario.name: scenario
            for scenario in normalized_scenarios
        }

    @property
    def model_names(self) -> tuple[str, ...]:
        return tuple(
            self._scenarios
        )

    def __call__(
        self,
        *,
        model_name: str,
        config: ReplayConfig,
        output_root: str | Path,
    ) -> Any:
        if (
            not isinstance(model_name, str)
            or not model_name.strip()
        ):
            raise ValueError(
                "model_name must be a non-empty string"
            )

        if not isinstance(
            config,
            ReplayConfig,
        ):
            raise TypeError(
                "config must be ReplayConfig"
            )

        scenario = self._scenarios.get(
            model_name
        )

        if scenario is None:
            raise ValueError(
                "unsupported model_name: "
                f"{model_name}"
            )

        return self._run_scenario(
            scenario=scenario,
            config=config,
            output_root=Path(output_root),
        )
