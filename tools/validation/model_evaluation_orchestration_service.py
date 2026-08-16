from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from lrp.evaluation import (
    ChampionPromotionPolicy,
    EvaluationWindow,
)

from tools.validation.champion_decision_reporting_service import (
    ChampionDecisionReportingService,
)
from tools.validation.historical_replay_models import (
    ReplayConfig,
)
from tools.validation.model_evaluation_champion import (
    select_historical_champion,
)
from tools.validation.model_evaluation_runner import (
    HistoricalModelEvaluationRunner,
)
from tools.validation.model_replay_evaluator import (
    HistoricalModelReplayEvaluator,
)
from tools.validation.model_replay_execute_adapter import (
    HistoricalModelReplayExecuteAdapter,
)
from tools.validation.model_replay_provider import (
    HistoricalModelReplayProvider,
)
from tools.validation.model_replay_scenario_runner import (
    HistoricalModelReplayScenarioRunner,
)


@dataclass(frozen=True)
class HistoricalModelEvaluationOrchestrationResult:
    matrix: object
    champion: object
    artifact_path: Path


class HistoricalModelEvaluationOrchestrationService:
    def __init__(
        self,
        *,
        scenario_runner_factory=(
            HistoricalModelReplayScenarioRunner
        ),
        execute_adapter_factory=(
            HistoricalModelReplayExecuteAdapter
        ),
        provider_factory=HistoricalModelReplayProvider,
        evaluator_factory=HistoricalModelReplayEvaluator,
        evaluation_runner_factory=(
            HistoricalModelEvaluationRunner
        ),
        champion_selector: Callable[..., object] = (
            select_historical_champion
        ),
        reporting_service=None,
    ) -> None:
        self._scenario_runner_factory = (
            scenario_runner_factory
        )
        self._execute_adapter_factory = (
            execute_adapter_factory
        )
        self._provider_factory = provider_factory
        self._evaluator_factory = evaluator_factory
        self._evaluation_runner_factory = (
            evaluation_runner_factory
        )
        self._champion_selector = champion_selector
        self._reporting_service = (
            reporting_service
            if reporting_service is not None
            else ChampionDecisionReportingService()
        )

    def run(
        self,
        *,
        history: Iterable[object],
        model_names: Iterable[str],
        windows: Iterable[EvaluationWindow],
        replay_output_root: str | Path,
        report_output_root: str | Path,
        base_config: ReplayConfig,
        policy: ChampionPromotionPolicy | None = None,
    ) -> HistoricalModelEvaluationOrchestrationResult:
        history_tuple = tuple(history)

        if not history_tuple:
            raise ValueError(
                "history must not be empty"
            )

        model_names_tuple = tuple(model_names)

        if not model_names_tuple:
            raise ValueError(
                "model_names must not be empty"
            )

        windows_tuple = tuple(windows)

        if not windows_tuple:
            raise ValueError(
                "windows must not be empty"
            )

        scenario_runner = self._scenario_runner_factory(
            history=history_tuple,
        )

        execute_adapter = self._execute_adapter_factory(
            run_scenario=scenario_runner,
        )

        provider = self._provider_factory(
            execute=execute_adapter,
            output_root=Path(replay_output_root),
            base_config=base_config,
        )

        evaluator = self._evaluator_factory(
            replay_rows=provider,
        )

        evaluation_runner = (
            self._evaluation_runner_factory(
                evaluator=evaluator,
            )
        )

        matrix = evaluation_runner.run(
            model_names=model_names_tuple,
            windows=windows_tuple,
        )

        champion = self._champion_selector(
            matrix=matrix,
            policy=policy,
        )

        artifact_path = self._reporting_service.write(
            report=champion,
            output_root=Path(report_output_root),
        )

        return (
            HistoricalModelEvaluationOrchestrationResult(
                matrix=matrix,
                champion=champion,
                artifact_path=artifact_path,
            )
        )
