from __future__ import annotations

from pathlib import Path

import pytest

from lrp.evaluation import (
    ChampionPromotionPolicy,
    EvaluationWindow,
)

from tools.validation.historical_replay_models import (
    ReplayConfig,
)


def _window(
    name: str,
    start_round: int,
    end_round: int,
) -> EvaluationWindow:
    return EvaluationWindow(
        name=name,
        start_round=start_round,
        end_round=end_round,
    )


def _base_config() -> ReplayConfig:
    return ReplayConfig(
        start_round=1228,
        end_round=1228,
        seed_base=20260802,
        temperature=0.85,
        candidate_count=100,
        top_k=10,
        practical_k=5,
        long_gap_window=5,
        confidence=0.8,
        mode="fast",
    )


def test_service_runs_full_orchestration(
    tmp_path: Path,
) -> None:
    from tools.validation.model_evaluation_orchestration_service import (
        HistoricalModelEvaluationOrchestrationService,
    )

    history = (
        object(),
        object(),
    )

    windows = (
        _window(
            "recent",
            1229,
            1230,
        ),
    )

    calls: list[
        tuple[str, object]
    ] = []

    matrix = object()
    champion = object()
    artifact_path = (
        tmp_path /
        "champion_decision.json"
    )

    class FakeScenarioRunner:
        def __init__(
            self,
            *,
            history: tuple[object, ...],
        ) -> None:
            calls.append(
                (
                    "scenario_runner",
                    history,
                )
            )

    class FakeExecuteAdapter:
        def __init__(
            self,
            *,
            run_scenario: object,
        ) -> None:
            calls.append(
                (
                    "execute_adapter",
                    run_scenario,
                )
            )

    class FakeProvider:
        def __init__(
            self,
            *,
            execute: object,
            output_root: Path,
            base_config: ReplayConfig,
        ) -> None:
            calls.append(
                (
                    "provider",
                    (
                        execute,
                        output_root,
                        base_config,
                    ),
                )
            )

        def __call__(
            self,
            model_name: str,
            window: EvaluationWindow,
        ):
            raise AssertionError(
                "provider should be consumed "
                "through evaluator/runner"
            )

    class FakeEvaluator:
        def __init__(
            self,
            *,
            replay_rows: object,
        ) -> None:
            calls.append(
                (
                    "evaluator",
                    replay_rows,
                )
            )

        def __call__(
            self,
            model_name: str,
            window: EvaluationWindow,
        ):
            raise AssertionError(
                "fake evaluation runner owns "
                "matrix creation"
            )

    class FakeEvaluationRunner:
        def __init__(
            self,
            *,
            evaluator: object,
        ) -> None:
            calls.append(
                (
                    "evaluation_runner",
                    evaluator,
                )
            )

        def run(
            self,
            *,
            model_names,
            windows,
        ):
            calls.append(
                (
                    "run",
                    (
                        tuple(model_names),
                        tuple(windows),
                    ),
                )
            )
            return matrix

    def fake_select(
        *,
        matrix: object,
        policy: ChampionPromotionPolicy | None = None,
    ):
        calls.append(
            (
                "select",
                (
                    matrix,
                    policy,
                ),
            )
        )
        return champion

    class FakeReportingService:
        def write(
            self,
            *,
            report: object,
            output_root: Path,
        ) -> Path:
            calls.append(
                (
                    "write",
                    (
                        report,
                        output_root,
                    ),
                )
            )
            return artifact_path

    policy = ChampionPromotionPolicy(
        minimum_composite_margin=0.0,
        minimum_significance_score=0.0,
    )

    service = (
        HistoricalModelEvaluationOrchestrationService(
            scenario_runner_factory=FakeScenarioRunner,
            execute_adapter_factory=FakeExecuteAdapter,
            provider_factory=FakeProvider,
            evaluator_factory=FakeEvaluator,
            evaluation_runner_factory=FakeEvaluationRunner,
            champion_selector=fake_select,
            reporting_service=FakeReportingService(),
        )
    )

    result = service.run(
        history=history,
        model_names=(
            "baseline",
            "calibration",
        ),
        windows=windows,
        replay_output_root=(
            tmp_path /
            "replay"
        ),
        report_output_root=(
            tmp_path /
            "report"
        ),
        base_config=_base_config(),
        policy=policy,
    )

    assert result.matrix is matrix
    assert result.champion is champion
    assert (
        result.artifact_path
        == artifact_path
    )

    assert [
        name
        for name, _ in calls
    ] == [
        "scenario_runner",
        "execute_adapter",
        "provider",
        "evaluator",
        "evaluation_runner",
        "run",
        "select",
        "write",
    ]

    run_call = calls[5][1]

    assert run_call[0] == (
        "baseline",
        "calibration",
    )
    assert run_call[1] == windows

    select_call = calls[6][1]

    assert select_call == (
        matrix,
        policy,
    )

    write_call = calls[7][1]

    assert write_call == (
        champion,
        tmp_path / "report",
    )


def test_service_uses_default_dependencies() -> None:
    from tools.validation.model_evaluation_orchestration_service import (
        HistoricalModelEvaluationOrchestrationService,
    )

    service = (
        HistoricalModelEvaluationOrchestrationService()
    )

    assert service is not None


def test_service_rejects_empty_history(
    tmp_path: Path,
) -> None:
    from tools.validation.model_evaluation_orchestration_service import (
        HistoricalModelEvaluationOrchestrationService,
    )

    service = (
        HistoricalModelEvaluationOrchestrationService()
    )

    with pytest.raises(
        ValueError,
        match="history must not be empty",
    ):
        service.run(
            history=(),
            model_names=("baseline",),
            windows=(
                _window(
                    "recent",
                    1229,
                    1230,
                ),
            ),
            replay_output_root=(
                tmp_path /
                "replay"
            ),
            report_output_root=(
                tmp_path /
                "report"
            ),
            base_config=_base_config(),
        )


def test_service_rejects_empty_model_names(
    tmp_path: Path,
) -> None:
    from tools.validation.model_evaluation_orchestration_service import (
        HistoricalModelEvaluationOrchestrationService,
    )

    service = (
        HistoricalModelEvaluationOrchestrationService()
    )

    with pytest.raises(
        ValueError,
        match="model_names must not be empty",
    ):
        service.run(
            history=(object(),),
            model_names=(),
            windows=(
                _window(
                    "recent",
                    1229,
                    1230,
                ),
            ),
            replay_output_root=(
                tmp_path /
                "replay"
            ),
            report_output_root=(
                tmp_path /
                "report"
            ),
            base_config=_base_config(),
        )


def test_service_rejects_empty_windows(
    tmp_path: Path,
) -> None:
    from tools.validation.model_evaluation_orchestration_service import (
        HistoricalModelEvaluationOrchestrationService,
    )

    service = (
        HistoricalModelEvaluationOrchestrationService()
    )

    with pytest.raises(
        ValueError,
        match="windows must not be empty",
    ):
        service.run(
            history=(object(),),
            model_names=("baseline",),
            windows=(),
            replay_output_root=(
                tmp_path /
                "replay"
            ),
            report_output_root=(
                tmp_path /
                "report"
            ),
            base_config=_base_config(),
        )
