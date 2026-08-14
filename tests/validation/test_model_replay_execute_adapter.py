from __future__ import annotations

from pathlib import Path

import pytest

from tools.validation.historical_replay_models import (
    ReplayConfig,
)

from tools.validation.regime_learning_comparison_runner import (
    RegimeLearningScenario,
)

from tools.validation.model_replay_execute_adapter import (
    HistoricalModelReplayExecuteAdapter,
)


def test_adapter_resolves_baseline_scenario(
    tmp_path: Path,
) -> None:
    seen = []

    def run_scenario(
        *,
        scenario: RegimeLearningScenario,
        config: ReplayConfig,
        output_root: Path,
    ) -> object:
        seen.append(
            (
                scenario,
                config,
                output_root,
            )
        )

        return object()

    adapter = HistoricalModelReplayExecuteAdapter(
        run_scenario=run_scenario,
    )

    config = ReplayConfig(
        start_round=1220,
        end_round=1222,
    )

    result = adapter(
        model_name="baseline",
        config=config,
        output_root=tmp_path,
    )

    assert result is not None
    assert len(seen) == 1

    scenario, actual_config, actual_root = seen[0]

    assert scenario.name == "baseline"
    assert scenario.calibration_enabled is False
    assert scenario.bayesian_enabled is False

    assert actual_config is config
    assert actual_root == tmp_path


@pytest.mark.parametrize(
    (
        "model_name",
        "calibration_enabled",
        "bayesian_enabled",
    ),
    (
        (
            "baseline",
            False,
            False,
        ),
        (
            "calibration",
            True,
            False,
        ),
        (
            "bayesian",
            False,
            True,
        ),
        (
            "combined",
            True,
            True,
        ),
    ),
)
def test_adapter_resolves_supported_models(
    tmp_path: Path,
    model_name: str,
    calibration_enabled: bool,
    bayesian_enabled: bool,
) -> None:
    seen = []

    def run_scenario(
        *,
        scenario: RegimeLearningScenario,
        config: ReplayConfig,
        output_root: Path,
    ) -> object:
        seen.append(scenario)
        return object()

    adapter = HistoricalModelReplayExecuteAdapter(
        run_scenario=run_scenario,
    )

    adapter(
        model_name=model_name,
        config=ReplayConfig(
            start_round=1220,
            end_round=1220,
        ),
        output_root=tmp_path,
    )

    scenario = seen[0]

    assert scenario.name == model_name
    assert (
        scenario.calibration_enabled
        is calibration_enabled
    )
    assert (
        scenario.bayesian_enabled
        is bayesian_enabled
    )


def test_adapter_rejects_blank_model_name(
    tmp_path: Path,
) -> None:
    adapter = HistoricalModelReplayExecuteAdapter(
        run_scenario=lambda **kwargs: object(),
    )

    with pytest.raises(
        ValueError,
        match="model_name",
    ):
        adapter(
            model_name=" ",
            config=ReplayConfig(
                start_round=1220,
                end_round=1220,
            ),
            output_root=tmp_path,
        )


def test_adapter_rejects_unknown_model_name(
    tmp_path: Path,
) -> None:
    adapter = HistoricalModelReplayExecuteAdapter(
        run_scenario=lambda **kwargs: object(),
    )

    with pytest.raises(
        ValueError,
        match="unsupported model_name",
    ):
        adapter(
            model_name="unknown",
            config=ReplayConfig(
                start_round=1220,
                end_round=1220,
            ),
            output_root=tmp_path,
        )


def test_adapter_requires_replay_config(
    tmp_path: Path,
) -> None:
    adapter = HistoricalModelReplayExecuteAdapter(
        run_scenario=lambda **kwargs: object(),
    )

    with pytest.raises(
        TypeError,
        match="ReplayConfig",
    ):
        adapter(
            model_name="baseline",
            config=object(),
            output_root=tmp_path,
        )


def test_adapter_requires_callable_runner() -> None:
    with pytest.raises(
        TypeError,
        match="run_scenario",
    ):
        HistoricalModelReplayExecuteAdapter(
            run_scenario=object(),
        )
