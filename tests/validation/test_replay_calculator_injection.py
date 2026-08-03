from __future__ import annotations

from pathlib import Path

import pytest

from lrp.evolution.algorithms.adaptive import (
    AdaptiveWeightCalculator,
)
from tools.validation.historical_replay_executor import (
    HistoricalReplayExecutor,
)
from tools.validation.historical_replay_models import (
    ReplayConfig,
)


def make_config() -> ReplayConfig:
    return ReplayConfig(
        start_round=1132,
        end_round=1231,
        seed_base=20260802,
        candidate_count=1000,
        top_k=20,
        practical_k=5,
        mode="fast",
    )


def test_executor_accepts_adaptive_calculator(
    tmp_path: Path,
) -> None:
    calculator = AdaptiveWeightCalculator(
        adjustment_scale=0.0625,
        minimum_weight=0.03,
    )

    executor = HistoricalReplayExecutor(
        history=(object(),),
        config=make_config(),
        learning_root=tmp_path / "learning",
        profile_root=tmp_path / "profiles",
        adaptive_calculator=calculator,
    )

    assert executor.adaptive_calculator is calculator


def test_executor_defaults_to_pipeline_default(
    tmp_path: Path,
) -> None:
    executor = HistoricalReplayExecutor(
        history=(object(),),
        config=make_config(),
        learning_root=tmp_path / "learning",
        profile_root=tmp_path / "profiles",
    )

    assert executor.adaptive_calculator is None


def test_executor_rejects_invalid_calculator(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        TypeError,
        match="adaptive_calculator",
    ):
        HistoricalReplayExecutor(
            history=(object(),),
            config=make_config(),
            learning_root=tmp_path / "learning",
            profile_root=tmp_path / "profiles",
            adaptive_calculator=object(),  # type: ignore[arg-type]
        )


def test_profile_service_builds_with_custom_calculator(
    tmp_path: Path,
) -> None:
    calculator = AdaptiveWeightCalculator(
        adjustment_scale=0.10,
        minimum_weight=0.03,
    )

    executor = HistoricalReplayExecutor(
        history=(object(),),
        config=make_config(),
        learning_root=tmp_path / "learning",
        profile_root=tmp_path / "profiles",
        adaptive_calculator=calculator,
    )

    service = executor._build_profile_service()

    assert service.coordinator is not None
