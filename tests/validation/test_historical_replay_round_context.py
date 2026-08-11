from __future__ import annotations

from pathlib import Path

from lrp.evolution.contracts.learning_context import (
    LearningContext,
)
from tools.validation.historical_replay_executor import (
    HistoricalReplayExecutor,
    ReplayState,
)
from tools.validation.historical_replay_models import (
    ReplayConfig,
)


def make_executor(
    tmp_path: Path,
) -> HistoricalReplayExecutor:
    return HistoricalReplayExecutor(
        history=(object(),),
        config=ReplayConfig(
            start_round=1132,
            end_round=1231,
            seed_base=20260802,
            candidate_count=1000,
            top_k=20,
            practical_k=5,
            mode="fast",
        ),
        learning_root=tmp_path / "learning",
        profile_root=tmp_path / "profiles",
    )


def test_carried_learning_context_advances_round(
    tmp_path: Path,
) -> None:
    executor = make_executor(tmp_path)

    previous = LearningContext(
        cycle_id=(
            "historical-replay-1132-1231"
        ),
        round_no=1132,
        metadata={
            "review_count": 1,
        },
    )

    current = executor._learning_context(
        state=ReplayState(
            learning_context=previous
        ),
        round_no=1133,
    )

    assert current.round_no == 1133
    assert current.cycle_id == previous.cycle_id
    assert current.metadata == previous.metadata
    assert current is not previous


def test_initial_learning_context_uses_current_round(
    tmp_path: Path,
) -> None:
    executor = make_executor(tmp_path)

    context = executor._learning_context(
        state=None,
        round_no=1132,
    )

    assert context.round_no == 1132
    assert context.cycle_id == (
        "historical-replay-1132-1231"
    )


def test_executor_can_enable_regime_calibration_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "regime-calibration"

    executor = HistoricalReplayExecutor(
        history=(object(),),
        config=ReplayConfig(
            start_round=1132,
            end_round=1231,
            seed_base=20260802,
            candidate_count=1000,
            top_k=20,
            practical_k=5,
            mode="fast",
        ),
        learning_root=tmp_path / "learning",
        profile_root=tmp_path / "profiles",
        regime_calibration_root=root,
    )

    assert executor.regime_calibration_root == root


def test_learning_service_uses_regime_calibration_repository(
    tmp_path: Path,
) -> None:
    from lrp.regimes.calibration_repository import (
        RegimeCalibrationRepository,
    )
    from lrp.regimes.calibration_updater import (
        RegimeCalibrationUpdater,
    )
    from lrp.regimes.reward_calculator import (
        RegimeRewardCalculator,
    )

    root = tmp_path / "regime-calibration"

    executor = HistoricalReplayExecutor(
        history=(object(),),
        config=ReplayConfig(
            start_round=1132,
            end_round=1231,
            seed_base=20260802,
            candidate_count=1000,
            top_k=20,
            practical_k=5,
            mode="fast",
        ),
        learning_root=tmp_path / "learning",
        profile_root=tmp_path / "profiles",
        regime_calibration_root=root,
    )

    service = executor._build_learning_service()

    assert isinstance(
        service.regime_reward_calculator,
        RegimeRewardCalculator,
    )
    assert isinstance(
        service.regime_calibration_updater,
        RegimeCalibrationUpdater,
    )
    assert isinstance(
        service.regime_calibration_repository,
        RegimeCalibrationRepository,
    )
    assert (
        service.regime_calibration_repository.root
        == root
    )


def test_learning_service_keeps_regime_learning_disabled_by_default(
    tmp_path: Path,
) -> None:
    executor = make_executor(tmp_path)

    service = executor._build_learning_service()

    assert service.regime_reward_calculator is None
    assert service.regime_calibration_updater is None
    assert service.regime_calibration_repository is None

def test_replay_regime_updater_uses_adaptive_learning_rate(
    tmp_path: Path,
) -> None:
    from lrp.regimes.learning_rate import (
        AdaptiveLearningRatePolicy,
    )

    root = tmp_path / "regime-calibration"

    executor = HistoricalReplayExecutor(
        history=(object(),),
        config=ReplayConfig(
            start_round=1132,
            end_round=1231,
            seed_base=20260802,
            candidate_count=1000,
            top_k=20,
            practical_k=5,
            mode="fast",
        ),
        learning_root=tmp_path / "learning",
        profile_root=tmp_path / "profiles",
        regime_calibration_root=root,
    )

    service = executor._build_learning_service()

    updater = service.regime_calibration_updater

    assert updater is not None
    assert isinstance(
        updater.learning_rate_policy,
        AdaptiveLearningRatePolicy,
    )

def test_executor_can_enable_regime_bayesian_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "regime-bayesian"

    executor = HistoricalReplayExecutor(
        history=(object(),),
        config=ReplayConfig(
            start_round=1132,
            end_round=1231,
            seed_base=20260802,
            candidate_count=1000,
            top_k=20,
            practical_k=5,
            mode="fast",
        ),
        learning_root=tmp_path / "learning",
        profile_root=tmp_path / "profiles",
        regime_bayesian_root=root,
    )

    assert executor.regime_bayesian_root == root


def test_learning_service_uses_regime_bayesian_repository(
    tmp_path: Path,
) -> None:
    from lrp.regimes.bayesian_repository import (
        RegimeBayesianRepository,
    )
    from lrp.regimes.bayesian_updater import (
        RegimeBayesianUpdater,
    )
    from lrp.regimes.reward_calculator import (
        RegimeRewardCalculator,
    )

    root = tmp_path / "regime-bayesian"

    executor = HistoricalReplayExecutor(
        history=(object(),),
        config=ReplayConfig(
            start_round=1132,
            end_round=1231,
            seed_base=20260802,
            candidate_count=1000,
            top_k=20,
            practical_k=5,
            mode="fast",
        ),
        learning_root=tmp_path / "learning",
        profile_root=tmp_path / "profiles",
        regime_bayesian_root=root,
    )

    service = executor._build_learning_service()

    assert isinstance(
        service.regime_reward_calculator,
        RegimeRewardCalculator,
    )
    assert isinstance(
        service.regime_bayesian_updater,
        RegimeBayesianUpdater,
    )
    assert isinstance(
        service.regime_bayesian_repository,
        RegimeBayesianRepository,
    )
    assert (
        service.regime_bayesian_repository.root
        == root
    )
