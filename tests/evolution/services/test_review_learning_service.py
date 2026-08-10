from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from lrp.evolution.contracts.learning_context import (
    LearningContext,
)
from lrp.evolution.contracts.review_learning import (
    ReviewLearningResult,
)
from lrp.evolution.repositories.file_snapshot_repository import (
    FileSnapshotRepository,
)
from lrp.evolution.services.persistent_learning_runner import (
    PersistentLearningRunner,
)
from lrp.evolution.services.persistent_learning_service import (
    PersistentLearningService,
)
from lrp.evolution.services.review_learning_service import (
    ReviewLearningService,
)
from lrp.evolution.services.snapshot_factory import (
    SnapshotFactory,
)


FIXED_TIME = datetime(
    2026,
    8,
    2,
    0,
    0,
    tzinfo=timezone.utc,
)


def make_context() -> LearningContext:
    return LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
        version=1,
    )


def make_review() -> dict[str, object]:
    return {
        "summary": {
            "set_count": 10,
            "best_main_hits": 4,
            "practical_best_hits": 3,
        }
    }


def make_service(
    tmp_path: Path,
) -> ReviewLearningService:
    persistence = PersistentLearningService(
        FileSnapshotRepository(tmp_path),
        snapshot_factory=SnapshotFactory(
            clock=lambda: FIXED_TIME
        ),
    )
    runner = PersistentLearningRunner(
        persistence
    )

    return ReviewLearningService(runner)


def test_learn_persists_review_learning(
    tmp_path: Path,
) -> None:
    service = make_service(tmp_path)

    result = service.learn(
        context=make_context(),
        review_payload=make_review(),
        snapshot_id="review-1220",
        policy="thompson",
        metadata={
            "round": 1220,
        },
    )

    assert isinstance(
        result,
        ReviewLearningResult,
    )
    assert result.feedback_count == 2
    assert result.policy == "thompson"
    assert result.step_count == 2
    assert result.final_context.version == 3
    assert (
        tmp_path / "review-1220.json"
    ).is_file()


def test_learn_stores_expected_rewards(
    tmp_path: Path,
) -> None:
    result = make_service(tmp_path).learn(
        context=make_context(),
        review_payload=make_review(),
        snapshot_id="review-1220",
        policy="thompson",
    )

    rewards = result.final_context.rewards

    assert rewards[
        "prediction_review:"
        "thompson:"
        "portfolio_top_k"
    ] == pytest.approx(0.55)

    assert rewards[
        "prediction_review:"
        "thompson:"
        "practical_top5"
    ] == pytest.approx(0.20)


def test_learn_sets_last_policy_and_arm(
    tmp_path: Path,
) -> None:
    result = make_service(tmp_path).learn(
        context=make_context(),
        review_payload=make_review(),
        snapshot_id="review-1220",
        policy="thompson",
    )

    assert (
        result.final_context.selected_policy
        == "thompson"
    )
    assert (
        result.final_context.selected_arm
        == "practical_top5"
    )


def test_snapshot_metadata_is_enriched(
    tmp_path: Path,
) -> None:
    result = make_service(tmp_path).learn(
        context=make_context(),
        review_payload=make_review(),
        snapshot_id="review-1220",
        policy="thompson",
        metadata={
            "round": 1220,
        },
    )

    metadata = result.snapshot.metadata

    assert metadata["round"] == 1220
    assert metadata["learning_source"] == (
        "prediction_review"
    )
    assert metadata["feedback_count"] == 2
    assert metadata["review_set_count"] == 10
    assert metadata["policy"] == "thompson"

    assert metadata[
        "reward_vector_portfolio_hit"
    ] == 0.55
    assert metadata[
        "reward_vector_practical_hit"
    ] == 0.20
    assert metadata[
        "reward_vector_sample_size"
    ] == 10
    assert metadata[
        "reward_vector_source"
    ] == "prediction_review"
    assert metadata[
        "reward_vector_policy"
    ] == "thompson"


def test_original_metadata_is_not_mutated(
    tmp_path: Path,
) -> None:
    metadata = {
        "round": 1220,
    }

    make_service(tmp_path).learn(
        context=make_context(),
        review_payload=make_review(),
        snapshot_id="review-1220",
        metadata=metadata,
    )

    assert metadata == {
        "round": 1220,
    }


def test_duplicate_snapshot_is_rejected(
    tmp_path: Path,
) -> None:
    service = make_service(tmp_path)

    service.learn(
        context=make_context(),
        review_payload=make_review(),
        snapshot_id="review-1220",
    )

    with pytest.raises(FileExistsError):
        service.learn(
            context=make_context(),
            review_payload=make_review(),
            snapshot_id="review-1220",
        )


def test_overwrite_is_supported(
    tmp_path: Path,
) -> None:
    service = make_service(tmp_path)

    service.learn(
        context=make_context(),
        review_payload=make_review(),
        snapshot_id="review-1220",
    )

    result = service.learn(
        context=make_context(),
        review_payload={
            "summary": {
                "set_count": 10,
                "best_main_hits": 5,
                "practical_best_hits": 4,
            }
        },
        snapshot_id="review-1220",
        overwrite=True,
    )

    assert result.final_context.rewards[
        "prediction_review:portfolio_top_k"
    ] == pytest.approx(0.85)


def test_invalid_runner_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="PersistentLearningRunner",
    ):
        ReviewLearningService(
            object(),  # type: ignore[arg-type]
        )


def test_invalid_context_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        TypeError,
        match="LearningContext",
    ):
        make_service(tmp_path).learn(
            context=object(),  # type: ignore[arg-type]
            review_payload=make_review(),
            snapshot_id="review-1220",
        )


def test_invalid_review_payload_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        TypeError,
        match="must be a mapping",
    ):
        make_service(tmp_path).learn(
            context=make_context(),
            review_payload=object(),  # type: ignore[arg-type]
            snapshot_id="review-1220",
        )


def test_review_set_count_reaches_final_context(
    tmp_path: Path,
) -> None:
    result = make_service(tmp_path).learn(
        context=make_context(),
        review_payload=make_review(),
        snapshot_id="review-1220",
    )

    assert result.final_context.metadata[
        "review_set_count"
    ] == 10


def test_original_context_metadata_is_not_mutated(
    tmp_path: Path,
) -> None:
    context = LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
        metadata={
            "seed": 20260802,
        },
    )

    result = make_service(tmp_path).learn(
        context=context,
        review_payload=make_review(),
        snapshot_id="review-1220",
    )

    assert context.metadata == {
        "seed": 20260802,
    }
    assert result.final_context.metadata[
        "seed"
    ] == 20260802
    assert result.final_context.metadata[
        "review_set_count"
    ] == 10


def test_learning_accumulates_review_sample_metadata(
    tmp_path,
) -> None:
    service = make_service(tmp_path)

    first_review = make_review()
    first_review["summary"]["set_count"] = 20

    first = service.learn(
        context=make_context(),
        review_payload=first_review,
        snapshot_id="review-1222",
        policy="thompson",
    )

    second_review = make_review()
    second_review["summary"]["set_count"] = 20

    second = service.learn(
        context=first.final_context,
        review_payload=second_review,
        snapshot_id="review-1223",
        policy="thompson",
    )

    assert first.final_context.metadata[
        "review_set_count"
    ] == 20
    assert first.final_context.metadata[
        "cumulative_review_set_count"
    ] == 20
    assert first.final_context.metadata[
        "review_count"
    ] == 1

    assert second.final_context.metadata[
        "review_set_count"
    ] == 20
    assert second.final_context.metadata[
        "cumulative_review_set_count"
    ] == 40
    assert second.final_context.metadata[
        "review_count"
    ] == 2



def test_learning_snapshot_metadata_contains_reward_vector(
    tmp_path,
) -> None:
    service = make_service(tmp_path)

    result = service.learn(
        context=make_context(),
        review_payload=make_review(),
        snapshot_id="review-1222",
        policy="thompson",
    )

    metadata = result.snapshot.metadata

    assert metadata[
        "reward_vector_portfolio_hit"
    ] == 0.55
    assert metadata[
        "reward_vector_practical_hit"
    ] == 0.20
    assert metadata[
        "reward_vector_sample_size"
    ] == result.final_context.metadata[
        "review_set_count"
    ]
    assert metadata[
        "reward_vector_policy"
    ] == "thompson"


def test_learning_context_contains_flattened_reward_vector(
    tmp_path,
) -> None:
    result = make_service(tmp_path).learn(
        context=make_context(),
        review_payload=make_review(),
        snapshot_id="review-1222",
        policy="thompson",
    )

    metadata = result.final_context.metadata

    assert metadata[
        "reward_vector_portfolio_hit"
    ] == 0.55
    assert metadata[
        "reward_vector_practical_hit"
    ] == 0.20
    assert metadata[
        "reward_vector_sample_size"
    ] == metadata["review_set_count"]
    assert metadata[
        "reward_vector_source"
    ] == "prediction_review"
    assert metadata[
        "reward_vector_policy"
    ] == "thompson"


def test_reward_vector_metadata_is_replaced_by_latest_review(
    tmp_path,
) -> None:
    service = make_service(tmp_path)

    first = service.learn(
        context=make_context(),
        review_payload=make_review(),
        snapshot_id="review-1222",
        policy="thompson",
    )

    second_review = make_review()
    second_review["summary"][
        "best_main_hits"
    ] = 5
    second_review["summary"][
        "practical_best_hits"
    ] = 4

    second = service.learn(
        context=first.final_context,
        review_payload=second_review,
        snapshot_id="review-1223",
        policy="thompson",
    )

    metadata = second.final_context.metadata

    assert metadata[
        "reward_vector_portfolio_hit"
    ] == 0.85
    assert metadata[
        "reward_vector_practical_hit"
    ] == 0.55
    assert metadata[
        "review_count"
    ] == 2


def make_prediction_payload() -> dict[str, object]:
    probabilities = []

    for number in range(1, 46):
        probabilities.append(
            {
                "number": number,
                "components": {
                    "hot": number / 45.0,
                    "cold": 0.5,
                    "gap": 0.5,
                    "trend": 0.5,
                    "transition": 0.5,
                },
            }
        )

    return {
        "probability_vector": {
            "probabilities": probabilities,
        }
    }


def test_learning_stores_feature_attribution_signals(
    tmp_path,
) -> None:
    result = make_service(tmp_path).learn(
        context=make_context(),
        review_payload=make_review(),
        prediction_payload=make_prediction_payload(),
        winning_numbers=(
            40,
            41,
            42,
            43,
            44,
            45,
        ),
        snapshot_id="review-1222",
        policy="thompson",
    )

    metadata = result.final_context.metadata

    assert metadata["feature_signal_hot"] > 0.0
    assert metadata[
        "feature_signal_cold"
    ] == pytest.approx(0.0)
    assert metadata[
        "feature_signal_gap"
    ] == pytest.approx(0.0)
    assert metadata[
        "feature_signal_trend"
    ] == pytest.approx(0.0)
    assert metadata[
        "feature_signal_transition"
    ] == pytest.approx(0.0)

    snapshot_metadata = result.snapshot.metadata

    assert snapshot_metadata[
        "feature_signal_hot"
    ] == pytest.approx(
        metadata["feature_signal_hot"]
    )


def test_learning_without_attribution_inputs_is_supported(
    tmp_path,
) -> None:
    result = make_service(tmp_path).learn(
        context=make_context(),
        review_payload=make_review(),
        snapshot_id="review-1222",
        policy="thompson",
    )

    assert not any(
        key.startswith("feature_signal_")
        for key in result.final_context.metadata
    )


def test_partial_attribution_inputs_are_rejected(
    tmp_path,
) -> None:
    with pytest.raises(
        ValueError,
        match="winning_numbers is required",
    ):
        make_service(tmp_path).learn(
            context=make_context(),
            review_payload=make_review(),
            prediction_payload=make_prediction_payload(),
            snapshot_id="review-1222",
        )

def test_global_regime_from_prediction_reaches_snapshot(
    tmp_path: Path,
) -> None:
    global_regime = {
        "primary": "gap_recovery",
        "confidence": 0.78,
        "secondary": "cluster_rotation",
        "secondary_confidence": 0.54,
        "scores": {
            "gap_recovery": 0.78,
            "cluster_rotation": 0.54,
        },
        "features": {
            "average_recency": 0.41,
        },
        "mode": "shadow",
    }

    review = make_review()
    review["prediction_metadata"] = {
        "global_regime": global_regime,
    }

    result = make_service(tmp_path).learn(
        context=make_context(),
        review_payload=review,
        snapshot_id="review-1220",
    )

    assert (
        result.snapshot.metadata[
            "global_regime_primary"
        ]
        == "gap_recovery"
    )
    assert (
        result.snapshot.metadata[
            "global_regime_confidence"
        ]
        == 0.78
    )
    assert (
        result.snapshot.metadata[
            "global_regime_secondary"
        ]
        == "cluster_rotation"
    )
    assert (
        result.snapshot.metadata[
            "global_regime_secondary_confidence"
        ]
        == 0.54
    )
    assert (
        result.snapshot.metadata[
            "global_regime_mode"
        ]
        == "shadow"
    )


def test_global_regime_from_review_reaches_final_context(
    tmp_path: Path,
) -> None:
    global_regime = {
        "primary": "high_band_expansion",
        "confidence": 0.71,
        "secondary": None,
        "secondary_confidence": None,
        "scores": {
            "high_band_expansion": 0.71,
        },
        "features": {
            "high_band_ratio": 0.67,
        },
        "mode": "shadow",
    }

    review = make_review()
    review["prediction_metadata"] = {
        "global_regime": global_regime,
    }

    result = make_service(tmp_path).learn(
        context=make_context(),
        review_payload=review,
        snapshot_id="review-1220",
    )

    assert (
        result.snapshot.metadata[
            "global_regime_primary"
        ]
        == "high_band_expansion"
    )
    assert (
        result.snapshot.metadata[
            "global_regime_confidence"
        ]
        == 0.71
    )
    assert (
        result.snapshot.metadata[
            "global_regime_mode"
        ]
        == "shadow"
    )
    assert (
        result.final_context.metadata[
            "global_regime_primary"
        ]
        == "high_band_expansion"
    )
    assert (
        result.final_context.metadata[
            "global_regime_confidence"
        ]
        == 0.71
    )
    assert (
        result.final_context.metadata[
            "global_regime_mode"
        ]
        == "shadow"
    )


def test_review_learning_service_accepts_regime_dependencies(
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

    calculator = RegimeRewardCalculator()
    updater = RegimeCalibrationUpdater()
    repository = RegimeCalibrationRepository(
        tmp_path / "regime-calibration"
    )

    persistence = PersistentLearningService(
        FileSnapshotRepository(
            tmp_path / "learning"
        ),
        snapshot_factory=SnapshotFactory(
            clock=lambda: FIXED_TIME
        ),
    )
    runner = PersistentLearningRunner(
        persistence
    )

    service = ReviewLearningService(
        runner,
        regime_reward_calculator=calculator,
        regime_calibration_updater=updater,
        regime_calibration_repository=repository,
    )

    assert (
        service.regime_reward_calculator
        is calculator
    )
    assert (
        service.regime_calibration_updater
        is updater
    )
    assert (
        service.regime_calibration_repository
        is repository
    )


def test_review_learning_service_defaults_regime_dependencies_to_none(
    tmp_path: Path,
) -> None:
    service = make_service(tmp_path)

    assert service.regime_reward_calculator is None
    assert service.regime_calibration_updater is None
    assert service.regime_calibration_repository is None


@pytest.mark.parametrize(
    "field_name",
    [
        "regime_reward_calculator",
        "regime_calibration_updater",
        "regime_calibration_repository",
    ],
)
def test_review_learning_service_rejects_invalid_regime_dependency(
    tmp_path: Path,
    field_name: str,
) -> None:
    persistence = PersistentLearningService(
        FileSnapshotRepository(tmp_path),
        snapshot_factory=SnapshotFactory(
            clock=lambda: FIXED_TIME
        ),
    )
    runner = PersistentLearningRunner(
        persistence
    )

    with pytest.raises(
        TypeError,
        match=field_name,
    ):
        ReviewLearningService(
            runner,
            **{
                field_name: object(),
            },
        )

def test_apply_regime_learning_creates_first_snapshot(
    tmp_path: Path,
) -> None:
    from lrp.evolution.contracts.review_reward_vector import (
        ReviewRewardVector,
    )
    from lrp.regimes.calibration_repository import (
        RegimeCalibrationRepository,
    )
    from lrp.regimes.calibration_updater import (
        RegimeCalibrationUpdater,
    )
    from lrp.regimes.reward_calculator import (
        RegimeRewardCalculator,
    )

    repository = RegimeCalibrationRepository(
        tmp_path / "regime-calibration"
    )

    persistence = PersistentLearningService(
        FileSnapshotRepository(
            tmp_path / "learning"
        ),
        snapshot_factory=SnapshotFactory(
            clock=lambda: FIXED_TIME
        ),
    )
    runner = PersistentLearningRunner(
        persistence
    )

    service = ReviewLearningService(
        runner,
        regime_reward_calculator=(
            RegimeRewardCalculator()
        ),
        regime_calibration_updater=(
            RegimeCalibrationUpdater()
        ),
        regime_calibration_repository=repository,
    )

    reward_vector = ReviewRewardVector(
        portfolio_hit=0.5,
        practical_hit=0.2,
        rank_quality=0.1,
        coverage=0.0,
        diversity=0.0,
        stability=0.0,
        sample_size=10,
        metadata={},
    )

    result = service._apply_regime_learning(
        reward_vector=reward_vector,
        global_regime={
            "primary": "gap_recovery",
            "confidence": 1.0,
        },
        review_set_count=10,
    )

    assert result is not None
    assert result["applied"] is True
    assert result["revision"] == 1
    assert result["sample_size"] == 10
    assert result["regime"] == "gap_recovery"

    snapshot = repository.load_latest()

    assert snapshot.revision == 1
    assert snapshot.sample_size == 10
    assert (
        snapshot.calibration.gap_recovery
        > 1.0
    )
    assert (
        snapshot.calibration.cluster_rotation
        == 1.0
    )
    assert (
        snapshot.calibration.high_band_expansion
        == 1.0
    )
    assert (
        snapshot.calibration.low_band_expansion
        == 1.0
    )


def test_apply_regime_learning_increments_revision_and_sample_size(
    tmp_path: Path,
) -> None:
    from lrp.evolution.contracts.review_reward_vector import (
        ReviewRewardVector,
    )
    from lrp.regimes.calibration_repository import (
        RegimeCalibrationRepository,
    )
    from lrp.regimes.calibration_updater import (
        RegimeCalibrationUpdater,
    )
    from lrp.regimes.reward_calculator import (
        RegimeRewardCalculator,
    )

    repository = RegimeCalibrationRepository(
        tmp_path / "regime-calibration"
    )

    persistence = PersistentLearningService(
        FileSnapshotRepository(
            tmp_path / "learning"
        ),
        snapshot_factory=SnapshotFactory(
            clock=lambda: FIXED_TIME
        ),
    )
    runner = PersistentLearningRunner(
        persistence
    )

    service = ReviewLearningService(
        runner,
        regime_reward_calculator=(
            RegimeRewardCalculator()
        ),
        regime_calibration_updater=(
            RegimeCalibrationUpdater()
        ),
        regime_calibration_repository=repository,
    )

    reward_vector = ReviewRewardVector(
        portfolio_hit=0.5,
        practical_hit=0.2,
        rank_quality=0.1,
        coverage=0.0,
        diversity=0.0,
        stability=0.0,
        sample_size=10,
        metadata={},
    )

    service._apply_regime_learning(
        reward_vector=reward_vector,
        global_regime={
            "primary": "gap_recovery",
            "confidence": 1.0,
        },
        review_set_count=10,
    )

    result = service._apply_regime_learning(
        reward_vector=reward_vector,
        global_regime={
            "primary": "gap_recovery",
            "confidence": 1.0,
        },
        review_set_count=5,
    )

    assert result is not None
    assert result["revision"] == 2
    assert result["sample_size"] == 15

    assert repository.revisions() == (
        1,
        2,
    )


def test_apply_regime_learning_is_disabled_without_dependencies(
    tmp_path: Path,
) -> None:
    from lrp.evolution.contracts.review_reward_vector import (
        ReviewRewardVector,
    )

    service = make_service(tmp_path)

    reward_vector = ReviewRewardVector(
        portfolio_hit=0.0,
        practical_hit=0.0,
        rank_quality=0.0,
        coverage=0.0,
        diversity=0.0,
        stability=0.0,
        sample_size=10,
        metadata={},
    )

    result = service._apply_regime_learning(
        reward_vector=reward_vector,
        global_regime={
            "primary": "gap_recovery",
            "confidence": 1.0,
        },
        review_set_count=10,
    )

    assert result is None

def test_learn_persists_regime_calibration_end_to_end(
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

    regime_repository = RegimeCalibrationRepository(
        tmp_path / "regime-calibration"
    )

    persistence = PersistentLearningService(
        FileSnapshotRepository(
            tmp_path / "learning"
        ),
        snapshot_factory=SnapshotFactory(
            clock=lambda: FIXED_TIME
        ),
    )
    runner = PersistentLearningRunner(
        persistence
    )

    service = ReviewLearningService(
        runner,
        regime_reward_calculator=(
            RegimeRewardCalculator()
        ),
        regime_calibration_updater=(
            RegimeCalibrationUpdater()
        ),
        regime_calibration_repository=(
            regime_repository
        ),
    )

    review = make_review()
    review["prediction_metadata"] = {
        "global_regime": {
            "primary": "gap_recovery",
            "confidence": 1.0,
            "secondary": None,
            "secondary_confidence": None,
            "scores": {
                "gap_recovery": 1.0,
            },
            "features": {
                "average_gap_reversion": 0.8,
            },
            "mode": "active",
        }
    }

    result = service.learn(
        context=make_context(),
        review_payload=review,
        snapshot_id="review-1220",
        policy="thompson",
    )

    assert result.snapshot_id == "review-1220"

    regime_snapshot = (
        regime_repository.load_latest()
    )

    assert regime_snapshot.revision == 1
    assert regime_snapshot.sample_size == 10

    assert (
        regime_snapshot
        .calibration
        .gap_recovery
        > 1.0
    )

    assert (
        regime_snapshot
        .calibration
        .cluster_rotation
        == 1.0
    )
    assert (
        regime_snapshot
        .calibration
        .high_band_expansion
        == 1.0
    )
    assert (
        regime_snapshot
        .calibration
        .low_band_expansion
        == 1.0
    )