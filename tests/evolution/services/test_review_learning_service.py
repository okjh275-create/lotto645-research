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

    assert result.snapshot.metadata == {
        "round": 1220,
        "learning_source": (
            "prediction_review"
        ),
        "feedback_count": 2,
        "policy": "thompson",
    }


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
