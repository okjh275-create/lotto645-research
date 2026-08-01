from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from lrp.evolution.contracts.learning_context import (
    LearningContext,
)
from lrp.evolution.contracts.learning_cycle import (
    LearningCycleResult,
)
from lrp.evolution.contracts.persistent_learning import (
    PersistentLearningRunResult,
)
from lrp.evolution.contracts.review_learning import (
    ReviewLearningResult,
)
from lrp.evolution.contracts.snapshot_schema import (
    LearningCycleSnapshot,
)


def make_run_result() -> PersistentLearningRunResult:
    context = LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
    )
    learning_result = LearningCycleResult(
        initial_context=context,
        final_context=context,
        steps=(),
    )
    snapshot = LearningCycleSnapshot(
        snapshot_id="review-1220",
        result=learning_result,
        created_at_utc=datetime(
            2026,
            8,
            2,
            tzinfo=timezone.utc,
        ),
    )

    return PersistentLearningRunResult(
        learning_result=learning_result,
        snapshot=snapshot,
    )


def test_result_creation() -> None:
    run_result = make_run_result()

    result = ReviewLearningResult(
        run_result=run_result,
        feedback_count=2,
        policy=" thompson ",
    )

    assert result.run_result is run_result
    assert result.feedback_count == 2
    assert result.policy == "thompson"
    assert result.snapshot_id == "review-1220"


def test_result_delegates_properties() -> None:
    result = ReviewLearningResult(
        run_result=make_run_result(),
        feedback_count=2,
        policy=None,
    )

    assert (
        result.learning_result
        is result.run_result.learning_result
    )
    assert (
        result.snapshot
        is result.run_result.snapshot
    )
    assert (
        result.final_context
        is result.run_result.final_context
    )
    assert result.step_count == 0


def test_result_is_frozen() -> None:
    result = ReviewLearningResult(
        run_result=make_run_result(),
        feedback_count=2,
        policy=None,
    )

    with pytest.raises(FrozenInstanceError):
        result.feedback_count = 3  # type: ignore[misc]


def test_invalid_run_result_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="PersistentLearningRunResult",
    ):
        ReviewLearningResult(
            run_result=object(),  # type: ignore[arg-type]
            feedback_count=2,
            policy=None,
        )


@pytest.mark.parametrize(
    "feedback_count",
    [0, -1],
)
def test_invalid_feedback_count_is_rejected(
    feedback_count: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="greater than or equal to 1",
    ):
        ReviewLearningResult(
            run_result=make_run_result(),
            feedback_count=feedback_count,
            policy=None,
        )


def test_invalid_feedback_count_type_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="must be an integer",
    ):
        ReviewLearningResult(
            run_result=make_run_result(),
            feedback_count=True,  # type: ignore[arg-type]
            policy=None,
        )


def test_empty_policy_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        ReviewLearningResult(
            run_result=make_run_result(),
            feedback_count=2,
            policy=" ",
        )
