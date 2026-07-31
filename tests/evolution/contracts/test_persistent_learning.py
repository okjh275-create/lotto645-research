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
from lrp.evolution.contracts.snapshot_schema import (
    LearningCycleSnapshot,
)


def make_learning_result() -> LearningCycleResult:
    context = LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
    )

    return LearningCycleResult(
        initial_context=context,
        final_context=context,
        steps=(),
        metadata={
            "feedback_count": 0,
            "cycle_completed": True,
        },
    )


def make_snapshot(
    result: LearningCycleResult,
) -> LearningCycleSnapshot:
    return LearningCycleSnapshot(
        snapshot_id="snapshot-1220",
        result=result,
        created_at_utc=datetime(
            2026,
            7,
            31,
            14,
            30,
            tzinfo=timezone.utc,
        ),
    )


def test_persistent_result_creation() -> None:
    learning_result = make_learning_result()
    snapshot = make_snapshot(
        learning_result
    )

    result = PersistentLearningRunResult(
        learning_result=learning_result,
        snapshot=snapshot,
    )

    assert (
        result.learning_result
        is learning_result
    )
    assert result.snapshot is snapshot
    assert result.snapshot_id == (
        "snapshot-1220"
    )


def test_persistent_result_delegates_properties() -> None:
    learning_result = make_learning_result()

    result = PersistentLearningRunResult(
        learning_result=learning_result,
        snapshot=make_snapshot(
            learning_result
        ),
    )

    assert (
        result.initial_context
        is learning_result.initial_context
    )
    assert (
        result.final_context
        is learning_result.final_context
    )
    assert result.steps == ()
    assert result.step_count == 0
    assert result.version_delta == 0


def test_persistent_result_is_frozen() -> None:
    learning_result = make_learning_result()

    result = PersistentLearningRunResult(
        learning_result=learning_result,
        snapshot=make_snapshot(
            learning_result
        ),
    )

    with pytest.raises(FrozenInstanceError):
        result.snapshot = make_snapshot(  # type: ignore[misc]
            learning_result
        )


def test_invalid_learning_result_is_rejected() -> None:
    valid_result = make_learning_result()

    with pytest.raises(
        TypeError,
        match="LearningCycleResult",
    ):
        PersistentLearningRunResult(
            learning_result=object(),  # type: ignore[arg-type]
            snapshot=make_snapshot(
                valid_result
            ),
        )


def test_invalid_snapshot_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="LearningCycleSnapshot",
    ):
        PersistentLearningRunResult(
            learning_result=make_learning_result(),
            snapshot=object(),  # type: ignore[arg-type]
        )


def test_mismatched_snapshot_result_is_rejected() -> None:
    first_context = LearningContext(
        cycle_id="cycle-a",
        round_no=1220,
    )
    second_context = LearningContext(
        cycle_id="cycle-b",
        round_no=1221,
    )

    first_result = LearningCycleResult(
        initial_context=first_context,
        final_context=first_context,
        steps=(),
    )
    second_result = LearningCycleResult(
        initial_context=second_context,
        final_context=second_context,
        steps=(),
    )

    snapshot = LearningCycleSnapshot(
        snapshot_id="snapshot-1220",
        result=second_result,
        created_at_utc=datetime(
            2026,
            7,
            31,
            14,
            30,
            tzinfo=timezone.utc,
        ),
    )

    with pytest.raises(
        ValueError,
        match="must match",
    ):
        PersistentLearningRunResult(
            learning_result=first_result,
            snapshot=snapshot,
        )
