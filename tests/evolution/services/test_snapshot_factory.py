from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from lrp.evolution.contracts.learning_context import (
    LearningContext,
)
from lrp.evolution.contracts.learning_cycle import (
    LearningCycleResult,
    LearningCycleStep,
)
from lrp.evolution.services.snapshot_factory import (
    SnapshotFactory,
)


FIXED_TIME = datetime(
    2026,
    7,
    31,
    14,
    30,
    tzinfo=timezone.utc,
)


def make_result() -> LearningCycleResult:
    initial = LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
        version=1,
    )
    final = LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
        version=2,
        rewards={
            "result:ucb1:strategy_a": 0.75,
        },
        selected_policy="ucb1",
        selected_arm="strategy_a",
    )

    return LearningCycleResult(
        initial_context=initial,
        final_context=final,
        steps=(
            LearningCycleStep(
                index=1,
                name="reinforcement_feedback",
                version_before=1,
                version_after=2,
                reward_key=(
                    "result:ucb1:strategy_a"
                ),
            ),
        ),
        metadata={
            "feedback_count": 1,
        },
    )


def test_factory_creates_snapshot() -> None:
    factory = SnapshotFactory(
        clock=lambda: FIXED_TIME
    )
    result = make_result()

    snapshot = factory.create(
        result,
        snapshot_id="snapshot-1220",
        metadata={
            "source": "weekly",
        },
    )

    assert snapshot.snapshot_id == (
        "snapshot-1220"
    )
    assert snapshot.result == result
    assert snapshot.created_at_utc == (
        FIXED_TIME
    )
    assert snapshot.metadata == {
        "source": "weekly",
    }


def test_factory_uses_empty_metadata_by_default() -> None:
    factory = SnapshotFactory(
        clock=lambda: FIXED_TIME
    )

    snapshot = factory.create(
        make_result(),
        snapshot_id="snapshot-1220",
    )

    assert snapshot.metadata == {}


def test_factory_copies_metadata() -> None:
    factory = SnapshotFactory(
        clock=lambda: FIXED_TIME
    )
    metadata = {
        "source": "weekly",
    }

    snapshot = factory.create(
        make_result(),
        snapshot_id="snapshot-1220",
        metadata=metadata,
    )
    metadata["source"] = "changed"

    assert snapshot.metadata == {
        "source": "weekly",
    }


def test_factory_strips_snapshot_id() -> None:
    factory = SnapshotFactory(
        clock=lambda: FIXED_TIME
    )

    snapshot = factory.create(
        make_result(),
        snapshot_id="  snapshot-1220  ",
    )

    assert snapshot.snapshot_id == (
        "snapshot-1220"
    )


def test_factory_converts_time_to_utc() -> None:
    kst = timezone(
        timedelta(hours=9)
    )
    local_time = datetime(
        2026,
        7,
        31,
        23,
        30,
        tzinfo=kst,
    )
    factory = SnapshotFactory(
        clock=lambda: local_time
    )

    snapshot = factory.create(
        make_result(),
        snapshot_id="snapshot-1220",
    )

    assert snapshot.created_at_utc == datetime(
        2026,
        7,
        31,
        14,
        30,
        tzinfo=timezone.utc,
    )


def test_invalid_result_is_rejected() -> None:
    factory = SnapshotFactory(
        clock=lambda: FIXED_TIME
    )

    with pytest.raises(
        TypeError,
        match="LearningCycleResult",
    ):
        factory.create(
            object(),  # type: ignore[arg-type]
            snapshot_id="snapshot-1220",
        )


@pytest.mark.parametrize(
    "snapshot_id",
    ["", " ", "   "],
)
def test_empty_snapshot_id_is_rejected(
    snapshot_id: str,
) -> None:
    factory = SnapshotFactory(
        clock=lambda: FIXED_TIME
    )

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        factory.create(
            make_result(),
            snapshot_id=snapshot_id,
        )


def test_invalid_snapshot_id_type_is_rejected() -> None:
    factory = SnapshotFactory(
        clock=lambda: FIXED_TIME
    )

    with pytest.raises(
        TypeError,
        match="must be a string",
    ):
        factory.create(
            make_result(),
            snapshot_id=1220,  # type: ignore[arg-type]
        )


def test_invalid_metadata_is_rejected() -> None:
    factory = SnapshotFactory(
        clock=lambda: FIXED_TIME
    )

    with pytest.raises(
        TypeError,
        match="metadata must be a mapping",
    ):
        factory.create(
            make_result(),
            snapshot_id="snapshot-1220",
            metadata=[],  # type: ignore[arg-type]
        )


def test_invalid_clock_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="clock must be callable",
    ):
        SnapshotFactory(
            clock=object(),  # type: ignore[arg-type]
        )


def test_clock_must_return_datetime() -> None:
    factory = SnapshotFactory(
        clock=lambda: "now"  # type: ignore[return-value]
    )

    with pytest.raises(
        TypeError,
        match="must return a datetime",
    ):
        factory.create(
            make_result(),
            snapshot_id="snapshot-1220",
        )


def test_clock_must_return_aware_datetime() -> None:
    factory = SnapshotFactory(
        clock=lambda: datetime(
            2026,
            7,
            31,
            14,
            30,
        )
    )

    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        factory.create(
            make_result(),
            snapshot_id="snapshot-1220",
        )
