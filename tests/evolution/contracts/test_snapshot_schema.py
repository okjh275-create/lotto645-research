from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from types import MappingProxyType

import pytest

from lrp.evolution.contracts.learning_context import (
    LearningContext,
)
from lrp.evolution.contracts.learning_cycle import (
    LearningCycleResult,
    LearningCycleStep,
)
from lrp.evolution.contracts.snapshot_schema import (
    LearningCycleSnapshot,
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
            "draw_result:ucb1:strategy_a": 0.75,
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
                    "draw_result:ucb1:strategy_a"
                ),
            ),
        ),
        metadata={
            "feedback_count": 1,
            "cycle_completed": True,
        },
    )


def test_snapshot_creation() -> None:
    created_at = datetime(
        2026,
        7,
        31,
        8,
        30,
        tzinfo=timezone.utc,
    )

    snapshot = LearningCycleSnapshot(
        snapshot_id="snapshot-1220",
        result=make_result(),
        created_at_utc=created_at,
    )

    assert snapshot.snapshot_id == (
        "snapshot-1220"
    )
    assert snapshot.schema_version == 1
    assert snapshot.source == (
        "learning_cycle"
    )
    assert snapshot.created_at_utc == (
        created_at
    )


def test_snapshot_normalizes_text() -> None:
    snapshot = LearningCycleSnapshot(
        snapshot_id=" snapshot-1220 ",
        result=make_result(),
        created_at_utc=datetime.now(
            timezone.utc
        ),
        source=" weekly_learning ",
    )

    assert snapshot.snapshot_id == (
        "snapshot-1220"
    )
    assert snapshot.source == (
        "weekly_learning"
    )


def test_snapshot_is_frozen() -> None:
    snapshot = LearningCycleSnapshot.create(
        snapshot_id="snapshot-1220",
        result=make_result(),
    )

    with pytest.raises(FrozenInstanceError):
        snapshot.source = "changed"  # type: ignore[misc]


def test_create_generates_utc_timestamp() -> None:
    snapshot = LearningCycleSnapshot.create(
        snapshot_id="snapshot-1220",
        result=make_result(),
    )

    assert snapshot.created_at_utc.tzinfo == (
        timezone.utc
    )
    assert snapshot.created_at_utc.utcoffset() == (
        timedelta(0)
    )


def test_non_utc_timestamp_is_normalized() -> None:
    kst = timezone(
        timedelta(hours=9)
    )
    timestamp = datetime(
        2026,
        7,
        31,
        17,
        30,
        tzinfo=kst,
    )

    snapshot = LearningCycleSnapshot(
        snapshot_id="snapshot-1220",
        result=make_result(),
        created_at_utc=timestamp,
    )

    assert snapshot.created_at_utc == datetime(
        2026,
        7,
        31,
        8,
        30,
        tzinfo=timezone.utc,
    )


def test_snapshot_exposes_result_identity() -> None:
    snapshot = LearningCycleSnapshot.create(
        snapshot_id="snapshot-1220",
        result=make_result(),
    )

    assert snapshot.cycle_id == "cycle-1220"
    assert snapshot.round_no == 1220
    assert snapshot.context_version == 2
    assert snapshot.step_count == 1


def test_metadata_is_read_only() -> None:
    snapshot = LearningCycleSnapshot.create(
        snapshot_id="snapshot-1220",
        result=make_result(),
        metadata={
            "seed": 20260731,
        },
    )

    assert isinstance(
        snapshot.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        snapshot.metadata["seed"] = 1  # type: ignore[index]


def test_input_metadata_is_detached() -> None:
    metadata = {
        "seed": 20260731,
    }

    snapshot = LearningCycleSnapshot.create(
        snapshot_id="snapshot-1220",
        result=make_result(),
        metadata=metadata,
    )

    metadata["seed"] = 1

    assert snapshot.metadata["seed"] == (
        20260731
    )


def test_payload_contains_schema_fields() -> None:
    created_at = datetime(
        2026,
        7,
        31,
        8,
        30,
        15,
        123456,
        tzinfo=timezone.utc,
    )

    snapshot = LearningCycleSnapshot(
        snapshot_id="snapshot-1220",
        result=make_result(),
        created_at_utc=created_at,
        metadata={
            "seed": 20260731,
        },
    )

    payload = snapshot.to_payload()

    assert payload["schema_version"] == 1
    assert payload["snapshot_id"] == (
        "snapshot-1220"
    )
    assert payload["created_at_utc"] == (
        "2026-07-31T08:30:15.123456Z"
    )
    assert payload["source"] == (
        "learning_cycle"
    )
    assert payload["cycle_id"] == (
        "cycle-1220"
    )
    assert payload["round_no"] == 1220
    assert payload["context_version"] == 2
    assert payload["step_count"] == 1
    assert payload["metadata"] == {
        "seed": 20260731,
    }


def test_payload_result_is_detached() -> None:
    snapshot = LearningCycleSnapshot.create(
        snapshot_id="snapshot-1220",
        result=make_result(),
    )

    payload = snapshot.to_payload()
    result_payload = payload["result"]

    assert isinstance(
        result_payload,
        dict,
    )

    final_context = result_payload[
        "final_context"
    ]
    assert isinstance(
        final_context,
        dict,
    )

    rewards = final_context["rewards"]
    assert isinstance(rewards, dict)

    rewards[
        "draw_result:ucb1:strategy_a"
    ] = 0.1

    assert (
        snapshot.result.final_context.rewards[
            "draw_result:ucb1:strategy_a"
        ]
        == pytest.approx(0.75)
    )


@pytest.mark.parametrize(
    "snapshot_id",
    ["", "   "],
)
def test_empty_snapshot_id_is_rejected(
    snapshot_id: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="snapshot_id must not be empty",
    ):
        LearningCycleSnapshot.create(
            snapshot_id=snapshot_id,
            result=make_result(),
        )


def test_invalid_result_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="LearningCycleResult",
    ):
        LearningCycleSnapshot.create(
            snapshot_id="snapshot-1220",
            result=object(),  # type: ignore[arg-type]
        )


def test_naive_datetime_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        LearningCycleSnapshot(
            snapshot_id="snapshot-1220",
            result=make_result(),
            created_at_utc=datetime(
                2026,
                7,
                31,
                8,
                30,
            ),
        )


def test_invalid_datetime_type_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="must be a datetime",
    ):
        LearningCycleSnapshot(
            snapshot_id="snapshot-1220",
            result=make_result(),
            created_at_utc="2026-07-31",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "schema_version",
    [0, -1],
)
def test_invalid_schema_version_is_rejected(
    schema_version: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="greater than or equal to 1",
    ):
        LearningCycleSnapshot.create(
            snapshot_id="snapshot-1220",
            result=make_result(),
            schema_version=schema_version,
        )


def test_boolean_schema_version_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="must be an integer",
    ):
        LearningCycleSnapshot.create(
            snapshot_id="snapshot-1220",
            result=make_result(),
            schema_version=True,
        )


def test_empty_source_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="source must not be empty",
    ):
        LearningCycleSnapshot.create(
            snapshot_id="snapshot-1220",
            result=make_result(),
            source=" ",
        )


def test_complex_metadata_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="metadata values must be scalar",
    ):
        LearningCycleSnapshot.create(
            snapshot_id="snapshot-1220",
            result=make_result(),
            metadata={
                "nested": {
                    "value": 1,
                },
            },  # type: ignore[arg-type]
        )


def test_non_finite_metadata_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="must be finite",
    ):
        LearningCycleSnapshot.create(
            snapshot_id="snapshot-1220",
            result=make_result(),
            metadata={
                "score": float("nan"),
            },
        )


def test_duplicate_normalized_metadata_key_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="duplicate normalized",
    ):
        LearningCycleSnapshot.create(
            snapshot_id="snapshot-1220",
            result=make_result(),
            metadata={
                "seed": 20260731,
                " seed ": 1,
            },
        )
