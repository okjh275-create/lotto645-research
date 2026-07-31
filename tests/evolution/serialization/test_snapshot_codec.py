from __future__ import annotations

from datetime import datetime, timezone

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
from lrp.evolution.serialization.snapshot_codec import (
    SnapshotCodec,
)


def make_snapshot() -> LearningCycleSnapshot:
    initial = LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
        version=3,
        signals={
            "frequency": 0.4,
        },
        rewards={
            "existing:strategy_c": 0.1,
        },
        weights={
            "recency": 0.35,
        },
        metadata={
            "seed": 20260731,
        },
    )
    final = LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
        version=5,
        signals={
            "frequency": 0.4,
        },
        rewards={
            "existing:strategy_c": 0.1,
            "result:ucb1:strategy_a": 0.8,
            "result:thompson:strategy_b": 0.5,
        },
        selected_policy="thompson",
        selected_arm="strategy_b",
        weights={
            "recency": 0.35,
        },
        metadata={
            "seed": 20260731,
            "feedback_source": "result",
        },
    )
    result = LearningCycleResult(
        initial_context=initial,
        final_context=final,
        steps=(
            LearningCycleStep(
                index=1,
                name="reinforcement_feedback",
                version_before=3,
                version_after=4,
                reward_key=(
                    "result:ucb1:strategy_a"
                ),
            ),
            LearningCycleStep(
                index=2,
                name="reinforcement_feedback",
                version_before=4,
                version_after=5,
                reward_key=(
                    "result:thompson:strategy_b"
                ),
            ),
        ),
        metadata={
            "feedback_count": 2,
            "cycle_completed": True,
        },
    )

    return LearningCycleSnapshot(
        snapshot_id="snapshot-1220-v5",
        result=result,
        created_at_utc=datetime(
            2026,
            7,
            31,
            8,
            30,
            15,
            123456,
            tzinfo=timezone.utc,
        ),
        source="weekly_learning",
        metadata={
            "environment": "test",
        },
    )


def test_encode_returns_snapshot_payload() -> None:
    snapshot = make_snapshot()

    payload = SnapshotCodec().encode(
        snapshot
    )

    assert payload == snapshot.to_payload()


def test_encode_rejects_invalid_snapshot() -> None:
    with pytest.raises(
        TypeError,
        match="LearningCycleSnapshot",
    ):
        SnapshotCodec().encode(
            object(),  # type: ignore[arg-type]
        )


def test_decode_restores_snapshot() -> None:
    original = make_snapshot()

    restored = SnapshotCodec().decode(
        original.to_payload()
    )

    assert restored == original
    assert restored.to_payload() == (
        original.to_payload()
    )


def test_round_trip_preserves_context_fields() -> None:
    original = make_snapshot()

    restored = SnapshotCodec().decode(
        SnapshotCodec().encode(original)
    )

    final = restored.result.final_context

    assert final.cycle_id == "cycle-1220"
    assert final.round_no == 1220
    assert final.version == 5
    assert final.selected_policy == (
        "thompson"
    )
    assert final.selected_arm == (
        "strategy_b"
    )
    assert final.signals == {
        "frequency": 0.4,
    }
    assert final.weights == {
        "recency": 0.35,
    }


def test_round_trip_preserves_steps() -> None:
    original = make_snapshot()

    restored = SnapshotCodec().decode(
        original.to_payload()
    )

    assert restored.result.steps == (
        original.result.steps
    )
    assert restored.result.step_count == 2
    assert restored.result.version_delta == 2


def test_decode_accepts_z_timestamp() -> None:
    payload = make_snapshot().to_payload()

    restored = SnapshotCodec().decode(
        payload
    )

    assert restored.created_at_utc == datetime(
        2026,
        7,
        31,
        8,
        30,
        15,
        123456,
        tzinfo=timezone.utc,
    )


def test_decode_rejects_non_mapping_payload() -> None:
    with pytest.raises(
        TypeError,
        match="payload must be a mapping",
    ):
        SnapshotCodec().decode(
            [],  # type: ignore[arg-type]
        )


def test_decode_rejects_missing_field() -> None:
    payload = make_snapshot().to_payload()
    del payload["snapshot_id"]

    with pytest.raises(
        ValueError,
        match="missing required field",
    ):
        SnapshotCodec().decode(payload)


def test_decode_rejects_unsupported_schema() -> None:
    payload = make_snapshot().to_payload()
    payload["schema_version"] = 2

    with pytest.raises(
        ValueError,
        match="unsupported snapshot schema",
    ):
        SnapshotCodec().decode(payload)


def test_decode_rejects_invalid_timestamp() -> None:
    payload = make_snapshot().to_payload()
    payload["created_at_utc"] = "invalid"

    with pytest.raises(
        ValueError,
        match="ISO-8601",
    ):
        SnapshotCodec().decode(payload)


def test_decode_rejects_naive_timestamp() -> None:
    payload = make_snapshot().to_payload()
    payload["created_at_utc"] = (
        "2026-07-31T08:30:15"
    )

    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        SnapshotCodec().decode(payload)


def test_decode_rejects_identity_mismatch() -> None:
    payload = make_snapshot().to_payload()
    payload["round_no"] = 1221

    with pytest.raises(
        ValueError,
        match="round_no does not match",
    ):
        SnapshotCodec().decode(payload)


def test_decode_rejects_step_count_mismatch() -> None:
    payload = make_snapshot().to_payload()
    payload["step_count"] = 3

    with pytest.raises(
        ValueError,
        match="step_count does not match",
    ):
        SnapshotCodec().decode(payload)


def test_decode_rejects_result_step_count_mismatch() -> None:
    payload = make_snapshot().to_payload()
    result_payload = payload["result"]

    assert isinstance(result_payload, dict)

    result_payload["step_count"] = 3

    with pytest.raises(
        ValueError,
        match="result step_count",
    ):
        SnapshotCodec().decode(payload)


def test_decode_rejects_non_contiguous_step_indexes() -> None:
    payload = make_snapshot().to_payload()
    result_payload = payload["result"]

    assert isinstance(result_payload, dict)

    steps = result_payload["steps"]
    assert isinstance(steps, list)

    steps[1]["index"] = 3

    with pytest.raises(
        ValueError,
        match="step indexes must be contiguous",
    ):
        SnapshotCodec().decode(payload)


def test_decoded_payload_is_detached() -> None:
    payload = make_snapshot().to_payload()

    restored = SnapshotCodec().decode(
        payload
    )

    result_payload = payload["result"]
    assert isinstance(result_payload, dict)

    final_context = result_payload[
        "final_context"
    ]
    assert isinstance(final_context, dict)

    rewards = final_context["rewards"]
    assert isinstance(rewards, dict)

    rewards[
        "result:thompson:strategy_b"
    ] = -1.0

    assert (
        restored.result.final_context.rewards[
            "result:thompson:strategy_b"
        ]
        == pytest.approx(0.5)
    )
