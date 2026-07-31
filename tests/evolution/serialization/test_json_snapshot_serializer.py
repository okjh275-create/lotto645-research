from __future__ import annotations

import json
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
from lrp.evolution.serialization.json_snapshot_serializer import (
    JsonSnapshotSerializer,
)
from lrp.evolution.serialization.snapshot_codec import (
    SnapshotCodec,
)


def make_snapshot() -> LearningCycleSnapshot:
    initial = LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
        version=1,
        metadata={
            "description": "주간 학습",
        },
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
        metadata={
            "description": "주간 학습",
        },
    )
    result = LearningCycleResult(
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

    return LearningCycleSnapshot(
        snapshot_id="snapshot-1220",
        result=result,
        created_at_utc=datetime(
            2026,
            7,
            31,
            8,
            30,
            tzinfo=timezone.utc,
        ),
        metadata={
            "label": "학습 스냅샷",
        },
    )


def test_default_serializer_creation() -> None:
    serializer = JsonSnapshotSerializer()

    assert isinstance(
        serializer._codec,  # noqa: SLF001
        SnapshotCodec,
    )


def test_custom_codec_is_accepted() -> None:
    codec = SnapshotCodec()

    serializer = JsonSnapshotSerializer(
        codec=codec
    )

    assert (
        serializer._codec  # noqa: SLF001
        is codec
    )


def test_invalid_codec_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="SnapshotCodec",
    ):
        JsonSnapshotSerializer(
            codec=object(),  # type: ignore[arg-type]
        )


def test_serialize_returns_valid_json() -> None:
    serialized = (
        JsonSnapshotSerializer().serialize(
            make_snapshot()
        )
    )

    payload = json.loads(serialized)

    assert payload["snapshot_id"] == (
        "snapshot-1220"
    )
    assert payload["round_no"] == 1220


def test_serialization_is_deterministic() -> None:
    serializer = JsonSnapshotSerializer()
    snapshot = make_snapshot()

    first = serializer.serialize(snapshot)
    second = serializer.serialize(snapshot)

    assert first == second


def test_serialization_is_compact() -> None:
    serialized = (
        JsonSnapshotSerializer().serialize(
            make_snapshot()
        )
    )

    assert "\n" not in serialized
    assert ": " not in serialized
    assert ", " not in serialized


def test_unicode_is_preserved() -> None:
    serialized = (
        JsonSnapshotSerializer().serialize(
            make_snapshot()
        )
    )

    assert "학습 스냅샷" in serialized
    assert "\\ud559" not in serialized


def test_round_trip_preserves_snapshot() -> None:
    serializer = JsonSnapshotSerializer()
    original = make_snapshot()

    serialized = serializer.serialize(
        original
    )
    restored = serializer.deserialize(
        serialized
    )

    assert restored == original
    assert restored.to_payload() == (
        original.to_payload()
    )


def test_serialize_rejects_invalid_snapshot() -> None:
    with pytest.raises(
        TypeError,
        match="LearningCycleSnapshot",
    ):
        JsonSnapshotSerializer().serialize(
            object(),  # type: ignore[arg-type]
        )


def test_deserialize_rejects_non_string() -> None:
    with pytest.raises(
        TypeError,
        match="must be a string",
    ):
        JsonSnapshotSerializer().deserialize(
            1,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "serialized",
    ["", "   "],
)
def test_deserialize_rejects_empty_string(
    serialized: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        JsonSnapshotSerializer().deserialize(
            serialized
        )


def test_deserialize_rejects_invalid_json() -> None:
    with pytest.raises(
        ValueError,
        match="valid JSON",
    ):
        JsonSnapshotSerializer().deserialize(
            "{invalid}"
        )


@pytest.mark.parametrize(
    "serialized",
    [
        "[]",
        '"snapshot"',
        "1",
        "null",
    ],
)
def test_deserialize_rejects_non_object_root(
    serialized: str,
) -> None:
    with pytest.raises(
        TypeError,
        match="root must be an object",
    ):
        JsonSnapshotSerializer().deserialize(
            serialized
        )


def test_deserialize_rejects_unsupported_schema() -> None:
    serializer = JsonSnapshotSerializer()
    payload = make_snapshot().to_payload()
    payload["schema_version"] = 99

    serialized = json.dumps(payload)

    with pytest.raises(
        ValueError,
        match="unsupported snapshot schema",
    ):
        serializer.deserialize(serialized)
