from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from lrp.evolution.contracts.learning_context import (
    LearningContext,
)
from lrp.evolution.contracts.learning_cycle import (
    LearningCycleResult,
    LearningCycleStep,
)
from lrp.evolution.repositories.file_snapshot_repository import (
    FileSnapshotRepository,
)
from lrp.evolution.services.persistent_learning_service import (
    PersistentLearningService,
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


def make_result(
    *,
    reward: float = 0.75,
) -> LearningCycleResult:
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
            "result:ucb1:strategy_a": reward,
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


def make_service(
    tmp_path: Path,
) -> PersistentLearningService:
    repository = FileSnapshotRepository(
        tmp_path
    )
    factory = SnapshotFactory(
        clock=lambda: FIXED_TIME
    )

    return PersistentLearningService(
        repository,
        snapshot_factory=factory,
    )


def test_service_persists_snapshot(
    tmp_path: Path,
) -> None:
    service = make_service(tmp_path)
    result = make_result()

    snapshot = service.persist(
        result,
        snapshot_id="snapshot-1220",
        metadata={
            "source": "weekly",
        },
    )

    assert snapshot.result == result
    assert snapshot.created_at_utc == (
        FIXED_TIME
    )
    assert service.exists(
        "snapshot-1220"
    )
    assert (
        tmp_path / "snapshot-1220.json"
    ).is_file()


def test_service_loads_persisted_snapshot(
    tmp_path: Path,
) -> None:
    service = make_service(tmp_path)

    original = service.persist(
        make_result(),
        snapshot_id="snapshot-1220",
    )
    restored = service.load(
        "snapshot-1220"
    )

    assert restored == original


def test_service_lists_snapshot_ids(
    tmp_path: Path,
) -> None:
    service = make_service(tmp_path)

    service.persist(
        make_result(),
        snapshot_id="snapshot-c",
    )
    service.persist(
        make_result(),
        snapshot_id="snapshot-a",
    )
    service.persist(
        make_result(),
        snapshot_id="snapshot-b",
    )

    assert service.list_ids() == (
        "snapshot-a",
        "snapshot-b",
        "snapshot-c",
    )


def test_service_deletes_snapshot(
    tmp_path: Path,
) -> None:
    service = make_service(tmp_path)

    service.persist(
        make_result(),
        snapshot_id="snapshot-1220",
    )

    assert service.delete(
        "snapshot-1220"
    )
    assert not service.exists(
        "snapshot-1220"
    )


def test_service_returns_false_when_delete_missing(
    tmp_path: Path,
) -> None:
    service = make_service(tmp_path)

    assert (
        service.delete("missing")
        is False
    )


def test_service_rejects_duplicate_by_default(
    tmp_path: Path,
) -> None:
    service = make_service(tmp_path)

    service.persist(
        make_result(reward=0.25),
        snapshot_id="snapshot-1220",
    )

    with pytest.raises(
        FileExistsError,
        match="already exists",
    ):
        service.persist(
            make_result(reward=0.9),
            snapshot_id="snapshot-1220",
        )


def test_service_can_overwrite_snapshot(
    tmp_path: Path,
) -> None:
    service = make_service(tmp_path)

    service.persist(
        make_result(reward=0.25),
        snapshot_id="snapshot-1220",
    )
    service.persist(
        make_result(reward=0.9),
        snapshot_id="snapshot-1220",
        overwrite=True,
    )

    restored = service.load(
        "snapshot-1220"
    )

    assert (
        restored.result.final_context.rewards[
            "result:ucb1:strategy_a"
        ]
        == pytest.approx(0.9)
    )


def test_service_uses_default_factory(
    tmp_path: Path,
) -> None:
    repository = FileSnapshotRepository(
        tmp_path
    )

    service = PersistentLearningService(
        repository
    )

    assert isinstance(
        service.snapshot_factory,
        SnapshotFactory,
    )


def test_service_exposes_repository(
    tmp_path: Path,
) -> None:
    repository = FileSnapshotRepository(
        tmp_path
    )

    service = PersistentLearningService(
        repository
    )

    assert service.repository is repository


def test_invalid_repository_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="SnapshotRepository",
    ):
        PersistentLearningService(
            object(),  # type: ignore[arg-type]
        )


def test_invalid_factory_is_rejected(
    tmp_path: Path,
) -> None:
    repository = FileSnapshotRepository(
        tmp_path
    )

    with pytest.raises(
        TypeError,
        match="SnapshotFactory",
    ):
        PersistentLearningService(
            repository,
            snapshot_factory=object(),  # type: ignore[arg-type]
        )


def test_invalid_overwrite_type_is_rejected(
    tmp_path: Path,
) -> None:
    service = make_service(tmp_path)

    with pytest.raises(
        TypeError,
        match="overwrite must be a boolean",
    ):
        service.persist(
            make_result(),
            snapshot_id="snapshot-1220",
            overwrite=1,  # type: ignore[arg-type]
        )
