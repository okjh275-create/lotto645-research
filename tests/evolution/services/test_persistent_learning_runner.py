from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from lrp.evolution.contracts.learning_context import (
    LearningContext,
)
from lrp.evolution.contracts.persistent_learning import (
    PersistentLearningRunResult,
)
from lrp.evolution.contracts.reinforcement import (
    RewardFeedback,
)
from lrp.evolution.repositories.file_snapshot_repository import (
    FileSnapshotRepository,
)
from lrp.evolution.services.learning_cycle import (
    LearningCycle,
)
from lrp.evolution.services.persistent_learning_runner import (
    PersistentLearningRunner,
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


def make_runner(
    tmp_path: Path,
) -> PersistentLearningRunner:
    repository = FileSnapshotRepository(
        tmp_path
    )
    factory = SnapshotFactory(
        clock=lambda: FIXED_TIME
    )
    service = PersistentLearningService(
        repository,
        snapshot_factory=factory,
    )

    return PersistentLearningRunner(
        service
    )


def make_context() -> LearningContext:
    return LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
        version=1,
    )


def make_feedbacks() -> tuple[RewardFeedback, ...]:
    return (
        RewardFeedback(
            source="draw_result",
            policy="ucb1",
            arm="strategy_a",
            reward=0.75,
        ),
        RewardFeedback(
            source="validation",
            policy="thompson",
            arm="strategy_b",
            reward=0.25,
        ),
    )


def test_runner_runs_cycle_and_persists_snapshot(
    tmp_path: Path,
) -> None:
    runner = make_runner(tmp_path)

    result = runner.run(
        context=make_context(),
        feedbacks=make_feedbacks(),
        snapshot_id="snapshot-1220",
        metadata={
            "source": "weekly",
        },
    )

    assert isinstance(
        result,
        PersistentLearningRunResult,
    )
    assert result.snapshot_id == (
        "snapshot-1220"
    )
    assert result.step_count == 2
    assert result.version_delta == 2
    assert result.final_context.version == 3
    assert result.snapshot.created_at_utc == (
        FIXED_TIME
    )
    assert result.snapshot.metadata == {
        "source": "weekly",
    }
    assert (
        tmp_path / "snapshot-1220.json"
    ).is_file()


def test_runner_snapshot_contains_learning_result(
    tmp_path: Path,
) -> None:
    runner = make_runner(tmp_path)

    result = runner.run(
        context=make_context(),
        feedbacks=make_feedbacks(),
        snapshot_id="snapshot-1220",
    )

    assert (
        result.snapshot.result
        == result.learning_result
    )


def test_runner_persisted_snapshot_can_be_loaded(
    tmp_path: Path,
) -> None:
    runner = make_runner(tmp_path)

    result = runner.run(
        context=make_context(),
        feedbacks=make_feedbacks(),
        snapshot_id="snapshot-1220",
    )

    restored = (
        runner.persistence_service.load(
            "snapshot-1220"
        )
    )

    assert restored == result.snapshot


def test_runner_uses_default_learning_cycle(
    tmp_path: Path,
) -> None:
    runner = make_runner(tmp_path)

    assert isinstance(
        runner.learning_cycle,
        LearningCycle,
    )


def test_runner_accepts_custom_learning_cycle(
    tmp_path: Path,
) -> None:
    service = PersistentLearningService(
        FileSnapshotRepository(tmp_path),
        snapshot_factory=SnapshotFactory(
            clock=lambda: FIXED_TIME
        ),
    )
    cycle = LearningCycle()

    runner = PersistentLearningRunner(
        service,
        learning_cycle=cycle,
    )

    assert runner.learning_cycle is cycle


def test_runner_exposes_persistence_service(
    tmp_path: Path,
) -> None:
    service = PersistentLearningService(
        FileSnapshotRepository(tmp_path)
    )

    runner = PersistentLearningRunner(
        service
    )

    assert (
        runner.persistence_service
        is service
    )


def test_runner_rejects_duplicate_snapshot(
    tmp_path: Path,
) -> None:
    runner = make_runner(tmp_path)

    runner.run(
        context=make_context(),
        feedbacks=(),
        snapshot_id="snapshot-1220",
    )

    with pytest.raises(
        FileExistsError,
        match="already exists",
    ):
        runner.run(
            context=make_context(),
            feedbacks=(),
            snapshot_id="snapshot-1220",
        )


def test_runner_can_overwrite_snapshot(
    tmp_path: Path,
) -> None:
    runner = make_runner(tmp_path)

    runner.run(
        context=make_context(),
        feedbacks=(
            RewardFeedback(
                source="result",
                arm="strategy_a",
                reward=0.25,
            ),
        ),
        snapshot_id="snapshot-1220",
    )

    runner.run(
        context=make_context(),
        feedbacks=(
            RewardFeedback(
                source="result",
                arm="strategy_a",
                reward=0.9,
            ),
        ),
        snapshot_id="snapshot-1220",
        overwrite=True,
    )

    restored = (
        runner.persistence_service.load(
            "snapshot-1220"
        )
    )

    assert (
        restored.result.final_context.rewards[
            "result:strategy_a"
        ]
        == pytest.approx(0.9)
    )


def test_runner_accepts_generator_feedbacks(
    tmp_path: Path,
) -> None:
    runner = make_runner(tmp_path)

    feedbacks = (
        RewardFeedback(
            source="result",
            arm=f"strategy_{index}",
            reward=0.5,
        )
        for index in range(3)
    )

    result = runner.run(
        context=make_context(),
        feedbacks=feedbacks,
        snapshot_id="snapshot-1220",
    )

    assert result.step_count == 3
    assert result.final_context.version == 4


def test_invalid_persistence_service_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="PersistentLearningService",
    ):
        PersistentLearningRunner(
            object(),  # type: ignore[arg-type]
        )


def test_invalid_learning_cycle_is_rejected(
    tmp_path: Path,
) -> None:
    service = PersistentLearningService(
        FileSnapshotRepository(tmp_path)
    )

    with pytest.raises(
        TypeError,
        match="LearningCycle",
    ):
        PersistentLearningRunner(
            service,
            learning_cycle=object(),  # type: ignore[arg-type]
        )


def test_invalid_overwrite_type_is_rejected(
    tmp_path: Path,
) -> None:
    runner = make_runner(tmp_path)

    with pytest.raises(
        TypeError,
        match="overwrite must be a boolean",
    ):
        runner.run(
            context=make_context(),
            feedbacks=(),
            snapshot_id="snapshot-1220",
            overwrite=1,  # type: ignore[arg-type]
        )


def test_cycle_validation_happens_before_persistence(
    tmp_path: Path,
) -> None:
    runner = make_runner(tmp_path)

    with pytest.raises(
        TypeError,
        match="context must be",
    ):
        runner.run(
            context=object(),  # type: ignore[arg-type]
            feedbacks=(),
            snapshot_id="snapshot-1220",
        )

    assert (
        runner.persistence_service.list_ids()
        == ()
    )


def test_failed_persistence_returns_no_result(
    tmp_path: Path,
) -> None:
    runner = make_runner(tmp_path)

    runner.run(
        context=make_context(),
        feedbacks=(),
        snapshot_id="snapshot-1220",
    )

    with pytest.raises(FileExistsError):
        runner.run(
            context=make_context(),
            feedbacks=make_feedbacks(),
            snapshot_id="snapshot-1220",
        )

    restored = (
        runner.persistence_service.load(
            "snapshot-1220"
        )
    )

    assert restored.result.step_count == 0
