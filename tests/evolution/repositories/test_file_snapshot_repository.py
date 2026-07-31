from __future__ import annotations

import json
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
from lrp.evolution.contracts.snapshot_schema import (
    LearningCycleSnapshot,
)
from lrp.evolution.repositories.file_snapshot_repository import (
    FileSnapshotRepository,
)
from lrp.evolution.serialization.json_snapshot_serializer import (
    JsonSnapshotSerializer,
)


def make_snapshot(
    snapshot_id: str = "snapshot-1220",
    *,
    reward: float = 0.75,
) -> LearningCycleSnapshot:
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
            "result:ucb1:strategy_a": reward,
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
        snapshot_id=snapshot_id,
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


def test_repository_creates_root_directory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "snapshots"

    repository = FileSnapshotRepository(
        root
    )

    assert root.is_dir()
    assert repository.root_directory == (
        root.resolve()
    )


def test_repository_accepts_string_path(
    tmp_path: Path,
) -> None:
    root = tmp_path / "snapshots"

    repository = FileSnapshotRepository(
        str(root)
    )

    assert repository.root_directory == (
        root.resolve()
    )


def test_repository_uses_default_serializer(
    tmp_path: Path,
) -> None:
    repository = FileSnapshotRepository(
        tmp_path
    )

    assert isinstance(
        repository._serializer,  # noqa: SLF001
        JsonSnapshotSerializer,
    )


def test_repository_accepts_custom_serializer(
    tmp_path: Path,
) -> None:
    serializer = JsonSnapshotSerializer()

    repository = FileSnapshotRepository(
        tmp_path,
        serializer=serializer,
    )

    assert (
        repository._serializer  # noqa: SLF001
        is serializer
    )


def test_invalid_serializer_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        TypeError,
        match="JsonSnapshotSerializer",
    ):
        FileSnapshotRepository(
            tmp_path,
            serializer=object(),  # type: ignore[arg-type]
        )


def test_save_creates_json_file(
    tmp_path: Path,
) -> None:
    repository = FileSnapshotRepository(
        tmp_path
    )

    repository.save(make_snapshot())

    path = tmp_path / "snapshot-1220.json"

    assert path.is_file()

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    assert payload["snapshot_id"] == (
        "snapshot-1220"
    )
    assert payload["round_no"] == 1220


def test_saved_file_uses_utf8(
    tmp_path: Path,
) -> None:
    repository = FileSnapshotRepository(
        tmp_path
    )

    repository.save(make_snapshot())

    content = (
        tmp_path / "snapshot-1220.json"
    ).read_text(encoding="utf-8")

    assert "학습 스냅샷" in content


def test_save_and_load_round_trip(
    tmp_path: Path,
) -> None:
    repository = FileSnapshotRepository(
        tmp_path
    )
    original = make_snapshot()

    repository.save(original)
    restored = repository.load(
        original.snapshot_id
    )

    assert restored == original
    assert restored.to_payload() == (
        original.to_payload()
    )


def test_exists_returns_expected_value(
    tmp_path: Path,
) -> None:
    repository = FileSnapshotRepository(
        tmp_path
    )

    assert (
        repository.exists("snapshot-1220")
        is False
    )

    repository.save(make_snapshot())

    assert (
        repository.exists("snapshot-1220")
        is True
    )


def test_save_rejects_existing_snapshot(
    tmp_path: Path,
) -> None:
    repository = FileSnapshotRepository(
        tmp_path
    )
    snapshot = make_snapshot()

    repository.save(snapshot)

    with pytest.raises(
        FileExistsError,
        match="already exists",
    ):
        repository.save(snapshot)


def test_save_can_overwrite_snapshot(
    tmp_path: Path,
) -> None:
    repository = FileSnapshotRepository(
        tmp_path
    )

    repository.save(
        make_snapshot(reward=0.25)
    )
    repository.save(
        make_snapshot(reward=0.9),
        overwrite=True,
    )

    restored = repository.load(
        "snapshot-1220"
    )

    assert (
        restored.result.final_context.rewards[
            "result:ucb1:strategy_a"
        ]
        == pytest.approx(0.9)
    )


def test_invalid_overwrite_type_is_rejected(
    tmp_path: Path,
) -> None:
    repository = FileSnapshotRepository(
        tmp_path
    )

    with pytest.raises(
        TypeError,
        match="overwrite must be a boolean",
    ):
        repository.save(
            make_snapshot(),
            overwrite=1,  # type: ignore[arg-type]
        )


def test_invalid_snapshot_is_rejected(
    tmp_path: Path,
) -> None:
    repository = FileSnapshotRepository(
        tmp_path
    )

    with pytest.raises(
        TypeError,
        match="LearningCycleSnapshot",
    ):
        repository.save(
            object(),  # type: ignore[arg-type]
        )


def test_load_missing_snapshot_raises(
    tmp_path: Path,
) -> None:
    repository = FileSnapshotRepository(
        tmp_path
    )

    with pytest.raises(
        FileNotFoundError,
        match="does not exist",
    ):
        repository.load("missing")


def test_list_ids_returns_sorted_ids(
    tmp_path: Path,
) -> None:
    repository = FileSnapshotRepository(
        tmp_path
    )

    repository.save(
        make_snapshot("snapshot-c")
    )
    repository.save(
        make_snapshot("snapshot-a")
    )
    repository.save(
        make_snapshot("snapshot-b")
    )

    assert repository.list_ids() == (
        "snapshot-a",
        "snapshot-b",
        "snapshot-c",
    )


def test_list_ids_ignores_non_json_files(
    tmp_path: Path,
) -> None:
    repository = FileSnapshotRepository(
        tmp_path
    )
    repository.save(make_snapshot())

    (tmp_path / "notes.txt").write_text(
        "not a snapshot",
        encoding="utf-8",
    )
    (tmp_path / ".temporary.tmp").write_text(
        "temporary",
        encoding="utf-8",
    )

    assert repository.list_ids() == (
        "snapshot-1220",
    )


def test_delete_existing_snapshot(
    tmp_path: Path,
) -> None:
    repository = FileSnapshotRepository(
        tmp_path
    )
    repository.save(make_snapshot())

    deleted = repository.delete(
        "snapshot-1220"
    )

    assert deleted is True
    assert (
        repository.exists("snapshot-1220")
        is False
    )


def test_delete_missing_snapshot(
    tmp_path: Path,
) -> None:
    repository = FileSnapshotRepository(
        tmp_path
    )

    assert repository.delete("missing") is False


@pytest.mark.parametrize(
    "snapshot_id",
    [
        "../snapshot",
        "folder/snapshot",
        r"folder\snapshot",
        ".",
        "..",
        "snapshot 1220",
        "스냅샷",
    ],
)
def test_unsafe_snapshot_id_is_rejected(
    tmp_path: Path,
    snapshot_id: str,
) -> None:
    repository = FileSnapshotRepository(
        tmp_path
    )

    with pytest.raises(
        ValueError,
        match="snapshot_id",
    ):
        repository.exists(snapshot_id)


def test_empty_snapshot_id_is_rejected(
    tmp_path: Path,
) -> None:
    repository = FileSnapshotRepository(
        tmp_path
    )

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        repository.load(" ")


def test_invalid_snapshot_id_type_is_rejected(
    tmp_path: Path,
) -> None:
    repository = FileSnapshotRepository(
        tmp_path
    )

    with pytest.raises(
        TypeError,
        match="must be a string",
    ):
        repository.exists(
            1220,  # type: ignore[arg-type]
        )


def test_corrupt_json_is_rejected(
    tmp_path: Path,
) -> None:
    repository = FileSnapshotRepository(
        tmp_path
    )

    path = tmp_path / "snapshot-1220.json"
    path.write_text(
        "{invalid}",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="valid JSON",
    ):
        repository.load("snapshot-1220")


def test_stored_identifier_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    serializer = JsonSnapshotSerializer()
    serialized = serializer.serialize(
        make_snapshot("snapshot-other")
    )

    path = tmp_path / "snapshot-1220.json"
    path.write_text(
        serialized,
        encoding="utf-8",
    )

    repository = FileSnapshotRepository(
        tmp_path
    )

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        repository.load("snapshot-1220")


def test_temporary_files_are_removed_after_save(
    tmp_path: Path,
) -> None:
    repository = FileSnapshotRepository(
        tmp_path
    )

    repository.save(make_snapshot())

    temporary_files = tuple(
        tmp_path.glob("*.tmp")
    )

    assert temporary_files == ()


def test_root_directory_file_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "snapshots"
    path.write_text(
        "not a directory",
        encoding="utf-8",
    )

    with pytest.raises(
        FileExistsError,
    ):
        FileSnapshotRepository(path)


@pytest.mark.parametrize(
    "root_directory",
    ["", "   "],
)
def test_empty_root_directory_is_rejected(
    root_directory: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        FileSnapshotRepository(
            root_directory
        )


def test_invalid_root_directory_type_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="string or Path",
    ):
        FileSnapshotRepository(
            1,  # type: ignore[arg-type]
        )
