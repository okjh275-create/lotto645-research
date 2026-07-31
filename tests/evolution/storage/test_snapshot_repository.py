from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from lrp.evolution import (
    AdaptiveWeightProfile,
    EvolutionSnapshot,
    EvolutionSnapshotSerializer,
    SnapshotNotFoundError,
    SnapshotRepository,
    SnapshotSerializationError,
)


FIXED_TIME = datetime(
    2026,
    7,
    31,
    12,
    0,
    0,
    tzinfo=timezone.utc,
)


def make_profile(
    revision: int,
) -> AdaptiveWeightProfile:
    return AdaptiveWeightProfile.default(
        revision=revision,
        generated_at=FIXED_TIME,
    )


def test_snapshot_create_preserves_profile() -> None:
    profile = make_profile(1)

    snapshot = EvolutionSnapshot.create(
        profile,
        saved_at=FIXED_TIME,
    )

    assert snapshot.profile == profile
    assert snapshot.revision == 1
    assert snapshot.saved_at == FIXED_TIME
    assert snapshot.schema_version == 1


def test_snapshot_round_trip() -> None:
    snapshot = EvolutionSnapshot.create(
        make_profile(2),
        saved_at=FIXED_TIME,
    )

    restored = EvolutionSnapshot.from_dict(
        snapshot.to_dict()
    )

    assert restored == snapshot


def test_snapshot_requires_timezone_aware_time() -> None:
    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        EvolutionSnapshot(
            profile=make_profile(1),
            saved_at=datetime(2026, 7, 31),
        )


def test_serializer_round_trip() -> None:
    serializer = EvolutionSnapshotSerializer()
    snapshot = EvolutionSnapshot.create(
        make_profile(3),
        saved_at=FIXED_TIME,
    )

    content = serializer.dumps(snapshot)
    restored = serializer.loads(content)

    assert restored == snapshot
    assert content.endswith("\n")


def test_serializer_rejects_invalid_json() -> None:
    serializer = EvolutionSnapshotSerializer()

    with pytest.raises(
        SnapshotSerializationError,
        match="invalid snapshot JSON",
    ):
        serializer.loads("{not-json}")


def test_serializer_rejects_wrong_root_type() -> None:
    serializer = EvolutionSnapshotSerializer()

    with pytest.raises(
        SnapshotSerializationError,
        match="root must be an object",
    ):
        serializer.loads("[]")


def test_serializer_rejects_unknown_schema() -> None:
    serializer = EvolutionSnapshotSerializer()
    snapshot = EvolutionSnapshot.create(
        make_profile(1),
        saved_at=FIXED_TIME,
    )
    payload = snapshot.to_dict()
    payload["schema_version"] = 999

    import json

    with pytest.raises(
        SnapshotSerializationError,
        match="invalid evolution snapshot payload",
    ):
        serializer.loads(json.dumps(payload))


def test_repository_save_and_load_revision(
    tmp_path: Path,
) -> None:
    repository = SnapshotRepository(tmp_path)

    saved = repository.save(
        make_profile(1),
        saved_at=FIXED_TIME,
    )
    loaded = repository.load_revision(1)

    assert loaded == saved
    assert repository.exists(1) is True


def test_repository_uses_padded_revision_filename(
    tmp_path: Path,
) -> None:
    repository = SnapshotRepository(tmp_path)

    repository.save(
        make_profile(7),
        saved_at=FIXED_TIME,
    )

    assert (
        tmp_path / "revision-00000007.json"
    ).exists()


def test_repository_prevents_revision_overwrite(
    tmp_path: Path,
) -> None:
    repository = SnapshotRepository(tmp_path)

    repository.save(
        make_profile(1),
        saved_at=FIXED_TIME,
    )

    with pytest.raises(FileExistsError):
        repository.save(
            make_profile(1),
            saved_at=FIXED_TIME,
        )


def test_repository_loads_latest_revision(
    tmp_path: Path,
) -> None:
    repository = SnapshotRepository(tmp_path)

    repository.save(
        make_profile(1),
        saved_at=FIXED_TIME,
    )
    repository.save(
        make_profile(3),
        saved_at=FIXED_TIME,
    )
    repository.save(
        make_profile(2),
        saved_at=FIXED_TIME,
    )

    assert repository.load_latest().revision == 3


def test_repository_history_is_revision_ordered(
    tmp_path: Path,
) -> None:
    repository = SnapshotRepository(tmp_path)

    repository.save(
        make_profile(3),
        saved_at=FIXED_TIME,
    )
    repository.save(
        make_profile(1),
        saved_at=FIXED_TIME,
    )
    repository.save(
        make_profile(2),
        saved_at=FIXED_TIME,
    )

    assert [
        snapshot.revision
        for snapshot in repository.history()
    ] == [1, 2, 3]

    assert repository.revisions() == (1, 2, 3)


def test_missing_revision_raises_not_found(
    tmp_path: Path,
) -> None:
    repository = SnapshotRepository(tmp_path)

    with pytest.raises(
        SnapshotNotFoundError,
        match="revision not found",
    ):
        repository.load_revision(10)


def test_missing_latest_raises_not_found(
    tmp_path: Path,
) -> None:
    repository = SnapshotRepository(tmp_path)

    with pytest.raises(
        SnapshotNotFoundError,
        match="no evolution snapshots",
    ):
        repository.load_latest()


def test_latest_skips_corrupt_newest_file(
    tmp_path: Path,
) -> None:
    repository = SnapshotRepository(tmp_path)

    repository.save(
        make_profile(1),
        saved_at=FIXED_TIME,
    )
    repository.save(
        make_profile(2),
        saved_at=FIXED_TIME,
    )

    (
        tmp_path / "revision-00000002.json"
    ).write_text(
        "{broken",
        encoding="utf-8",
    )

    assert repository.load_latest().revision == 1


def test_latest_can_fail_on_corrupt_newest_file(
    tmp_path: Path,
) -> None:
    repository = SnapshotRepository(tmp_path)

    repository.save(
        make_profile(1),
        saved_at=FIXED_TIME,
    )
    repository.save(
        make_profile(2),
        saved_at=FIXED_TIME,
    )

    (
        tmp_path / "revision-00000002.json"
    ).write_text(
        "{broken",
        encoding="utf-8",
    )

    with pytest.raises(
        SnapshotSerializationError,
    ):
        repository.load_latest(
            skip_corrupt=False
        )


def test_history_skips_corrupt_files(
    tmp_path: Path,
) -> None:
    repository = SnapshotRepository(tmp_path)

    repository.save(
        make_profile(1),
        saved_at=FIXED_TIME,
    )
    repository.save(
        make_profile(2),
        saved_at=FIXED_TIME,
    )

    (
        tmp_path / "revision-00000002.json"
    ).write_text(
        "invalid",
        encoding="utf-8",
    )

    assert [
        snapshot.revision
        for snapshot in repository.history()
    ] == [1]


def test_revision_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    repository = SnapshotRepository(tmp_path)
    serializer = EvolutionSnapshotSerializer()

    snapshot = EvolutionSnapshot.create(
        make_profile(2),
        saved_at=FIXED_TIME,
    )

    (
        tmp_path / "revision-00000001.json"
    ).write_text(
        serializer.dumps(snapshot),
        encoding="utf-8",
    )

    with pytest.raises(
        SnapshotSerializationError,
        match="does not match filename",
    ):
        repository.load_revision(1)


def test_unrelated_files_are_ignored(
    tmp_path: Path,
) -> None:
    repository = SnapshotRepository(tmp_path)

    repository.save(
        make_profile(1),
        saved_at=FIXED_TIME,
    )

    (tmp_path / "notes.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (
        tmp_path / "revision-invalid.json"
    ).write_text(
        "{}",
        encoding="utf-8",
    )

    assert repository.revisions() == (1,)


def test_public_storage_api_is_available(
    tmp_path: Path,
) -> None:
    repository = SnapshotRepository(tmp_path)

    assert repository.root == tmp_path