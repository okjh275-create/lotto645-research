from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest


def _load_api():
    from lrp.production.champion_registry_recovery import (
        ProductionRegistryBackupFile,
        ProductionRegistryBackupManifest,
        ProductionRegistryBackupResult,
    )

    return (
        ProductionRegistryBackupFile,
        ProductionRegistryBackupManifest,
        ProductionRegistryBackupResult,
    )


def test_backup_file_contract() -> None:
    (
        BackupFile,
        _,
        _,
    ) = _load_api()

    item = BackupFile(
        relative_path=(
            "active/champion_decision.json"
        ),
        size=123,
        sha256="a" * 64,
        role="active_decision",
    )

    assert item.relative_path == (
        "active/champion_decision.json"
    )
    assert item.size == 123
    assert item.sha256 == "a" * 64
    assert item.role == "active_decision"


def test_backup_file_is_immutable() -> None:
    (
        BackupFile,
        _,
        _,
    ) = _load_api()

    item = BackupFile(
        relative_path=(
            "active/publication.json"
        ),
        size=5,
        sha256="b" * 64,
        role="active_publication",
    )

    with pytest.raises(
        (
            FrozenInstanceError,
            AttributeError,
        )
    ):
        item.size = 9


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "relative_path":
                    "",
                "size":
                    1,
                "sha256":
                    "a" * 64,
                "role":
                    "active_decision",
            },
            "relative_path",
        ),
        (
            {
                "relative_path":
                    "/absolute.json",
                "size":
                    1,
                "sha256":
                    "a" * 64,
                "role":
                    "active_decision",
            },
            "relative_path",
        ),
        (
            {
                "relative_path":
                    "../escape.json",
                "size":
                    1,
                "sha256":
                    "a" * 64,
                "role":
                    "active_decision",
            },
            "relative_path",
        ),
        (
            {
                "relative_path":
                    "active/publication.json",
                "size":
                    -1,
                "sha256":
                    "a" * 64,
                "role":
                    "active_publication",
            },
            "size",
        ),
        (
            {
                "relative_path":
                    "active/publication.json",
                "size":
                    1,
                "sha256":
                    "xyz",
                "role":
                    "active_publication",
            },
            "sha256",
        ),
        (
            {
                "relative_path":
                    "active/publication.json",
                "size":
                    1,
                "sha256":
                    "a" * 64,
                "role":
                    "invalid-role",
            },
            "role",
        ),
    ],
)
def test_backup_file_rejects_invalid_contract(
    kwargs,
    message: str,
) -> None:
    (
        BackupFile,
        _,
        _,
    ) = _load_api()

    with pytest.raises(
        ValueError,
        match=message,
    ):
        BackupFile(
            **kwargs
        )


def test_backup_manifest_contract() -> None:
    (
        BackupFile,
        BackupManifest,
        _,
    ) = _load_api()

    files = (
        BackupFile(
            relative_path=(
                "active/champion_decision.json"
            ),
            size=11,
            sha256="1" * 64,
            role="active_decision",
        ),
        BackupFile(
            relative_path=(
                "active/publication.json"
            ),
            size=12,
            sha256="2" * 64,
            role="active_publication",
        ),
    )

    manifest = BackupManifest(
        schema_version=1,
        created_at="2026-08-19T00:00:00+00:00",
        source_registry_root="C:/registry",
        active_model="baseline",
        active_source_sha256="1" * 64,
        active_revision_id="3" * 64,
        files=files,
    )

    assert manifest.schema_version == 1
    assert manifest.created_at == (
        "2026-08-19T00:00:00+00:00"
    )
    assert manifest.source_registry_root == (
        "C:/registry"
    )
    assert manifest.active_model == "baseline"
    assert manifest.active_source_sha256 == (
        "1" * 64
    )
    assert manifest.active_revision_id == (
        "3" * 64
    )
    assert manifest.files == files


def test_backup_manifest_requires_schema_version_one() -> None:
    (
        _,
        BackupManifest,
        _,
    ) = _load_api()

    with pytest.raises(
        ValueError,
        match="schema_version",
    ):
        BackupManifest(
            schema_version=2,
            created_at=(
                "2026-08-19T00:00:00+00:00"
            ),
            source_registry_root="C:/registry",
            active_model="baseline",
            active_source_sha256="a" * 64,
            active_revision_id="b" * 64,
            files=(),
        )


def test_backup_manifest_requires_sorted_unique_files() -> None:
    (
        BackupFile,
        BackupManifest,
        _,
    ) = _load_api()

    first = BackupFile(
        relative_path=(
            "active/publication.json"
        ),
        size=1,
        sha256="a" * 64,
        role="active_publication",
    )

    second = BackupFile(
        relative_path=(
            "active/champion_decision.json"
        ),
        size=1,
        sha256="b" * 64,
        role="active_decision",
    )

    with pytest.raises(
        ValueError,
        match="files",
    ):
        BackupManifest(
            schema_version=1,
            created_at=(
                "2026-08-19T00:00:00+00:00"
            ),
            source_registry_root="C:/registry",
            active_model="baseline",
            active_source_sha256="b" * 64,
            active_revision_id="c" * 64,
            files=(
                first,
                second,
            ),
        )

    duplicate = BackupFile(
        relative_path=(
            "active/publication.json"
        ),
        size=2,
        sha256="c" * 64,
        role="active_publication",
    )

    with pytest.raises(
        ValueError,
        match="files",
    ):
        BackupManifest(
            schema_version=1,
            created_at=(
                "2026-08-19T00:00:00+00:00"
            ),
            source_registry_root="C:/registry",
            active_model="baseline",
            active_source_sha256="a" * 64,
            active_revision_id="c" * 64,
            files=(
                first,
                duplicate,
            ),
        )


def test_backup_manifest_to_payload_is_deterministic() -> None:
    (
        BackupFile,
        BackupManifest,
        _,
    ) = _load_api()

    manifest = BackupManifest(
        schema_version=1,
        created_at=(
            "2026-08-19T00:00:00+00:00"
        ),
        source_registry_root="C:/registry",
        active_model="baseline",
        active_source_sha256="a" * 64,
        active_revision_id="b" * 64,
        files=(
            BackupFile(
                relative_path=(
                    "active/champion_decision.json"
                ),
                size=10,
                sha256="a" * 64,
                role="active_decision",
            ),
        ),
    )

    payload_1 = manifest.as_dict()
    payload_2 = manifest.as_dict()

    assert payload_1 == payload_2

    encoded_1 = json.dumps(
        payload_1,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    encoded_2 = json.dumps(
        payload_2,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    assert encoded_1 == encoded_2


def test_backup_manifest_backup_id_is_content_addressed() -> None:
    (
        BackupFile,
        BackupManifest,
        _,
    ) = _load_api()

    manifest = BackupManifest(
        schema_version=1,
        created_at=(
            "2026-08-19T00:00:00+00:00"
        ),
        source_registry_root="C:/registry",
        active_model="baseline",
        active_source_sha256="a" * 64,
        active_revision_id="b" * 64,
        files=(
            BackupFile(
                relative_path=(
                    "active/champion_decision.json"
                ),
                size=10,
                sha256="a" * 64,
                role="active_decision",
            ),
        ),
    )

    backup_id_1 = manifest.backup_id
    backup_id_2 = manifest.backup_id

    assert len(backup_id_1) == 64
    assert backup_id_1 == backup_id_2
    assert all(
        character
        in "0123456789abcdef"
        for character in backup_id_1
    )


def test_backup_manifest_rejects_invalid_hash_identity() -> None:
    (
        _,
        BackupManifest,
        _,
    ) = _load_api()

    with pytest.raises(
        ValueError,
        match="active_source_sha256",
    ):
        BackupManifest(
            schema_version=1,
            created_at=(
                "2026-08-19T00:00:00+00:00"
            ),
            source_registry_root="C:/registry",
            active_model="baseline",
            active_source_sha256="not-a-hash",
            active_revision_id="b" * 64,
            files=(),
        )

    with pytest.raises(
        ValueError,
        match="active_revision_id",
    ):
        BackupManifest(
            schema_version=1,
            created_at=(
                "2026-08-19T00:00:00+00:00"
            ),
            source_registry_root="C:/registry",
            active_model="baseline",
            active_source_sha256="a" * 64,
            active_revision_id="bad",
            files=(),
        )


def test_backup_result_contract() -> None:
    (
        _,
        _,
        BackupResult,
    ) = _load_api()

    result = BackupResult(
        backup_root=Path(
            "C:/backups/abc"
        ),
        backup_id="a" * 64,
        manifest_path=Path(
            "C:/backups/abc/manifest.json"
        ),
        file_count=5,
    )

    assert result.backup_root == Path(
        "C:/backups/abc"
    )
    assert result.backup_id == "a" * 64
    assert result.manifest_path == Path(
        "C:/backups/abc/manifest.json"
    )
    assert result.file_count == 5


def test_backup_result_rejects_invalid_values() -> None:
    (
        _,
        _,
        BackupResult,
    ) = _load_api()

    with pytest.raises(
        ValueError,
        match="backup_id",
    ):
        BackupResult(
            backup_root=Path("backup"),
            backup_id="bad",
            manifest_path=Path(
                "backup/manifest.json"
            ),
            file_count=1,
        )

    with pytest.raises(
        ValueError,
        match="file_count",
    ):
        BackupResult(
            backup_root=Path("backup"),
            backup_id="a" * 64,
            manifest_path=Path(
                "backup/manifest.json"
            ),
            file_count=-1,
        )
