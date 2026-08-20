from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


def _load_service():
    from lrp.production.champion_registry_recovery import (
        ProductionRegistryBackupService,
    )

    return ProductionRegistryBackupService


def _write_json(
    path: Path,
    payload: dict,
) -> bytes:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    path.write_bytes(data)

    return data


def _sha256(
    data: bytes,
) -> str:
    return hashlib.sha256(
        data
    ).hexdigest()


def _build_registry(
    root: Path,
) -> Path:
    registry = root / "registry"

    active = registry / "active"
    history = registry / "history"
    decisions = history / "decisions"
    rollbacks = history / "rollbacks"

    decision_payload = {
        "selected_model":
            "x06b-model",

        "source_sha256":
            "1" * 64,
    }

    decision_bytes = _write_json(
        active / "champion_decision.json",
        decision_payload,
    )

    decision_sha = _sha256(
        decision_bytes
    )

    revision_id = "2" * 64

    publication_payload = {
        "selected_model":
            "x06b-model",

        "source_sha256":
            decision_sha,

        "revision_id":
            revision_id,
    }

    _write_json(
        active / "publication.json",
        publication_payload,
    )

    _write_json(
        history / f"{revision_id}.json",
        publication_payload,
    )

    _write_json(
        decisions / f"{decision_sha}.json",
        decision_payload,
    )

    _write_json(
        rollbacks / "rollback-001.json",
        {
            "event":
                "rollback",

            "revision_id":
                revision_id,
        },
    )

    return registry


def _tree(
    root: Path,
) -> dict[str, bytes]:
    if not root.exists():
        return {}

    result = {}

    for path in sorted(
        root.rglob("*")
    ):
        if path.is_file():

            relative = (
                path.relative_to(
                    root
                )
                .as_posix()
            )

            result[
                relative
            ] = path.read_bytes()

    return result


def test_backup_service_captures_complete_registry(
    tmp_path: Path,
) -> None:
    Service = _load_service()

    registry = _build_registry(
        tmp_path
    )

    destination = (
        tmp_path
        / "backups"
    )

    result = Service(
        registry
    ).backup(
        destination
    )

    assert result.backup_root.is_dir()

    payload = (
        result.backup_root
        / "payload"
    )

    expected = {
        "active/champion_decision.json",
        "active/publication.json",
        "history/"
        + "2" * 64
        + ".json",
    }

    actual = {
        path.relative_to(
            payload
        ).as_posix()
        for path in payload.rglob("*")
        if path.is_file()
    }

    assert expected <= actual

    assert any(
        path.startswith(
            "history/decisions/"
        )
        for path in actual
    )

    assert any(
        path.startswith(
            "history/rollbacks/"
        )
        for path in actual
    )


def test_backup_service_excludes_writer_lock(
    tmp_path: Path,
) -> None:
    Service = _load_service()

    registry = _build_registry(
        tmp_path
    )

    (
        registry
        / ".writer.lock"
    ).write_bytes(
        b"diagnostic-lock"
    )

    result = Service(
        registry
    ).backup(
        tmp_path
        / "backups"
    )

    payload = (
        result.backup_root
        / "payload"
    )

    assert not (
        payload
        / ".writer.lock"
    ).exists()


def test_backup_service_writes_manifest_and_complete_last(
    tmp_path: Path,
) -> None:
    Service = _load_service()

    registry = _build_registry(
        tmp_path
    )

    result = Service(
        registry
    ).backup(
        tmp_path
        / "backups"
    )

    assert (
        result.backup_root
        / "manifest.json"
    ).is_file()

    assert (
        result.backup_root
        / "COMPLETE"
    ).is_file()

    manifest = json.loads(
        (
            result.backup_root
            / "manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert manifest[
        "schema_version"
    ] == 1

    assert manifest[
        "files"
    ]

    assert (
        result.backup_id
        == result.backup_root.name
    )


def test_backup_manifest_matches_payload_bytes(
    tmp_path: Path,
) -> None:
    Service = _load_service()

    registry = _build_registry(
        tmp_path
    )

    result = Service(
        registry
    ).backup(
        tmp_path
        / "backups"
    )

    manifest = json.loads(
        (
            result.backup_root
            / "manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    payload_root = (
        result.backup_root
        / "payload"
    )

    for item in manifest["files"]:

        path = (
            payload_root
            / item[
                "relative_path"
            ]
        )

        data = path.read_bytes()

        assert len(data) == item["size"]

        assert (
            hashlib.sha256(
                data
            ).hexdigest()
            == item[
                "sha256"
            ]
        )


def test_backup_service_is_read_only(
    tmp_path: Path,
) -> None:
    Service = _load_service()

    registry = _build_registry(
        tmp_path
    )

    before = _tree(
        registry
    )

    Service(
        registry
    ).backup(
        tmp_path
        / "backups"
    )

    after = _tree(
        registry
    )

    # Writer-lock metadata may be created or refreshed.
    before.pop(
        ".writer.lock",
        None,
    )

    after.pop(
        ".writer.lock",
        None,
    )

    assert after == before


def test_backup_service_rejects_unknown_registry_file(
    tmp_path: Path,
) -> None:
    Service = _load_service()

    registry = _build_registry(
        tmp_path
    )

    unknown = (
        registry
        / "mystery.dat"
    )

    unknown.write_bytes(
        b"unknown"
    )

    destination = (
        tmp_path
        / "backups"
    )

    with pytest.raises(
        (
            ValueError,
            RuntimeError,
        )
    ):
        Service(
            registry
        ).backup(
            destination
        )

    completed = []

    if destination.exists():

        completed = [
            path
            for path in destination.iterdir()
            if (
                path.is_dir()
                and (
                    path
                    / "COMPLETE"
                ).exists()
            )
        ]

    assert completed == []


def test_backup_service_rejects_partial_active_pair(
    tmp_path: Path,
) -> None:
    Service = _load_service()

    registry = _build_registry(
        tmp_path
    )

    (
        registry
        / "active"
        / "publication.json"
    ).unlink()

    with pytest.raises(
        (
            FileNotFoundError,
            ValueError,
            RuntimeError,
        )
    ):
        Service(
            registry
        ).backup(
            tmp_path
            / "backups"
        )


def test_backup_service_rejects_mismatched_active_pair(
    tmp_path: Path,
) -> None:
    Service = _load_service()

    registry = _build_registry(
        tmp_path
    )

    publication_path = (
        registry
        / "active"
        / "publication.json"
    )

    publication = json.loads(
        publication_path.read_text(
            encoding="utf-8"
        )
    )

    publication[
        "source_sha256"
    ] = "f" * 64

    _write_json(
        publication_path,
        publication,
    )

    with pytest.raises(
        (
            ValueError,
            RuntimeError,
        )
    ):
        Service(
            registry
        ).backup(
            tmp_path
            / "backups"
        )


def test_backup_service_uses_registry_writer_lock(
    tmp_path: Path,
) -> None:
    Service = _load_service()

    registry = _build_registry(
        tmp_path
    )

    source = Path(
        "lrp/production/"
        "champion_registry_recovery.py"
    ).read_text(
        encoding="utf-8-sig"
    )

    assert (
        "ProductionRegistryWriterLock"
        in source
    )

    service = Service(
        registry
    )

    assert service is not None


def test_backup_failure_does_not_publish_complete_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Service = _load_service()

    registry = _build_registry(
        tmp_path
    )

    destination = (
        tmp_path
        / "backups"
    )

    original_write_bytes = (
        Path.write_bytes
    )

    calls = 0

    def fail_payload_write(
        self: Path,
        data: bytes,
    ):
        nonlocal calls

        if (
            "backups"
            in self.parts
            and "payload"
            in self.parts
        ):
            calls += 1

            if calls == 2:
                raise OSError(
                    "injected backup payload failure"
                )

        return original_write_bytes(
            self,
            data,
        )

    monkeypatch.setattr(
        Path,
        "write_bytes",
        fail_payload_write,
    )

    with pytest.raises(
        OSError,
        match=(
            "injected backup payload failure"
        ),
    ):
        Service(
            registry
        ).backup(
            destination
        )

    if destination.exists():

        for candidate in destination.iterdir():

            assert not (
                candidate
                / "COMPLETE"
            ).exists()


def test_backup_file_records_are_sorted(
    tmp_path: Path,
) -> None:
    Service = _load_service()

    registry = _build_registry(
        tmp_path
    )

    result = Service(
        registry
    ).backup(
        tmp_path
        / "backups"
    )

    manifest = json.loads(
        (
            result.backup_root
            / "manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    paths = [
        item[
            "relative_path"
        ]
        for item in manifest[
            "files"
        ]
    ]

    assert paths == sorted(
        paths
    )
