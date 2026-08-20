from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading
import time

import pytest


def _load_restore_api():
    from lrp.production.champion_registry_recovery import (
        ProductionRegistryRestoreAtomicityError,
        ProductionRegistryRestoreResult,
        ProductionRegistryRestoreService,
    )

    return (
        ProductionRegistryRestoreAtomicityError,
        ProductionRegistryRestoreResult,
        ProductionRegistryRestoreService,
    )


def _write_json(
    path: Path,
    payload: object,
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

    path.write_bytes(
        data
    )

    return data


def _sha256(
    data: bytes,
) -> str:
    return hashlib.sha256(
        data
    ).hexdigest()


def _tree(
    root: Path,
    *,
    exclude_writer_lock: bool = True,
) -> dict[str, bytes]:
    if not root.exists():
        return {}

    result: dict[
        str,
        bytes,
    ] = {}

    for path in sorted(
        root.rglob("*")
    ):
        if not path.is_file():
            continue

        relative = (
            path.relative_to(
                root
            )
            .as_posix()
        )

        if (
            exclude_writer_lock
            and relative == ".writer.lock"
        ):
            continue

        result[
            relative
        ] = path.read_bytes()

    return result


def _build_registry(
    root: Path,
    *,
    model: str,
    revision_digit: str,
) -> Path:
    registry = (
        root
        / "registry"
    )

    decision_payload = {
        "selected_model":
            model,

        "source_sha256":
            "1" * 64,
    }

    decision_bytes = _write_json(
        registry
        / "active"
        / "champion_decision.json",
        decision_payload,
    )

    decision_sha = _sha256(
        decision_bytes
    )

    revision_id = (
        revision_digit
        * 64
    )

    publication = {
        "selected_model":
            model,

        "source_sha256":
            decision_sha,

        "revision_id":
            revision_id,
    }

    _write_json(
        registry
        / "active"
        / "publication.json",
        publication,
    )

    _write_json(
        registry
        / "history"
        / f"{revision_id}.json",
        publication,
    )

    (
        registry
        / "history"
        / "decisions"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        registry
        / "history"
        / "decisions"
        / f"{decision_sha}.json"
    ).write_bytes(
        decision_bytes
    )

    _write_json(
        registry
        / "history"
        / "rollbacks"
        / "rollback-001.json",
        {
            "event":
                "rollback",

            "revision_id":
                revision_id,
        },
    )

    return registry


def _make_backup(
    root: Path,
) -> tuple[
    Path,
    Path,
]:
    from lrp.production.champion_registry_recovery import (
        ProductionRegistryBackupService,
    )

    source = _build_registry(
        root / "source",
        model="restore-source",
        revision_digit="7",
    )

    result = (
        ProductionRegistryBackupService(
            source
        )
        .backup(
            root / "backups"
        )
    )

    return (
        source,
        result.backup_root,
    )


def test_restore_result_contract(
    tmp_path: Path,
) -> None:
    (
        _,
        RestoreResult,
        _,
    ) = _load_restore_api()

    result = RestoreResult(
        registry_root=Path(
            "C:/registry"
        ),
        backup_id="a" * 64,
        restored_file_count=5,
    )

    assert result.registry_root == Path(
        "C:/registry"
    )

    assert result.backup_id == (
        "a" * 64
    )

    assert (
        result.restored_file_count
        == 5
    )

    with pytest.raises(
        ValueError,
        match="backup_id",
    ):
        RestoreResult(
            registry_root=Path(
                "registry"
            ),
            backup_id="bad",
            restored_file_count=1,
        )

    with pytest.raises(
        ValueError,
        match="restored_file_count",
    ):
        RestoreResult(
            registry_root=Path(
                "registry"
            ),
            backup_id="a" * 64,
            restored_file_count=-1,
        )


def test_restore_rejects_incomplete_backup(
    tmp_path: Path,
) -> None:
    (
        _,
        _,
        RestoreService,
    ) = _load_restore_api()

    _, backup_root = _make_backup(
        tmp_path
    )

    (
        backup_root
        / "COMPLETE"
    ).unlink()

    destination = (
        tmp_path
        / "destination"
        / "registry"
    )

    with pytest.raises(
        (
            FileNotFoundError,
            ValueError,
            RuntimeError,
        ),
    ):
        RestoreService(
            destination
        ).restore(
            backup_root
        )

    assert not destination.exists()


def test_restore_rejects_manifest_identity_mismatch(
    tmp_path: Path,
) -> None:
    (
        _,
        _,
        RestoreService,
    ) = _load_restore_api()

    _, backup_root = _make_backup(
        tmp_path
    )

    manifest_path = (
        backup_root
        / "manifest.json"
    )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    manifest[
        "active_model"
    ] = "tampered-model"

    _write_json(
        manifest_path,
        manifest,
    )

    destination = (
        tmp_path
        / "destination"
        / "registry"
    )

    with pytest.raises(
        (
            ValueError,
            RuntimeError,
        ),
    ):
        RestoreService(
            destination
        ).restore(
            backup_root
        )

    assert not destination.exists()


def test_restore_rejects_payload_hash_mismatch(
    tmp_path: Path,
) -> None:
    (
        _,
        _,
        RestoreService,
    ) = _load_restore_api()

    _, backup_root = _make_backup(
        tmp_path
    )

    target = (
        backup_root
        / "payload"
        / "active"
        / "champion_decision.json"
    )

    target.write_bytes(
        b'{"tampered":true}\n'
    )

    destination = (
        tmp_path
        / "destination"
        / "registry"
    )

    with pytest.raises(
        (
            ValueError,
            RuntimeError,
        ),
    ):
        RestoreService(
            destination
        ).restore(
            backup_root
        )

    assert not destination.exists()


def test_restore_rejects_partial_active_pair(
    tmp_path: Path,
) -> None:
    (
        _,
        _,
        RestoreService,
    ) = _load_restore_api()

    _, backup_root = _make_backup(
        tmp_path
    )

    (
        backup_root
        / "payload"
        / "active"
        / "publication.json"
    ).unlink()

    destination = (
        tmp_path
        / "destination"
        / "registry"
    )

    with pytest.raises(
        (
            FileNotFoundError,
            ValueError,
            RuntimeError,
        ),
    ):
        RestoreService(
            destination
        ).restore(
            backup_root
        )

    assert not destination.exists()


def test_restore_rejects_mismatched_active_pair(
    tmp_path: Path,
) -> None:
    (
        _,
        _,
        RestoreService,
    ) = _load_restore_api()

    _, backup_root = _make_backup(
        tmp_path
    )

    publication_path = (
        backup_root
        / "payload"
        / "active"
        / "publication.json"
    )

    publication = json.loads(
        publication_path.read_text(
            encoding="utf-8"
        )
    )

    publication[
        "selected_model"
    ] = "different-model"

    publication_bytes = _write_json(
        publication_path,
        publication,
    )

    manifest_path = (
        backup_root
        / "manifest.json"
    )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    for record in manifest[
        "files"
    ]:
        if (
            record[
                "relative_path"
            ]
            == "active/publication.json"
        ):
            record["size"] = len(
                publication_bytes
            )
            record["sha256"] = _sha256(
                publication_bytes
            )

    _write_json(
        manifest_path,
        manifest,
    )

    destination = (
        tmp_path
        / "destination"
        / "registry"
    )

    with pytest.raises(
        (
            ValueError,
            RuntimeError,
        ),
    ):
        RestoreService(
            destination
        ).restore(
            backup_root
        )

    assert not destination.exists()


def test_restore_rejects_unknown_destination_file(
    tmp_path: Path,
) -> None:
    (
        _,
        _,
        RestoreService,
    ) = _load_restore_api()

    _, backup_root = _make_backup(
        tmp_path
    )

    destination = _build_registry(
        tmp_path
        / "destination",
        model="existing",
        revision_digit="8",
    )

    unknown = (
        destination
        / "unexpected.bin"
    )

    unknown.write_bytes(
        b"unknown"
    )

    before = _tree(
        destination
    )

    with pytest.raises(
        (
            ValueError,
            RuntimeError,
        ),
    ):
        RestoreService(
            destination
        ).restore(
            backup_root
        )

    after = _tree(
        destination
    )

    assert after == before


def test_restore_uses_registry_writer_lock(
    tmp_path: Path,
) -> None:
    (
        _,
        _,
        RestoreService,
    ) = _load_restore_api()

    from lrp.production.production_registry_lock import (
        ProductionRegistryWriterLock,
    )

    _, backup_root = _make_backup(
        tmp_path
    )

    destination = (
        tmp_path
        / "destination"
        / "registry"
    )

    acquired = (
        threading.Event()
    )

    release = (
        threading.Event()
    )

    def holder() -> None:
        with ProductionRegistryWriterLock(
            destination,
            timeout=2.0,
        ):
            acquired.set()

            assert release.wait(
                timeout=5.0
            )

    holder_thread = threading.Thread(
        target=holder,
        daemon=True,
    )

    holder_thread.start()

    assert acquired.wait(
        timeout=2.0
    )

    result_box: list[
        object
    ] = []

    error_box: list[
        BaseException
    ] = []

    def restore() -> None:
        try:
            result_box.append(
                RestoreService(
                    destination
                ).restore(
                    backup_root
                )
            )

        except BaseException as exc:
            error_box.append(
                exc
            )

    restore_thread = threading.Thread(
        target=restore,
        daemon=True,
    )

    restore_thread.start()

    time.sleep(
        0.15
    )

    assert restore_thread.is_alive()
    assert result_box == []
    assert error_box == []

    release.set()

    holder_thread.join(
        timeout=3.0
    )

    restore_thread.join(
        timeout=3.0
    )

    assert not holder_thread.is_alive()
    assert not restore_thread.is_alive()
    assert error_box == []
    assert len(result_box) == 1


def test_restore_replaces_registry_with_backup_bytes(
    tmp_path: Path,
) -> None:
    (
        _,
        _,
        RestoreService,
    ) = _load_restore_api()

    source, backup_root = _make_backup(
        tmp_path
    )

    destination = _build_registry(
        tmp_path
        / "destination",
        model="old-model",
        revision_digit="8",
    )

    result = (
        RestoreService(
            destination
        )
        .restore(
            backup_root
        )
    )

    source_tree = _tree(
        source
    )

    restored_tree = _tree(
        destination
    )

    assert (
        restored_tree
        == source_tree
    )

    assert (
        result.restored_file_count
        == len(
            source_tree
        )
    )


def test_restore_does_not_restore_writer_lock(
    tmp_path: Path,
) -> None:
    (
        _,
        _,
        RestoreService,
    ) = _load_restore_api()

    _, backup_root = _make_backup(
        tmp_path
    )

    payload_lock = (
        backup_root
        / "payload"
        / ".writer.lock"
    )

    assert not payload_lock.exists()

    destination = (
        tmp_path
        / "destination"
        / "registry"
    )

    RestoreService(
        destination
    ).restore(
        backup_root
    )

    # A lock diagnostic file may be created by
    # the restore operation itself, but it must
    # never come from backup payload.
    payload_paths = {
        path.relative_to(
            backup_root
            / "payload"
        ).as_posix()
        for path in (
            backup_root
            / "payload"
        ).rglob("*")
        if path.is_file()
    }

    assert ".writer.lock" not in payload_paths


def test_restore_failure_restores_original_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        _,
        _,
        RestoreService,
    ) = _load_restore_api()

    _, backup_root = _make_backup(
        tmp_path
    )

    destination = _build_registry(
        tmp_path
        / "destination",
        model="original-model",
        revision_digit="9",
    )

    before = _tree(
        destination
    )

    original_write_bytes = (
        Path.write_bytes
    )

    writes = 0

    def fail_mid_restore(
        self: Path,
        data: bytes,
    ) -> int:
        nonlocal writes

        try:
            self.relative_to(
                destination
            )

        except ValueError:
            return original_write_bytes(
                self,
                data,
            )

        if (
            self.name
            != ".writer.lock"
        ):
            writes += 1

            if writes == 2:
                raise OSError(
                    "injected restore write failure"
                )

        return original_write_bytes(
            self,
            data,
        )

    monkeypatch.setattr(
        Path,
        "write_bytes",
        fail_mid_restore,
    )

    with pytest.raises(
        OSError,
        match="injected restore write failure",
    ):
        RestoreService(
            destination
        ).restore(
            backup_root
        )

    after = _tree(
        destination
    )

    assert after == before


def test_restore_removes_new_files_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        _,
        _,
        RestoreService,
    ) = _load_restore_api()

    source, backup_root = _make_backup(
        tmp_path
    )

    destination = (
        tmp_path
        / "destination"
        / "registry"
    )

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    before = _tree(
        destination
    )

    source_paths = set(
        _tree(
            source
        )
    )

    original_write_bytes = (
        Path.write_bytes
    )

    writes = 0

    def fail_after_first_new_file(
        self: Path,
        data: bytes,
    ) -> int:
        nonlocal writes

        try:
            relative = (
                self.relative_to(
                    destination
                )
                .as_posix()
            )

        except ValueError:
            return original_write_bytes(
                self,
                data,
            )

        if (
            relative != ".writer.lock"
            and relative in source_paths
        ):
            writes += 1

            if writes == 2:
                raise OSError(
                    "injected new-file failure"
                )

        return original_write_bytes(
            self,
            data,
        )

    monkeypatch.setattr(
        Path,
        "write_bytes",
        fail_after_first_new_file,
    )

    with pytest.raises(
        OSError,
        match="injected new-file failure",
    ):
        RestoreService(
            destination
        ).restore(
            backup_root
        )

    after = _tree(
        destination
    )

    assert after == before


def test_restore_compensation_failure_raises_atomicity_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        RestoreAtomicityError,
        _,
        RestoreService,
    ) = _load_restore_api()

    _, backup_root = _make_backup(
        tmp_path
    )

    destination = _build_registry(
        tmp_path
        / "destination",
        model="atomicity-original",
        revision_digit="9",
    )

    original_write_bytes = (
        Path.write_bytes
    )

    writes = 0
    failure_started = False

    def fail_restore_and_compensation(
        self: Path,
        data: bytes,
    ) -> int:
        nonlocal writes
        nonlocal failure_started

        try:
            self.relative_to(
                destination
            )

        except ValueError:
            return original_write_bytes(
                self,
                data,
            )

        if self.name == ".writer.lock":
            return original_write_bytes(
                self,
                data,
            )

        writes += 1

        if writes == 2:
            failure_started = True

            raise OSError(
                "primary restore failure"
            )

        if (
            failure_started
            and writes >= 3
        ):
            raise OSError(
                "compensation failure"
            )

        return original_write_bytes(
            self,
            data,
        )

    monkeypatch.setattr(
        Path,
        "write_bytes",
        fail_restore_and_compensation,
    )

    with pytest.raises(
        RestoreAtomicityError,
    ):
        RestoreService(
            destination
        ).restore(
            backup_root
        )


def test_restore_result_reports_backup_identity(
    tmp_path: Path,
) -> None:
    (
        _,
        _,
        RestoreService,
    ) = _load_restore_api()

    source, backup_root = _make_backup(
        tmp_path
    )

    destination = (
        tmp_path
        / "destination"
        / "registry"
    )

    result = (
        RestoreService(
            destination
        )
        .restore(
            backup_root
        )
    )

    assert (
        result.backup_id
        == backup_root.name
    )

    assert (
        result.registry_root
        == destination
    )

    assert (
        result.restored_file_count
        == len(
            _tree(
                source
            )
        )
    )
