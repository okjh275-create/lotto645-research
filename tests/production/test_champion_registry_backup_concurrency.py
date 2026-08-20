from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading
import time

import pytest

from lrp.production.champion_registry_recovery import (
    ProductionRegistryBackupService,
)
from lrp.production.production_registry_lock import (
    ProductionRegistryWriterLock,
)


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

    path.write_bytes(
        data
    )

    return data


def _build_registry(
    root: Path,
) -> Path:
    registry = (
        root
        / "registry"
    )

    active = (
        registry
        / "active"
    )

    history = (
        registry
        / "history"
    )

    decision_payload = {
        "selected_model":
            "x08-model",

        "source_sha256":
            "1" * 64,
    }

    decision_bytes = _write_json(
        active
        / "champion_decision.json",
        decision_payload,
    )

    decision_sha = (
        hashlib.sha256(
            decision_bytes
        )
        .hexdigest()
    )

    revision_id = (
        "2" * 64
    )

    publication = {
        "selected_model":
            "x08-model",

        "source_sha256":
            decision_sha,

        "revision_id":
            revision_id,
    }

    _write_json(
        active
        / "publication.json",
        publication,
    )

    _write_json(
        history
        / f"{revision_id}.json",
        publication,
    )

    _write_json(
        history
        / "decisions"
        / f"{decision_sha}.json",
        decision_payload,
    )

    _write_json(
        history
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
                path
                .relative_to(
                    root
                )
                .as_posix()
            )

            result[
                relative
            ] = path.read_bytes()

    return result


def test_backup_waits_for_existing_writer_lock(
    tmp_path: Path,
) -> None:
    registry = _build_registry(
        tmp_path
    )

    backups = (
        tmp_path
        / "backups"
    )

    acquired = (
        threading.Event()
    )

    release = (
        threading.Event()
    )

    def holder() -> None:
        with ProductionRegistryWriterLock(
            registry,
            timeout=2.0,
        ):
            acquired.set()

            assert release.wait(
                timeout=5.0
            )

    thread = threading.Thread(
        target=holder,
        daemon=True,
    )

    thread.start()

    assert acquired.wait(
        timeout=2.0
    )

    result_box: list[
        object
    ] = []

    error_box: list[
        BaseException
    ] = []

    def run_backup() -> None:
        try:
            result_box.append(
                ProductionRegistryBackupService(
                    registry
                ).backup(
                    backups
                )
            )

        except BaseException as exc:
            error_box.append(
                exc
            )

    backup_thread = threading.Thread(
        target=run_backup,
        daemon=True,
    )

    backup_thread.start()

    time.sleep(
        0.15
    )

    assert backup_thread.is_alive()
    assert not result_box
    assert not error_box

    release.set()

    thread.join(
        timeout=3.0
    )

    backup_thread.join(
        timeout=3.0
    )

    assert not thread.is_alive()
    assert not backup_thread.is_alive()
    assert not error_box
    assert len(result_box) == 1


def test_backup_snapshot_is_coherent_after_writer_release(
    tmp_path: Path,
) -> None:
    registry = _build_registry(
        tmp_path
    )

    backups = (
        tmp_path
        / "backups"
    )

    active = (
        registry
        / "active"
    )

    with ProductionRegistryWriterLock(
        registry,
        timeout=2.0,
    ):
        new_payload = {
            "selected_model":
                "x08-new",

            "source_sha256":
                "3" * 64,
        }

        decision_bytes = _write_json(
            active
            / "champion_decision.json",
            new_payload,
        )

        decision_sha = (
            hashlib.sha256(
                decision_bytes
            )
            .hexdigest()
        )

        _write_json(
            active
            / "publication.json",
            {
                "selected_model":
                    "x08-new",

                "source_sha256":
                    decision_sha,

                "revision_id":
                    "4" * 64,
            },
        )

        _write_json(
            registry
            / "history"
            / (
                ("4" * 64)
                + ".json"
            ),
            {
                "selected_model":
                    "x08-new",

                "source_sha256":
                    decision_sha,

                "revision_id":
                    "4" * 64,
            },
        )

        _write_json(
            registry
            / "history"
            / "decisions"
            / (
                decision_sha
                + ".json"
            ),
            new_payload,
        )

    result = (
        ProductionRegistryBackupService(
            registry
        )
        .backup(
            backups
        )
    )

    payload_root = (
        result.backup_root
        / "payload"
    )

    decision_bytes = (
        payload_root
        / "active"
        / "champion_decision.json"
    ).read_bytes()

    publication = json.loads(
        (
            payload_root
            / "active"
            / "publication.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert (
        publication[
            "selected_model"
        ]
        == "x08-new"
    )

    assert (
        publication[
            "source_sha256"
        ]
        == hashlib.sha256(
            decision_bytes
        ).hexdigest()
    )


def test_backup_failure_never_publishes_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _build_registry(
        tmp_path
    )

    backups = (
        tmp_path
        / "backups"
    )

    original_write_bytes = (
        Path.write_bytes
    )

    calls = 0

    def fail_mid_backup(
        self: Path,
        data: bytes,
    ) -> int:
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
                    "injected backup write failure"
                )

        return original_write_bytes(
            self,
            data,
        )

    monkeypatch.setattr(
        Path,
        "write_bytes",
        fail_mid_backup,
    )

    with pytest.raises(
        OSError,
        match="injected backup write failure",
    ):
        ProductionRegistryBackupService(
            registry
        ).backup(
            backups
        )

    complete_files = list(
        backups.rglob(
            "COMPLETE"
        )
    )

    assert complete_files == []


def test_backup_failure_does_not_mutate_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _build_registry(
        tmp_path
    )

    backups = (
        tmp_path
        / "backups"
    )

    before = _tree(
        registry
    )

    original_write_bytes = (
        Path.write_bytes
    )

    calls = 0

    def fail_mid_backup(
        self: Path,
        data: bytes,
    ) -> int:
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
                    "injected backup write failure"
                )

        return original_write_bytes(
            self,
            data,
        )

    monkeypatch.setattr(
        Path,
        "write_bytes",
        fail_mid_backup,
    )

    with pytest.raises(
        OSError,
    ):
        ProductionRegistryBackupService(
            registry
        ).backup(
            backups
        )

    after = _tree(
        registry
    )

    # Writer-lock diagnostic bytes are allowed to
    # change during a read-side coherent capture.
    before.pop(
        ".writer.lock",
        None,
    )

    after.pop(
        ".writer.lock",
        None,
    )

    assert after == before


def test_backup_success_complete_marker_matches_backup_id(
    tmp_path: Path,
) -> None:
    registry = _build_registry(
        tmp_path
    )

    result = (
        ProductionRegistryBackupService(
            registry
        )
        .backup(
            tmp_path
            / "backups"
        )
    )

    complete = (
        result.backup_root
        / "COMPLETE"
    )

    assert complete.is_file()

    assert (
        complete
        .read_text(
            encoding="utf-8"
        )
        .strip()
        == result.backup_id
    )
