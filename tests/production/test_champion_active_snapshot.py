from __future__ import annotations

import json
from pathlib import Path

import pytest


def _load_snapshot_api():
    from lrp.production.champion_active_snapshot import (
        ProductionChampionActiveSnapshot,
        ProductionChampionActiveSnapshotReader,
    )

    return (
        ProductionChampionActiveSnapshot,
        ProductionChampionActiveSnapshotReader,
    )


def _write_pair(
    registry_root: Path,
    *,
    model: str,
) -> tuple[bytes, bytes]:
    active = (
        registry_root
        / "active"
    )

    active.mkdir(
        parents=True,
        exist_ok=True,
    )

    decision_payload = {
        "selection": {
            "selected_model": model,
        },
    }

    decision_bytes = (
        json.dumps(
            decision_payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    ).encode(
        "utf-8"
    )

    publication_payload = {
        "selected_model": model,
        "source_sha256": (
            __import__(
                "hashlib"
            )
            .sha256(
                decision_bytes
            )
            .hexdigest()
        ),
    }

    publication_bytes = (
        json.dumps(
            publication_payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    ).encode(
        "utf-8"
    )

    (
        active
        / "champion_decision.json"
    ).write_bytes(
        decision_bytes
    )

    (
        active
        / "publication.json"
    ).write_bytes(
        publication_bytes
    )

    return (
        decision_bytes,
        publication_bytes,
    )


def test_snapshot_captures_both_active_files(
    tmp_path: Path,
) -> None:
    (
        Snapshot,
        Reader,
    ) = _load_snapshot_api()

    registry = (
        tmp_path
        / "registry"
    )

    (
        decision_bytes,
        publication_bytes,
    ) = _write_pair(
        registry,
        model="model-a",
    )

    result = Reader().read(
        registry
    )

    assert isinstance(
        result,
        Snapshot,
    )

    assert (
        result.decision_path
        == registry
        / "active"
        / "champion_decision.json"
    )

    assert (
        result.publication_path
        == registry
        / "active"
        / "publication.json"
    )

    assert (
        result.decision_bytes
        == decision_bytes
    )

    assert (
        result.publication_bytes
        == publication_bytes
    )


def test_snapshot_requires_complete_active_pair(
    tmp_path: Path,
) -> None:
    (
        _,
        Reader,
    ) = _load_snapshot_api()

    registry = (
        tmp_path
        / "registry"
    )

    active = (
        registry
        / "active"
    )

    active.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        active
        / "champion_decision.json"
    ).write_text(
        "{}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        FileNotFoundError,
    ):
        Reader().read(
            registry
        )


def test_snapshot_reader_is_read_only(
    tmp_path: Path,
) -> None:
    (
        _,
        Reader,
    ) = _load_snapshot_api()

    registry = (
        tmp_path
        / "registry"
    )

    _write_pair(
        registry,
        model="model-a",
    )

    before = {
        path.relative_to(
            registry
        ).as_posix():
        path.read_bytes()
        for path in registry.rglob(
            "*"
        )
        if path.is_file()
    }

    Reader().read(
        registry
    )

    after = {
        path.relative_to(
            registry
        ).as_posix():
        path.read_bytes()
        for path in registry.rglob(
            "*"
        )
        if path.is_file()
    }

    # .writer.lock is diagnostic lock state
    # and is not active/history mutation.
    filtered_before = {
        key: value
        for key, value in before.items()
        if key != ".writer.lock"
    }

    filtered_after = {
        key: value
        for key, value in after.items()
        if key != ".writer.lock"
    }

    assert (
        filtered_after
        == filtered_before
    )


def test_snapshot_reader_uses_registry_writer_lock(
    tmp_path: Path,
) -> None:
    (
        _,
        Reader,
    ) = _load_snapshot_api()

    registry = (
        tmp_path
        / "registry"
    )

    _write_pair(
        registry,
        model="model-a",
    )

    from lrp.production.production_registry_lock import (
        ProductionRegistryWriterLock,
    )

    lock = (
        ProductionRegistryWriterLock(
            registry,
            timeout=0.0,
        )
    )

    lock.acquire()

    try:
        from lrp.production.production_registry_lock import (
            ProductionRegistryLockTimeout,
        )

        with pytest.raises(
            ProductionRegistryLockTimeout,
        ):
            Reader(
                timeout=0.0,
            ).read(
                registry
            )

    finally:
        lock.release()


def test_snapshot_lock_releases_after_capture_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        _,
        Reader,
    ) = _load_snapshot_api()

    registry = (
        tmp_path
        / "registry"
    )

    _write_pair(
        registry,
        model="model-a",
    )

    publication = (
        registry
        / "active"
        / "publication.json"
    )

    original_read_bytes = (
        Path.read_bytes
    )

    def fail_publication(
        self: Path,
    ) -> bytes:
        if (
            self
            == publication
        ):
            raise OSError(
                "simulated publication read failure"
            )

        return original_read_bytes(
            self
        )

    monkeypatch.setattr(
        Path,
        "read_bytes",
        fail_publication,
    )

    with pytest.raises(
        OSError,
        match=(
            "simulated publication read failure"
        ),
    ):
        Reader().read(
            registry
        )

    # If the prior failure leaked ownership,
    # this immediate acquisition would fail.
    from lrp.production.production_registry_lock import (
        ProductionRegistryWriterLock,
    )

    with ProductionRegistryWriterLock(
        registry,
        timeout=0.0,
    ):
        pass
