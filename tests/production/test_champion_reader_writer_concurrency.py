from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import pytest

from lrp.production.champion_active_snapshot import (
    ProductionChampionActiveSnapshotReader,
)
from lrp.production.champion_registry_publisher import (
    ProductionChampionRegistryPublisher,
)
from lrp.production.champion_rollback import (
    ChampionRollbackService,
)


def _write_decision(
    path: Path,
    *,
    model: str,
) -> Path:
    payload = {
        "selection": {
            "selected_model": model,
        },
    }

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return path


def _publication_revision_id(
    registry_root: Path,
    *,
    source_sha256: str,
) -> str:
    history = (
        registry_root
        / "history"
    )

    matches = []

    for path in history.glob(
        "*.json"
    ):
        try:
            payload = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            continue

        if (
            payload.get(
                "source_sha256"
            )
            == source_sha256
        ):
            matches.append(
                path.stem
            )

    assert len(matches) == 1

    return matches[0]


def _snapshot_identity(
    snapshot,
) -> tuple[str, str]:
    decision_payload = json.loads(
        snapshot.decision_bytes.decode(
            "utf-8-sig"
        )
    )

    publication_payload = json.loads(
        snapshot.publication_bytes.decode(
            "utf-8-sig"
        )
    )

    decision_model = (
        decision_payload[
            "selection"
        ][
            "selected_model"
        ]
    )

    publication_model = (
        publication_payload[
            "selected_model"
        ]
    )

    source_sha = (
        publication_payload[
            "source_sha256"
        ]
    )

    decision_sha = (
        hashlib.sha256(
            snapshot.decision_bytes
        )
        .hexdigest()
    )

    assert decision_sha == source_sha

    assert (
        decision_model
        == publication_model
    )

    return (
        decision_model,
        decision_sha,
    )


def test_snapshot_reader_waits_for_publish_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = (
        tmp_path
        / "registry"
    )

    publisher = (
        ProductionChampionRegistryPublisher()
    )

    publisher.publish(
        source_decision=(
            _write_decision(
                tmp_path
                / "a.json",
                model="model-a",
            )
        ),
        registry_root=registry,
    )

    entered = threading.Event()
    release = threading.Event()

    original = (
        publisher
        ._write_publication_record
    )

    def blocking_write(
        *args,
        **kwargs,
    ):
        entered.set()

        assert release.wait(
            timeout=5.0
        )

        return original(
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        publisher,
        "_write_publication_record",
        blocking_write,
    )

    publish_error = []

    def publish_worker():
        try:
            publisher.publish(
                source_decision=(
                    _write_decision(
                        tmp_path
                        / "b.json",
                        model="model-b",
                    )
                ),
                registry_root=registry,
            )
        except BaseException as exc:
            publish_error.append(
                exc
            )

    publish_thread = threading.Thread(
        target=publish_worker,
        daemon=True,
    )

    publish_thread.start()

    assert entered.wait(
        timeout=5.0
    )

    reader_started = threading.Event()
    reader_finished = threading.Event()
    reader_result = []
    reader_error = []

    def reader_worker():
        reader_started.set()

        try:
            result = (
                ProductionChampionActiveSnapshotReader(
                    timeout=5.0,
                )
                .read(
                    registry
                )
            )

            reader_result.append(
                result
            )

        except BaseException as exc:
            reader_error.append(
                exc
            )

        finally:
            reader_finished.set()

    reader_thread = threading.Thread(
        target=reader_worker,
        daemon=True,
    )

    reader_thread.start()

    assert reader_started.wait(
        timeout=2.0
    )

    time.sleep(
        0.15
    )

    assert not reader_finished.is_set()

    release.set()

    publish_thread.join(
        timeout=5.0
    )

    reader_thread.join(
        timeout=5.0
    )

    assert not publish_thread.is_alive()
    assert not reader_thread.is_alive()

    assert publish_error == []
    assert reader_error == []

    assert len(
        reader_result
    ) == 1

    model, _ = _snapshot_identity(
        reader_result[0]
    )

    assert model == "model-b"


def test_snapshot_reader_waits_for_rollback_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = (
        tmp_path
        / "registry"
    )

    publisher = (
        ProductionChampionRegistryPublisher()
    )

    result_a = publisher.publish(
        source_decision=(
            _write_decision(
                tmp_path
                / "a.json",
                model="model-a",
            )
        ),
        registry_root=registry,
    )

    publisher.publish(
        source_decision=(
            _write_decision(
                tmp_path
                / "b.json",
                model="model-b",
            )
        ),
        registry_root=registry,
    )

    revision_a = (
        _publication_revision_id(
            registry,
            source_sha256=(
                result_a.source_sha256
            ),
        )
    )

    service = ChampionRollbackService(
        registry_root=registry
    )

    plan = service.plan(
        revision_a
    )

    entered = threading.Event()
    release = threading.Event()

    original = (
        service
        ._replace_active_pair
    )

    def blocking_replace(
        *args,
        **kwargs,
    ):
        entered.set()

        assert release.wait(
            timeout=5.0
        )

        return original(
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        service,
        "_replace_active_pair",
        blocking_replace,
    )

    rollback_error = []

    def rollback_worker():
        try:
            service.execute(
                plan
            )
        except BaseException as exc:
            rollback_error.append(
                exc
            )

    rollback_thread = threading.Thread(
        target=rollback_worker,
        daemon=True,
    )

    rollback_thread.start()

    assert entered.wait(
        timeout=5.0
    )

    reader_finished = threading.Event()
    reader_result = []
    reader_error = []

    def reader_worker():
        try:
            snapshot = (
                ProductionChampionActiveSnapshotReader(
                    timeout=5.0,
                )
                .read(
                    registry
                )
            )

            reader_result.append(
                snapshot
            )

        except BaseException as exc:
            reader_error.append(
                exc
            )

        finally:
            reader_finished.set()

    reader_thread = threading.Thread(
        target=reader_worker,
        daemon=True,
    )

    reader_thread.start()

    time.sleep(
        0.15
    )

    assert not reader_finished.is_set()

    release.set()

    rollback_thread.join(
        timeout=5.0
    )

    reader_thread.join(
        timeout=5.0
    )

    assert not rollback_thread.is_alive()
    assert not reader_thread.is_alive()

    assert rollback_error == []
    assert reader_error == []

    assert len(
        reader_result
    ) == 1

    model, sha = _snapshot_identity(
        reader_result[0]
    )

    assert model == "model-a"

    assert (
        sha
        == result_a.source_sha256
    )


def test_publish_reader_contention_never_returns_mixed_pair(
    tmp_path: Path,
) -> None:
    registry = (
        tmp_path
        / "registry"
    )

    publisher = (
        ProductionChampionRegistryPublisher()
    )

    publisher.publish(
        source_decision=(
            _write_decision(
                tmp_path
                / "seed.json",
                model="model-0",
            )
        ),
        registry_root=registry,
    )

    stop = threading.Event()
    errors = []
    observed = []

    def writer():
        try:
            for index in range(
                1,
                9,
            ):
                publisher.publish(
                    source_decision=(
                        _write_decision(
                            tmp_path
                            / f"m{index}.json",
                            model=(
                                f"model-{index}"
                            ),
                        )
                    ),
                    registry_root=registry,
                )
        except BaseException as exc:
            errors.append(
                exc
            )
        finally:
            stop.set()

    def reader():
        try:
            while not stop.is_set():
                snapshot = (
                    ProductionChampionActiveSnapshotReader(
                        timeout=5.0,
                    )
                    .read(
                        registry
                    )
                )

                identity = (
                    _snapshot_identity(
                        snapshot
                    )
                )

                observed.append(
                    identity
                )

        except BaseException as exc:
            errors.append(
                exc
            )

    writer_thread = threading.Thread(
        target=writer,
        daemon=True,
    )

    reader_thread = threading.Thread(
        target=reader,
        daemon=True,
    )

    reader_thread.start()
    writer_thread.start()

    writer_thread.join(
        timeout=10.0
    )

    reader_thread.join(
        timeout=10.0
    )

    assert not writer_thread.is_alive()
    assert not reader_thread.is_alive()

    assert errors == []

    assert observed

    for model, sha in observed:
        assert model.startswith(
            "model-"
        )

        assert len(sha) == 64


def test_reader_timeout_does_not_mutate_registry(
    tmp_path: Path,
) -> None:
    registry = (
        tmp_path
        / "registry"
    )

    publisher = (
        ProductionChampionRegistryPublisher()
    )

    publisher.publish(
        source_decision=(
            _write_decision(
                tmp_path
                / "a.json",
                model="model-a",
            )
        ),
        registry_root=registry,
    )

    before = {
        path.relative_to(
            registry
        ).as_posix():
        path.read_bytes()
        for path in registry.rglob(
            "*"
        )
        if (
            path.is_file()
            and path.name
            != ".writer.lock"
        )
    }

    from lrp.production.production_registry_lock import (
        ProductionRegistryLockTimeout,
        ProductionRegistryWriterLock,
    )

    with ProductionRegistryWriterLock(
        registry,
        timeout=0.0,
    ):

        with pytest.raises(
            ProductionRegistryLockTimeout,
        ):
            (
                ProductionChampionActiveSnapshotReader(
                    timeout=0.0,
                )
                .read(
                    registry
                )
            )

    after = {
        path.relative_to(
            registry
        ).as_posix():
        path.read_bytes()
        for path in registry.rglob(
            "*"
        )
        if (
            path.is_file()
            and path.name
            != ".writer.lock"
        )
    }

    assert after == before
