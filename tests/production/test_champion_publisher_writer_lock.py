from __future__ import annotations

import json
from pathlib import Path
import threading
import time

import pytest

from lrp.production.champion_registry_publisher import (
    ProductionChampionRegistryPublisher,
)

from lrp.production.production_registry_lock import (
    ProductionRegistryLockTimeout,
    ProductionRegistryWriterLock,
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


def _snapshot_registry(
    registry: Path,
) -> dict[str, bytes]:
    if not registry.exists():
        return {}

    return {
        path.relative_to(
            registry
        ).as_posix():
            path.read_bytes()
        for path in registry.rglob("*")
        if (
            path.is_file()
            and path.name
            != ".writer.lock"
        )
    }


def test_publisher_uses_writer_lock_symbol() -> None:
    import inspect

    from lrp.production import (
        champion_registry_publisher,
    )

    source = inspect.getsource(
        champion_registry_publisher
    )

    assert (
        "ProductionRegistryWriterLock"
        in source
    )


def test_publisher_times_out_before_registry_mutation(
    tmp_path: Path,
) -> None:
    registry = (
        tmp_path
        / "registry"
    )

    source = _write_decision(
        tmp_path
        / "decision.json",
        model="model-a",
    )

    external_lock = (
        ProductionRegistryWriterLock(
            registry,
            timeout=0.5,
        )
    )

    external_lock.acquire()

    before = _snapshot_registry(
        registry
    )

    publisher = (
        ProductionChampionRegistryPublisher()
    )

    try:
        with pytest.raises(
            ProductionRegistryLockTimeout
        ):
            publisher.publish(
                source_decision=source,
                registry_root=registry,
            )

    finally:
        external_lock.release()

    after = _snapshot_registry(
        registry
    )

    assert after == before


def test_publisher_blocks_while_same_registry_is_locked(
    tmp_path: Path,
) -> None:
    registry = (
        tmp_path
        / "registry"
    )

    source = _write_decision(
        tmp_path
        / "decision.json",
        model="model-a",
    )

    blocker = (
        ProductionRegistryWriterLock(
            registry,
            timeout=0.5,
        )
    )

    blocker.acquire()

    outcome: dict[
        str,
        object,
    ] = {}

    publisher = (
        ProductionChampionRegistryPublisher()
    )

    def run_publish() -> None:
        try:
            result = publisher.publish(
                source_decision=source,
                registry_root=registry,
            )

        except BaseException as exc:
            outcome[
                "error"
            ] = exc

        else:
            outcome[
                "result"
            ] = result

    thread = threading.Thread(
        target=run_publish,
    )

    try:
        thread.start()

        time.sleep(
            0.15
        )

        assert thread.is_alive()

        assert (
            "result"
            not in outcome
        )

    finally:
        blocker.release()

    thread.join(
        timeout=3.0
    )

    assert not thread.is_alive()

    assert (
        "error"
        not in outcome
    )

    assert (
        "result"
        in outcome
    )


def test_publisher_releases_lock_after_success(
    tmp_path: Path,
) -> None:
    registry = (
        tmp_path
        / "registry"
    )

    source = _write_decision(
        tmp_path
        / "decision.json",
        model="model-a",
    )

    publisher = (
        ProductionChampionRegistryPublisher()
    )

    publisher.publish(
        source_decision=source,
        registry_root=registry,
    )

    with ProductionRegistryWriterLock(
        registry,
        timeout=0.5,
    ):
        pass


def test_publisher_releases_lock_after_publication_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = (
        tmp_path
        / "registry"
    )

    source = _write_decision(
        tmp_path
        / "decision.json",
        model="model-a",
    )

    publisher = (
        ProductionChampionRegistryPublisher()
    )

    def fail_record(
        *args,
        **kwargs,
    ):
        raise OSError(
            "simulated publication failure"
        )

    monkeypatch.setattr(
        publisher,
        "_write_publication_record",
        fail_record,
    )

    with pytest.raises(
        OSError,
        match="simulated publication failure",
    ):
        publisher.publish(
            source_decision=source,
            registry_root=registry,
        )

    with ProductionRegistryWriterLock(
        registry,
        timeout=0.5,
    ):
        pass


def test_publisher_serializes_two_publish_operations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = (
        tmp_path
        / "registry"
    )

    source_a = _write_decision(
        tmp_path
        / "decision-a.json",
        model="model-a",
    )

    source_b = _write_decision(
        tmp_path
        / "decision-b.json",
        model="model-b",
    )

    first_entered = (
        threading.Event()
    )

    release_first = (
        threading.Event()
    )

    original = (
        ProductionChampionRegistryPublisher
        ._write_publication_record
    )

    call_count = 0
    call_guard = threading.Lock()

    def delayed_record(
        *args,
        **kwargs,
    ):
        nonlocal call_count

        with call_guard:
            call_count += 1
            current = call_count

        if current == 1:
            first_entered.set()

            assert release_first.wait(
                timeout=3.0
            )

        return original(
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        ProductionChampionRegistryPublisher,
        "_write_publication_record",
        staticmethod(
            delayed_record
        ),
    )

    results = []
    errors = []

    def publish(
        source: Path,
    ) -> None:
        try:
            result = (
                ProductionChampionRegistryPublisher()
                .publish(
                    source_decision=source,
                    registry_root=registry,
                )
            )

            results.append(
                result
            )

        except BaseException as exc:
            errors.append(
                exc
            )

    first = threading.Thread(
        target=publish,
        args=(
            source_a,
        ),
    )

    second = threading.Thread(
        target=publish,
        args=(
            source_b,
        ),
    )

    first.start()

    assert first_entered.wait(
        timeout=3.0
    )

    second.start()

    time.sleep(
        0.15
    )

    assert second.is_alive()

    release_first.set()

    first.join(
        timeout=3.0
    )

    second.join(
        timeout=3.0
    )

    assert not first.is_alive()
    assert not second.is_alive()

    assert errors == []

    assert len(
        results
    ) == 2


def test_publisher_lock_covers_history_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = (
        tmp_path
        / "registry"
    )

    source = _write_decision(
        tmp_path
        / "decision.json",
        model="model-a",
    )

    observed = {
        "decision": False,
        "publication": False,
    }

    original_decision = (
        ProductionChampionRegistryPublisher
        ._write_decision_revision
    )

    original_publication = (
        ProductionChampionRegistryPublisher
        ._write_publication_revision
    )

    def guarded_decision(
        *args,
        **kwargs,
    ):
        probe = (
            ProductionRegistryWriterLock(
                registry,
                timeout=0.0,
            )
        )

        with pytest.raises(
            ProductionRegistryLockTimeout
        ):
            probe.acquire()

        observed[
            "decision"
        ] = True

        return original_decision(
            *args,
            **kwargs,
        )

    def guarded_publication(
        *args,
        **kwargs,
    ):
        probe = (
            ProductionRegistryWriterLock(
                registry,
                timeout=0.0,
            )
        )

        with pytest.raises(
            ProductionRegistryLockTimeout
        ):
            probe.acquire()

        observed[
            "publication"
        ] = True

        return original_publication(
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        ProductionChampionRegistryPublisher,
        "_write_decision_revision",
        staticmethod(
            guarded_decision
        ),
    )

    monkeypatch.setattr(
        ProductionChampionRegistryPublisher,
        "_write_publication_revision",
        staticmethod(
            guarded_publication
        ),
    )

    (
        ProductionChampionRegistryPublisher()
        .publish(
            source_decision=source,
            registry_root=registry,
        )
    )

    assert observed == {
        "decision": True,
        "publication": True,
    }
