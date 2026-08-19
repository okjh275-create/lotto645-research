from __future__ import annotations

import json
from pathlib import Path
import threading
import time

import pytest

from lrp.production.champion_registry_publisher import (
    ProductionChampionRegistryPublisher,
)

from lrp.production.champion_rollback import (
    ChampionRollbackService,
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


def _publication_revision_id(
    registry_root: Path,
    *,
    source_sha256: str,
) -> str:
    history_root = (
        registry_root
        / "history"
    )

    matches = []

    for path in history_root.glob(
        "*.json"
    ):
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if (
            payload.get(
                "source_sha256"
            )
            == source_sha256
        ):
            matches.append(
                path.stem
            )

    assert len(
        matches
    ) == 1

    return matches[0]


def _prepare_registry(
    tmp_path: Path,
):
    registry = (
        tmp_path
        / "registry"
    )

    publisher = (
        ProductionChampionRegistryPublisher()
    )

    first = publisher.publish(
        source_decision=(
            _write_decision(
                tmp_path
                / "decision-a.json",
                model="model-a",
            )
        ),
        registry_root=registry,
    )

    second = publisher.publish(
        source_decision=(
            _write_decision(
                tmp_path
                / "decision-b.json",
                model="model-b",
            )
        ),
        registry_root=registry,
    )

    target_revision = (
        _publication_revision_id(
            registry,
            source_sha256=(
                first.source_sha256
            ),
        )
    )

    return (
        registry,
        first,
        second,
        target_revision,
    )


def _snapshot_active(
    registry: Path,
) -> dict[str, bytes]:
    active = (
        registry
        / "active"
    )

    if not active.exists():
        return {}

    return {
        path.name:
            path.read_bytes()
        for path in active.iterdir()
        if path.is_file()
    }


def test_rollback_uses_writer_lock_symbol() -> None:
    import inspect

    from lrp.production import (
        champion_rollback,
    )

    source = inspect.getsource(
        champion_rollback
    )

    assert (
        "ProductionRegistryWriterLock"
        in source
    )


def test_rollback_plan_remains_lock_free(
    tmp_path: Path,
) -> None:
    (
        registry,
        _,
        _,
        target_revision,
    ) = _prepare_registry(
        tmp_path
    )

    blocker = (
        ProductionRegistryWriterLock(
            registry,
            timeout=0.5,
        )
    )

    blocker.acquire()

    try:
        service = (
            ChampionRollbackService(
                registry_root=registry,
            )
        )

        plan = service.plan(
            target_revision
        )

        assert (
            plan.target_revision_id
            == target_revision
        )

    finally:
        blocker.release()


def test_rollback_execute_blocks_while_registry_is_locked(
    tmp_path: Path,
) -> None:
    (
        registry,
        _,
        _,
        target_revision,
    ) = _prepare_registry(
        tmp_path
    )

    service = (
        ChampionRollbackService(
            registry_root=registry,
        )
    )

    plan = service.plan(
        target_revision
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

    def execute() -> None:
        try:
            result = (
                service.execute(
                    plan
                )
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
        target=execute,
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


def test_rollback_timeout_occurs_before_active_mutation(
    tmp_path: Path,
) -> None:
    (
        registry,
        _,
        _,
        target_revision,
    ) = _prepare_registry(
        tmp_path
    )

    service = (
        ChampionRollbackService(
            registry_root=registry,
        )
    )

    plan = service.plan(
        target_revision
    )

    before = _snapshot_active(
        registry
    )

    blocker = (
        ProductionRegistryWriterLock(
            registry,
            timeout=0.5,
        )
    )

    blocker.acquire()

    try:
        with pytest.raises(
            ProductionRegistryLockTimeout
        ):
            service.execute(
                plan
            )

    finally:
        blocker.release()

    after = _snapshot_active(
        registry
    )

    assert after == before


def test_rollback_final_active_validation_runs_under_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        registry,
        _,
        _,
        target_revision,
    ) = _prepare_registry(
        tmp_path
    )

    service = (
        ChampionRollbackService(
            registry_root=registry,
        )
    )

    plan = service.plan(
        target_revision
    )

    original = (
        service
        ._read_active_source_sha256
    )

    observed = {
        "locked": False,
    }

    def guarded_read():
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
            "locked"
        ] = True

        return original()

    monkeypatch.setattr(
        service,
        "_read_active_source_sha256",
        guarded_read,
    )

    service.execute(
        plan
    )

    assert observed == {
        "locked": True,
    }


def test_rollback_target_reresolution_runs_under_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        registry,
        _,
        _,
        target_revision,
    ) = _prepare_registry(
        tmp_path
    )

    service = (
        ChampionRollbackService(
            registry_root=registry,
        )
    )

    plan = service.plan(
        target_revision
    )

    original = (
        service
        ._reader
        .resolve
    )

    observed = {
        "locked": False,
    }

    call_count = 0

    def guarded_resolve(
        *args,
        **kwargs,
    ):
        nonlocal call_count

        call_count += 1

        # plan() already resolved once.
        # execute() must re-resolve under lock.
        if call_count >= 1:
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
                "locked"
            ] = True

        return original(
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        service._reader,
        "resolve",
        guarded_resolve,
    )

    service.execute(
        plan
    )

    assert observed == {
        "locked": True,
    }


def test_rollback_active_replace_runs_under_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        registry,
        _,
        _,
        target_revision,
    ) = _prepare_registry(
        tmp_path
    )

    service = (
        ChampionRollbackService(
            registry_root=registry,
        )
    )

    plan = service.plan(
        target_revision
    )

    original = (
        service
        ._replace_active_pair
    )

    observed = {
        "locked": False,
    }

    def guarded_replace(
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
            "locked"
        ] = True

        return original(
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        service,
        "_replace_active_pair",
        guarded_replace,
    )

    service.execute(
        plan
    )

    assert observed == {
        "locked": True,
    }


def test_rollback_provenance_runs_under_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        registry,
        _,
        _,
        target_revision,
    ) = _prepare_registry(
        tmp_path
    )

    service = (
        ChampionRollbackService(
            registry_root=registry,
        )
    )

    plan = service.plan(
        target_revision
    )

    original = (
        service
        ._write_rollback_provenance
    )

    observed = {
        "locked": False,
    }

    def guarded_provenance(
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
            "locked"
        ] = True

        return original(
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        service,
        "_write_rollback_provenance",
        guarded_provenance,
    )

    service.execute(
        plan
    )

    assert observed == {
        "locked": True,
    }


def test_rollback_releases_lock_after_success(
    tmp_path: Path,
) -> None:
    (
        registry,
        _,
        _,
        target_revision,
    ) = _prepare_registry(
        tmp_path
    )

    service = (
        ChampionRollbackService(
            registry_root=registry,
        )
    )

    plan = service.plan(
        target_revision
    )

    service.execute(
        plan
    )

    with ProductionRegistryWriterLock(
        registry,
        timeout=0.5,
    ):
        pass


def test_rollback_releases_lock_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        registry,
        _,
        _,
        target_revision,
    ) = _prepare_registry(
        tmp_path
    )

    service = (
        ChampionRollbackService(
            registry_root=registry,
        )
    )

    plan = service.plan(
        target_revision
    )

    def fail_provenance(
        *args,
        **kwargs,
    ):
        raise OSError(
            "simulated rollback audit failure"
        )

    monkeypatch.setattr(
        service,
        "_write_rollback_provenance",
        fail_provenance,
    )

    with pytest.raises(
        OSError,
        match="simulated rollback audit failure",
    ):
        service.execute(
            plan
        )

    with ProductionRegistryWriterLock(
        registry,
        timeout=0.5,
    ):
        pass
