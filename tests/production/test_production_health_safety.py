from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def _api():
    module = importlib.import_module(
        "lrp.production.production_health"
    )

    return module.ProductionHealthService


def test_health_service_is_read_only(
    tmp_path: Path,
) -> None:
    Service = _api()

    registry = (
        tmp_path
        / "registry"
    )

    registry.mkdir(
        parents=True,
        exist_ok=True,
    )

    def managed_bytes() -> dict[str, bytes]:
        return {
            path.relative_to(
                registry
            ).as_posix():
                path.read_bytes()
            for path
            in registry.rglob("*")
            if (
                path.is_file()
                and path.name
                != ".writer.lock"
            )
        }

    before = managed_bytes()

    try:
        Service(
            registry_root=registry,
            snapshot_root=(
                tmp_path
                / "snapshots"
            ),
        ).snapshot()

    except Exception:
        pass

    after = managed_bytes()

    assert after == before

    unexpected_sidecars = [
        path.relative_to(
            registry
        ).as_posix()
        for path
        in registry.rglob("*")
        if (
            path.is_file()
            and path.name
            == ".writer.lock"
            and path.parent
            != registry
        )
    ]

    assert unexpected_sidecars == []


def test_health_service_does_not_directly_acquire_writer_lock() -> None:
    import inspect

    Service = _api()

    source = inspect.getsource(
        Service
    )

    assert (
        "ProductionRegistryWriterLock("
        not in source
    )


def test_health_service_may_use_existing_consistency_reader() -> None:
    import inspect

    Service = _api()

    source = inspect.getsource(
        Service
    )

    assert (
        "ProductionChampionAudit"
        in source
        or
        "ProductionChampionActiveSnapshotReader"
        in source
    )


def test_health_service_does_not_execute_recovery() -> None:
    import inspect

    Service = _api()

    source = inspect.getsource(
        Service
    )

    forbidden = [
        ".backup(",
        ".restore(",
    ]

    assert all(
        token not in source
        for token in forbidden
    )


def test_health_service_does_not_execute_retention_or_rollback() -> None:
    import inspect

    Service = _api()

    source = inspect.getsource(
        Service
    )

    forbidden = [
        "ChampionHistoryRetentionExecutor(",
        "ChampionRollbackService(",
        ".execute(",
        ".rollback(",
    ]

    assert all(
        token not in source
        for token in forbidden
    )
