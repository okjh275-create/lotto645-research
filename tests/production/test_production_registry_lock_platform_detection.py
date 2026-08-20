from __future__ import annotations

import platform
from pathlib import Path

import pytest

from lrp.production.production_registry_lock import (
    ProductionRegistryWriterLock,
)


def test_writer_lock_does_not_depend_on_platform_system(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = tmp_path / "registry"

    registry.mkdir(
        parents=True,
        exist_ok=True,
    )

    def fail_platform_system() -> str:
        raise RuntimeError(
            "platform.system must not be used "
            "by production registry locking"
        )

    monkeypatch.setattr(
        platform,
        "system",
        fail_platform_system,
    )

    with ProductionRegistryWriterLock(
        registry,
        timeout=0.5,
    ):
        assert (
            registry
            / ".writer.lock"
        ).is_file()


def test_writer_lock_releases_without_platform_system(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = tmp_path / "registry"

    registry.mkdir(
        parents=True,
        exist_ok=True,
    )

    calls = 0

    def fail_platform_system() -> str:
        nonlocal calls
        calls += 1

        raise RuntimeError(
            "platform.system must not be used"
        )

    monkeypatch.setattr(
        platform,
        "system",
        fail_platform_system,
    )

    lock = ProductionRegistryWriterLock(
        registry,
        timeout=0.5,
    )

    with lock:
        pass

    # The complete acquire/release lifecycle must
    # be independent from platform.system().
    assert calls == 0
