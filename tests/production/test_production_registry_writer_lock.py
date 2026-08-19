from __future__ import annotations

import importlib
import inspect
from pathlib import Path
import threading

import pytest


MODULE_NAME = (
    "lrp.production."
    "production_registry_lock"
)


def _load_api():
    module = importlib.import_module(
        MODULE_NAME
    )

    timeout_type = getattr(
        module,
        "ProductionRegistryLockTimeout",
    )

    lock_type = getattr(
        module,
        "ProductionRegistryWriterLock",
    )

    return (
        module,
        timeout_type,
        lock_type,
    )


def test_writer_lock_public_api_contract() -> None:
    (
        _,
        Timeout,
        WriterLock,
    ) = _load_api()

    assert issubclass(
        Timeout,
        TimeoutError,
    )

    signature = inspect.signature(
        WriterLock
    )

    parameters = (
        signature.parameters
    )

    assert (
        "registry_root"
        in parameters
    )

    assert (
        "timeout"
        in parameters
    )

    assert (
        parameters[
            "timeout"
        ].default
        == 5.0
    )

    for name in (
        "acquire",
        "release",
        "__enter__",
        "__exit__",
    ):
        assert callable(
            getattr(
                WriterLock,
                name,
            )
        )


def test_writer_lock_context_manager_releases_for_reacquisition(
    tmp_path: Path,
) -> None:
    (
        _,
        _,
        WriterLock,
    ) = _load_api()

    registry = (
        tmp_path
        / "registry"
    )

    with WriterLock(
        registry,
        timeout=0.5,
    ):
        assert (
            registry
            / ".writer.lock"
        ).exists()

    with WriterLock(
        registry,
        timeout=0.5,
    ):
        pass


def test_writer_lock_is_non_reentrant_for_same_registry(
    tmp_path: Path,
) -> None:
    (
        _,
        Timeout,
        WriterLock,
    ) = _load_api()

    registry = (
        tmp_path
        / "registry"
    )

    first = WriterLock(
        registry,
        timeout=0.5,
    )

    first.acquire()

    try:
        second = WriterLock(
            registry,
            timeout=0.05,
        )

        with pytest.raises(
            Timeout
        ):
            second.acquire()

    finally:
        first.release()


def test_writer_lock_serializes_threads_for_same_registry(
    tmp_path: Path,
) -> None:
    (
        _,
        Timeout,
        WriterLock,
    ) = _load_api()

    registry = (
        tmp_path
        / "registry"
    )

    first = WriterLock(
        registry,
        timeout=0.5,
    )

    first.acquire()

    outcome: dict[str, object] = {}

    def contend() -> None:
        candidate = WriterLock(
            registry,
            timeout=0.10,
        )

        try:
            candidate.acquire()

        except Timeout:
            outcome[
                "timed_out"
            ] = True

        else:
            outcome[
                "acquired"
            ] = True

            candidate.release()

    thread = threading.Thread(
        target=contend,
    )

    try:
        thread.start()
        thread.join(
            timeout=2.0
        )

        assert not thread.is_alive()

        assert outcome == {
            "timed_out": True,
        }

    finally:
        first.release()


def test_writer_lock_keeps_different_registries_independent(
    tmp_path: Path,
) -> None:
    (
        _,
        _,
        WriterLock,
    ) = _load_api()

    first_registry = (
        tmp_path
        / "registry-a"
    )

    second_registry = (
        tmp_path
        / "registry-b"
    )

    first = WriterLock(
        first_registry,
        timeout=0.5,
    )

    first.acquire()

    try:
        with WriterLock(
            second_registry,
            timeout=0.10,
        ):
            pass

    finally:
        first.release()


def test_writer_lock_releases_after_context_exception(
    tmp_path: Path,
) -> None:
    (
        _,
        _,
        WriterLock,
    ) = _load_api()

    registry = (
        tmp_path
        / "registry"
    )

    class SimulatedFailure(
        RuntimeError
    ):
        pass

    with pytest.raises(
        SimulatedFailure
    ):
        with WriterLock(
            registry,
            timeout=0.5,
        ):
            raise SimulatedFailure(
                "simulated writer failure"
            )

    with WriterLock(
        registry,
        timeout=0.5,
    ):
        pass


def test_writer_lock_zero_timeout_fails_closed_under_contention(
    tmp_path: Path,
) -> None:
    (
        _,
        Timeout,
        WriterLock,
    ) = _load_api()

    registry = (
        tmp_path
        / "registry"
    )

    first = WriterLock(
        registry,
        timeout=0.5,
    )

    first.acquire()

    try:
        second = WriterLock(
            registry,
            timeout=0.0,
        )

        with pytest.raises(
            Timeout
        ):
            second.acquire()

    finally:
        first.release()


def test_stale_lock_file_existence_does_not_mean_lock_owned(
    tmp_path: Path,
) -> None:
    (
        _,
        _,
        WriterLock,
    ) = _load_api()

    registry = (
        tmp_path
        / "registry"
    )

    registry.mkdir(
        parents=True,
        exist_ok=True,
    )

    lock_path = (
        registry
        / ".writer.lock"
    )

    lock_path.write_text(
        "stale diagnostic content\n",
        encoding="utf-8",
    )

    with WriterLock(
        registry,
        timeout=0.5,
    ):
        assert lock_path.exists()


def test_resolved_registry_identity_serializes_path_aliases(
    tmp_path: Path,
) -> None:
    (
        _,
        Timeout,
        WriterLock,
    ) = _load_api()

    registry = (
        tmp_path
        / "parent"
        / "registry"
    )

    registry.mkdir(
        parents=True,
        exist_ok=True,
    )

    alias = (
        registry
        / ".."
        / "registry"
    )

    first = WriterLock(
        registry,
        timeout=0.5,
    )

    first.acquire()

    try:
        second = WriterLock(
            alias,
            timeout=0.05,
        )

        with pytest.raises(
            Timeout
        ):
            second.acquire()

    finally:
        first.release()
