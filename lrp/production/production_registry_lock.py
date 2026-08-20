from __future__ import annotations

from contextlib import suppress
from datetime import datetime, timezone
import errno
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import BinaryIO


class ProductionRegistryLockTimeout(
    TimeoutError
):
    """Raised when a registry writer lock cannot be acquired."""


_REGISTRY_LOCKS_GUARD = threading.Lock()

_REGISTRY_LOCKS: dict[
    Path,
    threading.Lock,
] = {}


def _runtime_system() -> str:
    if os.name == "nt":
        return "Windows"

    if sys.platform.startswith(
        "linux"
    ):
        return "Linux"

    if sys.platform == "darwin":
        return "Darwin"

    return sys.platform


def _registry_thread_lock(
    registry_root: Path,
) -> threading.Lock:
    identity = (
        registry_root
        .resolve()
    )

    with _REGISTRY_LOCKS_GUARD:
        lock = _REGISTRY_LOCKS.get(
            identity
        )

        if lock is None:
            lock = threading.Lock()

            _REGISTRY_LOCKS[
                identity
            ] = lock

        return lock


class ProductionRegistryWriterLock:
    """Exclusive production-registry writer lock."""

    def __init__(
        self,
        registry_root: Path | str,
        *,
        timeout: float = 5.0,
    ) -> None:
        if isinstance(
            timeout,
            bool,
        ) or not isinstance(
            timeout,
            (int, float),
        ):
            raise TypeError(
                "timeout must be a number"
            )

        if timeout < 0:
            raise ValueError(
                "timeout must be non-negative"
            )

        self._registry_root = (
            Path(
                registry_root
            )
            .resolve()
        )

        self._timeout = float(
            timeout
        )

        self._thread_lock = (
            _registry_thread_lock(
                self._registry_root
            )
        )

        self._file: BinaryIO | None = None
        self._thread_owned = False
        self._os_owned = False

    @property
    def registry_root(
        self,
    ) -> Path:
        return self._registry_root

    @property
    def lock_path(
        self,
    ) -> Path:
        return (
            self._registry_root
            / ".writer.lock"
        )

    @property
    def timeout(
        self,
    ) -> float:
        return self._timeout

    def acquire(
        self,
    ) -> None:
        if (
            self._thread_owned
            or self._os_owned
            or self._file is not None
        ):
            raise RuntimeError(
                "writer lock instance is already acquired"
            )

        deadline = (
            time.monotonic()
            + self._timeout
        )

        if not self._acquire_thread_lock(
            deadline
        ):
            raise ProductionRegistryLockTimeout(
                "production registry writer lock timed out"
            )

        self._thread_owned = True

        try:
            self._registry_root.mkdir(
                parents=True,
                exist_ok=True,
            )

            self._file = self.lock_path.open(
                "a+b"
            )

            self._prepare_lock_byte()

            if not self._acquire_os_lock(
                deadline
            ):
                raise ProductionRegistryLockTimeout(
                    "production registry writer lock timed out"
                )

            self._os_owned = True

            self._write_diagnostic_metadata()

        except BaseException:
            self._release_after_failed_acquire()
            raise

    def release(
        self,
    ) -> None:
        if not self._thread_owned:
            raise RuntimeError(
                "writer lock is not acquired"
            )

        release_error: BaseException | None = None

        if (
            self._os_owned
            and self._file is not None
        ):
            try:
                self._release_os_lock()

            except BaseException as exc:
                release_error = exc

            else:
                self._os_owned = False

        if self._file is not None:
            try:
                self._file.close()

            except BaseException as exc:
                if release_error is None:
                    release_error = exc

            finally:
                self._file = None

        if self._thread_owned:
            self._thread_owned = False
            self._thread_lock.release()

        if release_error is not None:
            raise release_error

    def __enter__(
        self,
    ) -> ProductionRegistryWriterLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type,
        exc,
        traceback,
    ) -> bool:
        self.release()
        return False

    def _acquire_thread_lock(
        self,
        deadline: float,
    ) -> bool:
        remaining = max(
            0.0,
            deadline
            - time.monotonic(),
        )

        return self._thread_lock.acquire(
            timeout=remaining
        )

    def _prepare_lock_byte(
        self,
    ) -> None:
        assert self._file is not None

        self._file.seek(
            0,
            os.SEEK_END,
        )

        if self._file.tell() == 0:
            self._file.write(
                b"\0"
            )

            self._file.flush()

        self._file.seek(
            0
        )

    def _acquire_os_lock(
        self,
        deadline: float,
    ) -> bool:
        system = _runtime_system()

        while True:
            try:
                if system == "Windows":
                    self._acquire_windows_lock()

                elif system == "Linux":
                    self._acquire_posix_lock()

                elif system == "Darwin":
                    self._acquire_posix_lock()

                else:
                    raise RuntimeError(
                        "unsupported platform for production registry lock"
                    )

                return True

            except OSError as exc:
                if not self._is_contention_error(
                    exc
                ):
                    raise

                if (
                    time.monotonic()
                    >= deadline
                ):
                    return False

                time.sleep(
                    min(
                        0.05,
                        max(
                            0.0,
                            deadline
                            - time.monotonic(),
                        ),
                    )
                )

    def _release_os_lock(
        self,
    ) -> None:
        system = _runtime_system()

        if system == "Windows":
            self._release_windows_lock()

        elif system in {
            "Linux",
            "Darwin",
        }:
            self._release_posix_lock()

        else:
            raise RuntimeError(
                "unsupported platform for production registry lock"
            )

    def _acquire_windows_lock(
        self,
    ) -> None:
        import msvcrt

        assert self._file is not None

        self._file.seek(
            0
        )

        msvcrt.locking(
            self._file.fileno(),
            msvcrt.LK_NBLCK,
            1,
        )

    def _release_windows_lock(
        self,
    ) -> None:
        import msvcrt

        assert self._file is not None

        self._file.seek(
            0
        )

        msvcrt.locking(
            self._file.fileno(),
            msvcrt.LK_UNLCK,
            1,
        )

    def _acquire_posix_lock(
        self,
    ) -> None:
        import fcntl

        assert self._file is not None

        fcntl.flock(
            self._file.fileno(),
            (
                fcntl.LOCK_EX
                | fcntl.LOCK_NB
            ),
        )

    def _release_posix_lock(
        self,
    ) -> None:
        import fcntl

        assert self._file is not None

        fcntl.flock(
            self._file.fileno(),
            fcntl.LOCK_UN,
        )

    @staticmethod
    def _is_contention_error(
        exc: OSError,
    ) -> bool:
        if getattr(
            exc,
            "winerror",
            None,
        ) in {
            32,
            33,
            36,
        }:
            return True

        if exc.errno in {
            errno.EACCES,
            errno.EAGAIN,
        }:
            return True

        return False

    def _write_diagnostic_metadata(
        self,
    ) -> None:
        assert self._file is not None

        payload = {
            "pid": os.getpid(),
            "acquired_at": (
                datetime.now(
                    timezone.utc
                )
                .isoformat()
            ),
        }

        encoded = (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        ).encode(
            "utf-8"
        )

        self._file.seek(
            1
        )

        self._file.truncate()

        self._file.write(
            encoded
        )

        self._file.flush()

        with suppress(
            OSError
        ):
            os.fsync(
                self._file.fileno()
            )

    def _release_after_failed_acquire(
        self,
    ) -> None:
        if (
            self._os_owned
            and self._file is not None
        ):
            with suppress(
                BaseException
            ):
                self._release_os_lock()

        self._os_owned = False

        if self._file is not None:
            with suppress(
                BaseException
            ):
                self._file.close()

            self._file = None

        if self._thread_owned:
            self._thread_owned = False
            self._thread_lock.release()
