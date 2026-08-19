from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
import threading
import time

from lrp.production.champion_registry_publisher import (
    ProductionChampionRegistryPublisher,
)

from lrp.production.champion_rollback import (
    ChampionRollbackService,
)

from lrp.production.production_registry_lock import (
    ProductionRegistryWriterLock,
)


def _write_decision(
    path: Path,
    *,
    model: str,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "selection": {
                    "selected_model": model,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return path


def _revision_id(
    registry: Path,
    *,
    source_sha256: str,
) -> str:
    matches = []

    for path in (
        registry
        / "history"
    ).glob(
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

    assert len(matches) == 1

    return matches[0]


def test_same_registry_lock_serializes_independent_processes(
    tmp_path: Path,
) -> None:
    registry = (
        tmp_path
        / "registry"
    )

    holder = r"""
import sys
import time
from pathlib import Path

from lrp.production.production_registry_lock import (
    ProductionRegistryWriterLock,
)

registry = Path(sys.argv[1])
ready = Path(sys.argv[2])

with ProductionRegistryWriterLock(
    registry,
    timeout=2.0,
):
    ready.write_text(
        "ready",
        encoding="utf-8",
    )
    time.sleep(0.75)
"""

    ready = (
        tmp_path
        / "ready.txt"
    )

    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            holder,
            str(registry),
            str(ready),
        ],
        cwd=Path.cwd(),
    )

    try:
        deadline = (
            time.monotonic()
            + 3.0
        )

        while (
            not ready.exists()
            and time.monotonic()
            < deadline
        ):
            time.sleep(
                0.02
            )

        assert ready.exists()

        started = (
            time.monotonic()
        )

        with ProductionRegistryWriterLock(
            registry,
            timeout=2.0,
        ):
            entered = (
                time.monotonic()
            )

        assert (
            entered
            - started
            >= 0.45
        )

    finally:
        process.wait(
            timeout=5.0
        )

    assert process.returncode == 0


def test_same_registry_process_lock_times_out_fail_closed(
    tmp_path: Path,
) -> None:
    registry = (
        tmp_path
        / "registry"
    )

    holder = r"""
import sys
import time
from pathlib import Path

from lrp.production.production_registry_lock import (
    ProductionRegistryWriterLock,
)

registry = Path(sys.argv[1])
ready = Path(sys.argv[2])

with ProductionRegistryWriterLock(
    registry,
    timeout=2.0,
):
    ready.write_text(
        "ready",
        encoding="utf-8",
    )
    time.sleep(1.0)
"""

    contender = r"""
import sys
from pathlib import Path

from lrp.production.production_registry_lock import (
    ProductionRegistryLockTimeout,
    ProductionRegistryWriterLock,
)

registry = Path(sys.argv[1])

try:
    with ProductionRegistryWriterLock(
        registry,
        timeout=0.10,
    ):
        pass

except ProductionRegistryLockTimeout:
    raise SystemExit(23)

raise SystemExit(0)
"""

    ready = (
        tmp_path
        / "ready.txt"
    )

    first = subprocess.Popen(
        [
            sys.executable,
            "-c",
            holder,
            str(registry),
            str(ready),
        ],
        cwd=Path.cwd(),
    )

    try:
        deadline = (
            time.monotonic()
            + 3.0
        )

        while (
            not ready.exists()
            and time.monotonic()
            < deadline
        ):
            time.sleep(
                0.02
            )

        assert ready.exists()

        second = subprocess.run(
            [
                sys.executable,
                "-c",
                contender,
                str(registry),
            ],
            cwd=Path.cwd(),
            check=False,
        )

        assert second.returncode == 23

    finally:
        first.wait(
            timeout=5.0
        )

    assert first.returncode == 0


def test_different_registries_are_independent_across_processes(
    tmp_path: Path,
) -> None:
    registry_a = (
        tmp_path
        / "registry-a"
    )

    registry_b = (
        tmp_path
        / "registry-b"
    )

    holder = r"""
import sys
import time
from pathlib import Path

from lrp.production.production_registry_lock import (
    ProductionRegistryWriterLock,
)

registry = Path(sys.argv[1])
ready = Path(sys.argv[2])

with ProductionRegistryWriterLock(
    registry,
    timeout=2.0,
):
    ready.write_text(
        "ready",
        encoding="utf-8",
    )
    time.sleep(0.8)
"""

    ready = (
        tmp_path
        / "ready.txt"
    )

    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            holder,
            str(registry_a),
            str(ready),
        ],
        cwd=Path.cwd(),
    )

    try:
        deadline = (
            time.monotonic()
            + 3.0
        )

        while (
            not ready.exists()
            and time.monotonic()
            < deadline
        ):
            time.sleep(
                0.02
            )

        assert ready.exists()

        started = (
            time.monotonic()
        )

        with ProductionRegistryWriterLock(
            registry_b,
            timeout=0.5,
        ):
            elapsed = (
                time.monotonic()
                - started
            )

        assert elapsed < 0.35

    finally:
        process.wait(
            timeout=5.0
        )

    assert process.returncode == 0


def test_process_termination_releases_os_lock(
    tmp_path: Path,
) -> None:
    registry = (
        tmp_path
        / "registry"
    )

    holder = r"""
import sys
import time
from pathlib import Path

from lrp.production.production_registry_lock import (
    ProductionRegistryWriterLock,
)

registry = Path(sys.argv[1])
ready = Path(sys.argv[2])

lock = ProductionRegistryWriterLock(
    registry,
    timeout=2.0,
)

lock.acquire()

ready.write_text(
    "ready",
    encoding="utf-8",
)

time.sleep(30.0)
"""

    ready = (
        tmp_path
        / "ready.txt"
    )

    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            holder,
            str(registry),
            str(ready),
        ],
        cwd=Path.cwd(),
    )

    deadline = (
        time.monotonic()
        + 3.0
    )

    while (
        not ready.exists()
        and time.monotonic()
        < deadline
    ):
        time.sleep(
            0.02
        )

    assert ready.exists()

    process.terminate()

    process.wait(
        timeout=5.0
    )

    with ProductionRegistryWriterLock(
        registry,
        timeout=1.0,
    ):
        pass


def test_publish_vs_publish_serializes_critical_sections(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry = (
        tmp_path
        / "registry"
    )

    first_source = _write_decision(
        tmp_path
        / "a.json",
        model="model-a",
    )

    second_source = _write_decision(
        tmp_path
        / "b.json",
        model="model-b",
    )

    original = (
        ProductionChampionRegistryPublisher
        ._write_publication_record
    )

    entered = (
        threading.Event()
    )

    release = (
        threading.Event()
    )

    counter = 0
    guard = threading.Lock()

    def delayed(
        *args,
        **kwargs,
    ):
        nonlocal counter

        with guard:
            counter += 1
            number = counter

        if number == 1:
            entered.set()

            assert release.wait(
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
            delayed
        ),
    )

    errors = []

    def publish(
        source: Path,
    ) -> None:
        try:
            (
                ProductionChampionRegistryPublisher()
                .publish(
                    source_decision=source,
                    registry_root=registry,
                )
            )

        except BaseException as exc:
            errors.append(
                exc
            )

    first = threading.Thread(
        target=publish,
        args=(
            first_source,
        ),
    )

    second = threading.Thread(
        target=publish,
        args=(
            second_source,
        ),
    )

    first.start()

    assert entered.wait(
        timeout=3.0
    )

    second.start()

    time.sleep(
        0.15
    )

    assert second.is_alive()

    release.set()

    first.join(
        timeout=3.0
    )

    second.join(
        timeout=3.0
    )

    assert errors == []
    assert not first.is_alive()
    assert not second.is_alive()


def test_publish_vs_rollback_serializes_same_registry(
    tmp_path: Path,
    monkeypatch,
) -> None:
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

    revision = _revision_id(
        registry,
        source_sha256=(
            first.source_sha256
        ),
    )

    service = ChampionRollbackService(
        registry_root=registry,
    )

    plan = service.plan(
        revision
    )

    original = (
        ProductionChampionRegistryPublisher
        ._write_publication_record
    )

    entered = threading.Event()
    release = threading.Event()

    def delayed(
        *args,
        **kwargs,
    ):
        entered.set()

        assert release.wait(
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
            delayed
        ),
    )

    publish_error = []
    rollback_error = []

    third_source = _write_decision(
        tmp_path
        / "c.json",
        model="model-c",
    )

    def run_publish() -> None:
        try:
            publisher.publish(
                source_decision=third_source,
                registry_root=registry,
            )

        except BaseException as exc:
            publish_error.append(
                exc
            )

    def run_rollback() -> None:
        try:
            service.execute(
                plan
            )

        except BaseException as exc:
            rollback_error.append(
                exc
            )

    publish_thread = threading.Thread(
        target=run_publish
    )

    rollback_thread = threading.Thread(
        target=run_rollback
    )

    publish_thread.start()

    assert entered.wait(
        timeout=3.0
    )

    rollback_thread.start()

    time.sleep(
        0.15
    )

    assert rollback_thread.is_alive()

    release.set()

    publish_thread.join(
        timeout=3.0
    )

    rollback_thread.join(
        timeout=3.0
    )

    assert publish_error == []

    # Rollback may become stale after the
    # serialized publication completes.
    # That is correct optimistic behavior.
    assert len(
        rollback_error
    ) <= 1

    if rollback_error:
        assert isinstance(
            rollback_error[0],
            ValueError,
        )

    assert not publish_thread.is_alive()
    assert not rollback_thread.is_alive()


def test_rollback_vs_rollback_serializes_and_revalidates(
    tmp_path: Path,
    monkeypatch,
) -> None:
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

    revision = _revision_id(
        registry,
        source_sha256=(
            first.source_sha256
        ),
    )

    service_a = ChampionRollbackService(
        registry_root=registry,
    )

    service_b = ChampionRollbackService(
        registry_root=registry,
    )

    plan_a = service_a.plan(
        revision
    )

    plan_b = service_b.plan(
        revision
    )

    original = (
        service_a
        ._write_rollback_provenance
    )

    entered = threading.Event()
    release = threading.Event()

    def delayed(
        *args,
        **kwargs,
    ):
        entered.set()

        assert release.wait(
            timeout=3.0
        )

        return original(
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        service_a,
        "_write_rollback_provenance",
        delayed,
    )

    first_error = []
    second_error = []

    def run_first() -> None:
        try:
            service_a.execute(
                plan_a
            )

        except BaseException as exc:
            first_error.append(
                exc
            )

    def run_second() -> None:
        try:
            service_b.execute(
                plan_b
            )

        except BaseException as exc:
            second_error.append(
                exc
            )

    first_thread = threading.Thread(
        target=run_first
    )

    second_thread = threading.Thread(
        target=run_second
    )

    first_thread.start()

    assert entered.wait(
        timeout=3.0
    )

    second_thread.start()

    time.sleep(
        0.15
    )

    assert second_thread.is_alive()

    release.set()

    first_thread.join(
        timeout=3.0
    )

    second_thread.join(
        timeout=3.0
    )

    assert first_error == []

    assert len(
        second_error
    ) == 1

    assert isinstance(
        second_error[0],
        ValueError,
    )

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
