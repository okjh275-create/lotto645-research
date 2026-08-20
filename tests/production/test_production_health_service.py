from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest


DOMAINS = [
    "active_champion",
    "writer_lock",
    "recovery_readiness",
    "history_safety",
    "lifecycle_readiness",
]


def _api():
    module = importlib.import_module(
        "lrp.production.production_health"
    )

    return module.ProductionHealthService


def _write_registry(
    root: Path,
    *,
    selected_model: str | None = "baseline",
) -> Path:
    registry = root / "registry"

    active = registry / "active"

    active.mkdir(
        parents=True,
        exist_ok=True,
    )

    decision_raw = (
        json.dumps(
            {
                "selection": {
                    "selected_model":
                        selected_model,
                },
            },
            indent=2,
        )
        + "\n"
    ).encode("utf-8")

    decision_path = (
        active
        / "champion_decision.json"
    )

    decision_path.write_bytes(
        decision_raw
    )

    decision_sha = hashlib.sha256(
        decision_raw
    ).hexdigest()

    source_path = (
        root
        / "source.json"
    )

    source_path.write_bytes(
        decision_raw
    )

    (
        active
        / "publication.json"
    ).write_text(
        json.dumps(
            {
                "source_path":
                    str(source_path),

                "source_sha256":
                    decision_sha,

                "published_path":
                    str(decision_path),

                "published_at_kst":
                    "2026-08-20T00:00:00+09:00",

                "selected_model":
                    selected_model,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return registry


def _snapshot(
    tmp_path: Path,
    *,
    selected_model: str | None = "baseline",
):
    Service = _api()

    registry = _write_registry(
        tmp_path,
        selected_model=selected_model,
    )

    return Service(
        registry_root=registry,
        snapshot_root=(
            tmp_path
            / "snapshots"
        ),
    ).snapshot()


def test_health_service_pass_snapshot(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(
        tmp_path
    )

    assert snapshot.status == "PASS"


def test_health_service_warn_from_champion_audit(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(
        tmp_path,
        selected_model=None,
    )

    active = snapshot.domains[
        "active_champion"
    ]

    assert active["status"] == "WARN"

    assert (
        snapshot.status
        == "WARN"
    )


def test_health_service_fail_from_champion_audit(
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

    snapshot = Service(
        registry_root=registry,
        snapshot_root=(
            tmp_path
            / "snapshots"
        ),
    ).snapshot()

    assert (
        snapshot.domains[
            "active_champion"
        ]["status"]
        == "FAIL"
    )

    assert snapshot.status == "FAIL"


def test_health_service_pass_when_writer_lock_metadata_is_residual(
    tmp_path: Path,
) -> None:
    Service = _api()

    registry = _write_registry(
        tmp_path
    )

    metadata = (
        json.dumps(
            {
                "pid": 12345,
                "acquired_at":
                    "2026-08-20T00:00:00+00:00",
            }
        )
        + "\n"
    ).encode("utf-8")

    (
        registry
        / ".writer.lock"
    ).write_bytes(
        b"\x00"
        + metadata
    )

    snapshot = Service(
        registry_root=registry,
        snapshot_root=(
            tmp_path
            / "snapshots"
        ),
    ).snapshot()

    domain = snapshot.domains[
        "writer_lock"
    ]

    assert domain["status"] == "PASS"
    assert domain["present"] is True
    assert domain["metadata_valid"] is True

    assert not any(
        issue.domain == "writer_lock"
        for issue in snapshot.issues
    )


def test_health_service_warn_when_writer_lock_malformed(
    tmp_path: Path,
) -> None:
    Service = _api()

    registry = _write_registry(
        tmp_path
    )

    (
        registry
        / ".writer.lock"
    ).write_text(
        "not-json",
        encoding="utf-8",
    )

    snapshot = Service(
        registry_root=registry,
        snapshot_root=(
            tmp_path
            / "snapshots"
        ),
    ).snapshot()

    assert (
        snapshot.domains[
            "writer_lock"
        ]["status"]
        == "WARN"
    )

    assert any(
        issue.code
        == "WRITER_LOCK_MALFORMED"
        for issue in snapshot.issues
    )


def test_health_service_reports_recovery_readiness(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(
        tmp_path
    )

    domain = snapshot.domains[
        "recovery_readiness"
    ]

    assert domain["status"] == "PASS"
    assert domain["backup"] is True
    assert domain["restore"] is True
    assert domain["atomicity"] is True


def test_health_service_reports_history_safety(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(
        tmp_path
    )

    domain = snapshot.domains[
        "history_safety"
    ]

    assert domain["status"] == "PASS"
    assert domain["retention"] is True
    assert domain["rollback"] is True


def test_health_service_reports_lifecycle_readiness(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(
        tmp_path
    )

    domain = snapshot.domains[
        "lifecycle_readiness"
    ]

    assert domain["status"] == "PASS"
    assert domain["service"] is True


def test_health_service_aggregate_warn(
    tmp_path: Path,
) -> None:
    Service = _api()

    registry = _write_registry(
        tmp_path
    )

    (
        registry
        / ".writer.lock"
    ).write_text(
        "not-json",
        encoding="utf-8",
    )

    snapshot = Service(
        registry_root=registry,
        snapshot_root=(
            tmp_path
            / "snapshots"
        ),
    ).snapshot()

    assert (
        snapshot.domains[
            "writer_lock"
        ]["status"]
        == "WARN"
    )

    assert snapshot.status == "WARN"


def test_health_service_aggregate_fail(
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

    snapshot = Service(
        registry_root=registry,
        snapshot_root=(
            tmp_path
            / "snapshots"
        ),
    ).snapshot()

    assert snapshot.status == "FAIL"


def test_health_snapshot_exposes_active_identity(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(
        tmp_path
    )

    assert snapshot.active_model == "baseline"

    assert (
        snapshot.active_revision_id
        is None
    )


def test_health_snapshot_domains_are_exact_and_ordered(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(
        tmp_path
    )

    assert list(
        snapshot.domains
    ) == DOMAINS

    assert [
        check.domain
        for check in snapshot.checks
    ] == DOMAINS
