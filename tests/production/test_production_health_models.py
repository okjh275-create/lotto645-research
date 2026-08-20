from __future__ import annotations

import importlib
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest


def _api():
    module = importlib.import_module(
        "lrp.production.production_health"
    )

    return (
        module.ProductionHealthCheck,
        module.ProductionHealthIssue,
        module.ProductionHealthSnapshot,
    )


def test_health_check_contract() -> None:
    Check, _, _ = _api()

    check = Check(
        name="active_champion",
        domain="active_champion",
        status="PASS",
        detail="healthy",
    )

    assert check.name == "active_champion"
    assert check.domain == "active_champion"
    assert check.status == "PASS"
    assert check.detail == "healthy"

    with pytest.raises(
        FrozenInstanceError,
    ):
        check.status = "FAIL"


def test_health_issue_contract() -> None:
    _, Issue, _ = _api()

    issue = Issue(
        code="WRITER_LOCK_PRESENT",
        domain="writer_lock",
        severity="WARN",
        message="writer lock present",
    )

    assert issue.code == "WRITER_LOCK_PRESENT"
    assert issue.domain == "writer_lock"
    assert issue.severity == "WARN"
    assert issue.message == "writer lock present"

    with pytest.raises(
        FrozenInstanceError,
    ):
        issue.severity = "FAIL"


def test_health_snapshot_contract() -> None:
    Check, Issue, Snapshot = _api()

    snapshot = Snapshot(
        schema_version=1,
        generated_at=(
            datetime(
                2026,
                8,
                20,
                tzinfo=timezone.utc,
            ).isoformat()
        ),
        status="WARN",
        checks=(
            Check(
                name="writer_lock",
                domain="writer_lock",
                status="WARN",
                detail="present",
            ),
        ),
        issues=(
            Issue(
                code="WRITER_LOCK_PRESENT",
                domain="writer_lock",
                severity="WARN",
                message="present",
            ),
        ),
        active_model="baseline",
        active_revision_id="a" * 64,
        domains={
            "writer_lock": {
                "status": "WARN",
            },
        },
    )

    assert snapshot.schema_version == 1
    assert snapshot.status == "WARN"
    assert isinstance(
        snapshot.checks,
        tuple,
    )
    assert isinstance(
        snapshot.issues,
        tuple,
    )

    with pytest.raises(
        FrozenInstanceError,
    ):
        snapshot.status = "PASS"


def test_health_snapshot_payload_is_deterministic() -> None:
    Check, Issue, Snapshot = _api()

    snapshot = Snapshot(
        schema_version=1,
        generated_at="2026-08-20T00:00:00+00:00",
        status="WARN",
        checks=(
            Check(
                name="writer_lock",
                domain="writer_lock",
                status="WARN",
                detail="present",
            ),
        ),
        issues=(
            Issue(
                code="WRITER_LOCK_PRESENT",
                domain="writer_lock",
                severity="WARN",
                message="present",
            ),
        ),
        active_model="baseline",
        active_revision_id="b" * 64,
        domains={
            "writer_lock": {
                "status": "WARN",
            },
        },
    )

    first = snapshot.to_payload()
    second = snapshot.to_payload()

    assert first == second

    assert list(
        first
    ) == [
        "schema_version",
        "generated_at",
        "status",
        "checks",
        "issues",
        "active_model",
        "active_revision_id",
        "domains",
    ]
