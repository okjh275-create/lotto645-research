from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest


def _api():
    from lrp.production.production_reconciliation import (
        ProductionReconciliationCheck,
        ProductionReconciliationIssue,
        ProductionReconciliationResult,
    )

    return (
        ProductionReconciliationCheck,
        ProductionReconciliationIssue,
        ProductionReconciliationResult,
    )


def test_reconciliation_check_contract() -> None:
    Check, _, _ = _api()

    check = Check(
        name="active_pair",
        domain="active_pair",
        status="PASS",
        detail="coherent active pair",
    )

    assert check.name == "active_pair"
    assert check.domain == "active_pair"
    assert check.status == "PASS"
    assert check.detail == "coherent active pair"

    with pytest.raises(
        FrozenInstanceError
    ):
        check.status = "FAIL"


def test_reconciliation_issue_contract() -> None:
    _, Issue, _ = _api()

    issue = Issue(
        code="ORPHAN_REVISION",
        domain="revision_history",
        severity="WARN",
        message="orphan revision",
        path="history/aaaaaaaa.json",
    )

    assert issue.code == "ORPHAN_REVISION"
    assert issue.domain == "revision_history"
    assert issue.severity == "WARN"
    assert issue.path == "history/aaaaaaaa.json"


def test_reconciliation_result_contract() -> None:
    Check, Issue, Result = _api()

    check = Check(
        name="active_pair",
        domain="active_pair",
        status="PASS",
        detail="coherent",
    )

    issue = Issue(
        code="ORPHAN_REVISION",
        domain="revision_history",
        severity="WARN",
        message="orphan",
        path="history/orphan.json",
    )

    result = Result(
        schema_version=1,
        generated_at="2026-08-20T00:00:00+09:00",
        status="WARN",
        checks=(check,),
        issues=(issue,),
        active_model="baseline",
        active_source_sha256="a" * 64,
        active_revision_id="b" * 64,
        domains={
            "active_pair": {
                "status": "PASS",
            },
        },
    )

    assert result.schema_version == 1
    assert result.status == "WARN"
    assert result.active_model == "baseline"


def test_reconciliation_payload_is_deterministic() -> None:
    Check, Issue, Result = _api()

    check = Check(
        name="active_pair",
        domain="active_pair",
        status="PASS",
        detail="coherent",
    )

    issue = Issue(
        code="ORPHAN_REVISION",
        domain="revision_history",
        severity="WARN",
        message="orphan",
        path="history/orphan.json",
    )

    result = Result(
        schema_version=1,
        generated_at="2026-08-20T00:00:00+09:00",
        status="WARN",
        checks=(check,),
        issues=(issue,),
        active_model="baseline",
        active_source_sha256="a" * 64,
        active_revision_id="b" * 64,
        domains={
            "active_pair": {
                "status": "PASS",
            },
        },
    )

    first = result.to_payload()
    second = result.to_payload()

    assert first == second
    assert list(first) == [
        "schema_version",
        "generated_at",
        "status",
        "checks",
        "issues",
        "active_model",
        "active_source_sha256",
        "active_revision_id",
        "domains",
    ]
