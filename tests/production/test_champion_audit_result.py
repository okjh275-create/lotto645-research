from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from lrp.production import (
    ProductionChampionAuditCheck,
    ProductionChampionAuditIssue,
    ProductionChampionAuditResult,
)


def test_audit_check_contract() -> None:
    check = ProductionChampionAuditCheck(
        name="active_decision",
        status="PASS",
        detail="active decision is readable",
    )

    assert check.name == "active_decision"
    assert check.status == "PASS"
    assert (
        check.detail
        == "active decision is readable"
    )


def test_audit_issue_contract() -> None:
    issue = ProductionChampionAuditIssue(
        code="baseline_fallback",
        severity="WARN",
        message=(
            "production resolved to baseline"
        ),
    )

    assert issue.code == "baseline_fallback"
    assert issue.severity == "WARN"
    assert (
        issue.message
        == "production resolved to baseline"
    )


def test_audit_result_contract() -> None:
    result = ProductionChampionAuditResult(
        status="WARN",
        selected_model=None,
        resolved_model="baseline",
        fallback_applied=True,
        fallback_reason="no_selected_model",
        checks=(
            ProductionChampionAuditCheck(
                name="registry",
                status="PASS",
                detail="registry is readable",
            ),
        ),
        issues=(
            ProductionChampionAuditIssue(
                code="baseline_fallback",
                severity="WARN",
                message=(
                    "production resolved to baseline"
                ),
            ),
        ),
    )

    assert result.status == "WARN"
    assert result.selected_model is None
    assert result.resolved_model == "baseline"
    assert result.fallback_applied is True

    assert (
        result.fallback_reason
        == "no_selected_model"
    )

    assert len(result.checks) == 1
    assert len(result.issues) == 1


def test_audit_result_as_dict_is_deterministic() -> None:
    result = ProductionChampionAuditResult(
        status="WARN",
        selected_model=None,
        resolved_model="baseline",
        fallback_applied=True,
        fallback_reason="no_selected_model",
        checks=(
            ProductionChampionAuditCheck(
                name="registry",
                status="PASS",
                detail="registry is readable",
            ),
            ProductionChampionAuditCheck(
                name="activation",
                status="PASS",
                detail=(
                    "activation resolved "
                    "deterministically"
                ),
            ),
        ),
        issues=(
            ProductionChampionAuditIssue(
                code="baseline_fallback",
                severity="WARN",
                message=(
                    "production resolved to baseline"
                ),
            ),
        ),
    )

    assert result.as_dict() == {
        "status": "WARN",
        "selected_model": None,
        "resolved_model": "baseline",
        "fallback_applied": True,
        "fallback_reason": (
            "no_selected_model"
        ),
        "checks": [
            {
                "name": "registry",
                "status": "PASS",
                "detail": (
                    "registry is readable"
                ),
            },
            {
                "name": "activation",
                "status": "PASS",
                "detail": (
                    "activation resolved "
                    "deterministically"
                ),
            },
        ],
        "issues": [
            {
                "code": "baseline_fallback",
                "severity": "WARN",
                "message": (
                    "production resolved to baseline"
                ),
            },
        ],
    }


@pytest.mark.parametrize(
    "status",
    (
        "",
        "pass",
        "OK",
        "ERROR",
    ),
)
def test_audit_check_rejects_invalid_status(
    status: str,
) -> None:
    with pytest.raises(
        ValueError,
    ):
        ProductionChampionAuditCheck(
            name="registry",
            status=status,
            detail="test",
        )


@pytest.mark.parametrize(
    "severity",
    (
        "",
        "INFO",
        "ERROR",
        "warn",
    ),
)
def test_audit_issue_rejects_invalid_severity(
    severity: str,
) -> None:
    with pytest.raises(
        ValueError,
    ):
        ProductionChampionAuditIssue(
            code="test",
            severity=severity,
            message="test",
        )


@pytest.mark.parametrize(
    "status",
    (
        "",
        "OK",
        "ERROR",
        "warn",
    ),
)
def test_audit_result_rejects_invalid_status(
    status: str,
) -> None:
    with pytest.raises(
        ValueError,
    ):
        ProductionChampionAuditResult(
            status=status,
            selected_model=None,
            resolved_model="baseline",
            fallback_applied=False,
            fallback_reason=None,
            checks=(),
            issues=(),
        )


def test_check_name_must_not_be_empty() -> None:
    with pytest.raises(
        ValueError,
    ):
        ProductionChampionAuditCheck(
            name="",
            status="PASS",
            detail="test",
        )


def test_issue_code_must_not_be_empty() -> None:
    with pytest.raises(
        ValueError,
    ):
        ProductionChampionAuditIssue(
            code="",
            severity="FAIL",
            message="test",
        )


def test_result_requires_resolved_model() -> None:
    with pytest.raises(
        ValueError,
    ):
        ProductionChampionAuditResult(
            status="FAIL",
            selected_model=None,
            resolved_model="",
            fallback_applied=False,
            fallback_reason=None,
            checks=(),
            issues=(),
        )


def test_fallback_reason_required_when_applied() -> None:
    with pytest.raises(
        ValueError,
    ):
        ProductionChampionAuditResult(
            status="WARN",
            selected_model=None,
            resolved_model="baseline",
            fallback_applied=True,
            fallback_reason=None,
            checks=(),
            issues=(),
        )


def test_fallback_reason_must_be_none_when_not_applied() -> None:
    with pytest.raises(
        ValueError,
    ):
        ProductionChampionAuditResult(
            status="PASS",
            selected_model="combined",
            resolved_model="combined",
            fallback_applied=False,
            fallback_reason=(
                "unexpected_reason"
            ),
            checks=(),
            issues=(),
        )


def test_contract_objects_are_immutable() -> None:
    check = ProductionChampionAuditCheck(
        name="registry",
        status="PASS",
        detail="ok",
    )

    issue = ProductionChampionAuditIssue(
        code="test",
        severity="WARN",
        message="warning",
    )

    result = ProductionChampionAuditResult(
        status="WARN",
        selected_model=None,
        resolved_model="baseline",
        fallback_applied=True,
        fallback_reason="no_selected_model",
        checks=(check,),
        issues=(issue,),
    )

    with pytest.raises(
        FrozenInstanceError,
    ):
        check.status = "FAIL"

    with pytest.raises(
        FrozenInstanceError,
    ):
        issue.severity = "FAIL"

    with pytest.raises(
        FrozenInstanceError,
    ):
        result.status = "PASS"