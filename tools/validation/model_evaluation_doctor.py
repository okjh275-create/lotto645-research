"""Operational doctor for historical model-evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tools.validation.model_evaluation_run_audit import (
    ModelEvaluationRunAuditResult,
)
from tools.validation.model_evaluation_run_audit_service import (
    ModelEvaluationRunAuditReport,
    ModelEvaluationRunAuditService,
)


@dataclass(frozen=True, slots=True)
class ModelEvaluationDoctorResult:
    """Operational readiness result for model-evaluation artifacts."""

    audit_report: ModelEvaluationRunAuditReport
    issues: tuple[str, ...]

    @property
    def overall_ok(self) -> bool:
        return not self.issues

    @property
    def status(self) -> str:
        return (
            "PASS"
            if self.overall_ok
            else "FAIL"
        )

    @property
    def total_count(self) -> int:
        return self.audit_report.total_count

    @property
    def pass_count(self) -> int:
        return self.audit_report.pass_count

    @property
    def fail_count(self) -> int:
        return self.audit_report.fail_count

    @property
    def incomplete_count(self) -> int:
        return self.audit_report.incomplete_count

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "overall_ok": self.overall_ok,
            "total_count": self.total_count,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "incomplete_count": (
                self.incomplete_count
            ),
            "issues": list(
                self.issues
            ),
            "runs": [
                {
                    "run_id": result.run_id,
                    "status": result.status,
                    "issues": list(
                        result.issues
                    ),
                }
                for result in (
                    self.audit_report.results
                )
            ],
        }


class ModelEvaluationDoctor:
    """Inspect Project M evaluation artifacts for operational readiness."""

    def __init__(
        self,
        *,
        audit_service: ModelEvaluationRunAuditService | None = None,
    ) -> None:
        self._audit_service = (
            audit_service
            if audit_service is not None
            else ModelEvaluationRunAuditService()
        )

    def inspect(
        self,
        root: str | Path,
    ) -> ModelEvaluationDoctorResult:
        root = Path(root)

        if not root.exists():
            raise FileNotFoundError(
                root
            )

        if not root.is_dir():
            raise NotADirectoryError(
                root
            )

        report = self._audit_service.run(
            root
        )

        issues = self._issues(
            report
        )

        return ModelEvaluationDoctorResult(
            audit_report=report,
            issues=issues,
        )

    @staticmethod
    def _issues(
        report: ModelEvaluationRunAuditReport,
    ) -> tuple[str, ...]:
        if report.total_count == 0:
            return (
                "no_evaluation_runs",
            )

        issues: list[str] = []

        for result in report.results:
            issue = (
                ModelEvaluationDoctor
                ._result_issue(
                    result
                )
            )

            if issue is not None:
                issues.append(issue)

        return tuple(issues)

    @staticmethod
    def _result_issue(
        result: ModelEvaluationRunAuditResult,
    ) -> str | None:
        if result.status == "PASS":
            return None

        if result.status == "FAIL":
            return (
                "run_failed:"
                f"{result.run_id}"
            )

        if result.status == "INCOMPLETE":
            return (
                "run_incomplete:"
                f"{result.run_id}"
            )

        return (
            "run_unknown_status:"
            f"{result.run_id}"
        )
