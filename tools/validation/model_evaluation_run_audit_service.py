"""Batch discovery and audit service for model-evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tools.validation.model_evaluation_run_audit import (
    ModelEvaluationRunAudit,
    ModelEvaluationRunAuditResult,
)
from tools.validation.model_evaluation_run_discovery import (
    ModelEvaluationRunDiscovery,
)


@dataclass(frozen=True, slots=True)
class ModelEvaluationRunAuditReport:
    """Aggregate audit report for discovered evaluation runs."""

    results: tuple[
        ModelEvaluationRunAuditResult,
        ...,
    ]

    @property
    def total_count(self) -> int:
        return len(
            self.results
        )

    @property
    def pass_count(self) -> int:
        return sum(
            result.status == "PASS"
            for result in self.results
        )

    @property
    def fail_count(self) -> int:
        return sum(
            result.status == "FAIL"
            for result in self.results
        )

    @property
    def incomplete_count(self) -> int:
        return sum(
            result.status == "INCOMPLETE"
            for result in self.results
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "summary": {
                "total_count": (
                    self.total_count
                ),
                "pass_count": (
                    self.pass_count
                ),
                "fail_count": (
                    self.fail_count
                ),
                "incomplete_count": (
                    self.incomplete_count
                ),
            },
            "results": [
                result.as_dict()
                for result in self.results
            ],
        }


class ModelEvaluationRunAuditService:
    """Discover and audit all model-evaluation runs below a root."""

    def __init__(
        self,
        *,
        discovery: ModelEvaluationRunDiscovery | None = None,
        auditor: ModelEvaluationRunAudit | None = None,
    ) -> None:
        self._discovery = (
            discovery
            if discovery is not None
            else ModelEvaluationRunDiscovery()
        )

        self._auditor = (
            auditor
            if auditor is not None
            else ModelEvaluationRunAudit()
        )

    def run(
        self,
        root: str | Path,
    ) -> ModelEvaluationRunAuditReport:
        root = Path(root)

        if not root.exists():
            raise FileNotFoundError(
                root
            )

        if not root.is_dir():
            raise NotADirectoryError(
                root
            )

        discovered = (
            self._discovery.discover(
                root
            )
        )

        results = tuple(
            self._auditor.audit(
                record.root
                / "evaluation_run.json"
            )
            for record in discovered
        )

        return ModelEvaluationRunAuditReport(
            results=results
        )
