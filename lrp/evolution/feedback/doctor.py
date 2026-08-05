"""Adaptive automation doctor service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lrp.evolution.feedback.profile_integrity import (
    AdaptiveProfileIntegrityDoctor,
    AdaptiveProfileIntegrityReport,
)
from lrp.evolution.feedback.repository import (
    AdaptiveAutomationRepository,
)
from lrp.evolution.feedback.repository_status_analyzer import (
    AdaptiveRepositoryStatusAnalyzer,
)
from lrp.evolution.feedback.status import (
    AdaptiveStatusIssue,
    AdaptiveStatusReport,
)


@dataclass(frozen=True, slots=True)
class AdaptiveAutomationDoctorReport:

    repository: AdaptiveStatusReport

    profile: AdaptiveProfileIntegrityReport

    @property
    def overall_ok(self) -> bool:
        return (
            self.repository.overall_ok
            and self.profile.overall_ok
        )

    @property
    def latest_revision(self) -> int:
        return self.repository.latest_revision

    @property
    def rollback_count(self) -> int:
        return self.repository.rollback_count

    @property
    def recommendation_count(self) -> int:
        return self.repository.recommendation_count

    @property
    def issues(self) -> tuple[
        AdaptiveStatusIssue,
        ...
    ]:
        return (
            self.repository.issues
            + self.profile.issues
        )

    @property
    def error_count(self) -> int:
        return sum(
            issue.severity == "error"
            for issue in self.issues
        )

    @property
    def warning_count(self) -> int:
        return sum(
            issue.severity == "warning"
            for issue in self.issues
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "overall_ok":
                self.overall_ok,
            "latest_revision":
                self.latest_revision,
            "rollback_count":
                self.rollback_count,
            "recommendation_count":
                self.recommendation_count,
            "error_count":
                self.error_count,
            "warning_count":
                self.warning_count,
            "repository":
                self.repository.as_dict(),
            "profile":
                self.profile.as_dict(),
            "issues": [
                item.as_dict()
                for item in self.issues
            ],
        }


class AdaptiveAutomationDoctor:

    def __init__(
        self,
    ) -> None:

        self._repository = (
            AdaptiveRepositoryStatusAnalyzer()
        )

        self._profile = (
            AdaptiveProfileIntegrityDoctor()
        )

    def inspect(
        self,
        repository: AdaptiveAutomationRepository,
    ) -> AdaptiveAutomationDoctorReport:

        repository_result = (
            self._repository.analyze(
                repository
            )
        )

        profile_result = (
            self._profile.inspect(
                repository
            )
        )

        return (
            AdaptiveAutomationDoctorReport(
                repository=
                repository_result,
                profile=
                profile_result,
            )
        )
