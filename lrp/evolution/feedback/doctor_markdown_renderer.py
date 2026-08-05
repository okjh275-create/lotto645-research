"""Render adaptive automation doctor reports as Markdown."""

from __future__ import annotations

from lrp.evolution.feedback.doctor import (
    AdaptiveAutomationDoctorReport,
)
from lrp.evolution.feedback.status import (
    AdaptiveStatusIssue,
)


class AdaptiveAutomationDoctorMarkdownRenderer:
    """Render deterministic human-readable doctor reports."""

    def render(
        self,
        report: AdaptiveAutomationDoctorReport,
    ) -> str:
        if not isinstance(
            report,
            AdaptiveAutomationDoctorReport,
        ):
            raise TypeError(
                "report must be an "
                "AdaptiveAutomationDoctorReport"
            )

        lines: list[str] = [
            "# Adaptive Automation Doctor Report",
            "",
            "## Summary",
            "",
            (
                "- Overall status: "
                f"{self._status(report.overall_ok)}"
            ),
            (
                "- Latest revision: "
                f"{report.latest_revision}"
            ),
            (
                "- Rollback count: "
                f"{report.rollback_count}"
            ),
            (
                "- Recommendation count: "
                f"{report.recommendation_count}"
            ),
            (
                "- Error count: "
                f"{report.error_count}"
            ),
            (
                "- Warning count: "
                f"{report.warning_count}"
            ),
            "",
            "## Repository Status",
            "",
            (
                "- Overall: "
                f"{self._status(report.repository.overall_ok)}"
            ),
            (
                "- Repository: "
                f"{self._status(report.repository.repository_ok)}"
            ),
            (
                "- Profile: "
                f"{self._status(report.repository.profile_ok)}"
            ),
            (
                "- Validation: "
                f"{self._status(report.repository.validation_ok)}"
            ),
            (
                "- Latest revision: "
                f"{report.repository.latest_revision}"
            ),
            (
                "- Rollback count: "
                f"{report.repository.rollback_count}"
            ),
            (
                "- Recommendation count: "
                f"{report.repository.recommendation_count}"
            ),
            "",
            "## Profile Integrity",
            "",
            (
                "- Overall: "
                f"{self._status(report.profile.overall_ok)}"
            ),
            (
                "- Revision count: "
                f"{report.profile.revision_count}"
            ),
            (
                "- First revision: "
                f"{self._optional_integer(report.profile.first_revision)}"
            ),
            (
                "- Latest revision: "
                f"{self._optional_integer(report.profile.latest_revision)}"
            ),
            (
                "- Rollback count: "
                f"{report.profile.rollback_count}"
            ),
            (
                "- Duplicate profile count: "
                f"{report.profile.duplicate_profile_count}"
            ),
            "",
            "## Issues",
            "",
        ]

        issues = self._sorted_issues(
            report.issues
        )

        if issues:
            lines.extend(
                [
                    "| Severity | Category | Message |",
                    "|---|---|---|",
                ]
            )

            for issue in issues:
                lines.append(
                    "| "
                    f"{self._escape(issue.severity)} | "
                    f"{self._escape(issue.category)} | "
                    f"{self._escape(issue.message)} |"
                )
        else:
            lines.append(
                "No issues detected."
            )

        lines.extend(
            [
                "",
                "## Conclusion",
                "",
                (
                    "PASS"
                    if report.overall_ok
                    else "FAIL"
                ),
                "",
            ]
        )

        return "\n".join(lines)

    @staticmethod
    def _status(
        value: bool,
    ) -> str:
        if not isinstance(value, bool):
            raise TypeError(
                "status value must be boolean"
            )

        return "PASS" if value else "FAIL"

    @staticmethod
    def _optional_integer(
        value: int | None,
    ) -> str:
        if value is None:
            return "-"

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
        ):
            raise TypeError(
                "optional revision must be "
                "an integer or None"
            )

        return str(value)

    @staticmethod
    def _sorted_issues(
        issues: tuple[
            AdaptiveStatusIssue,
            ...,
        ],
    ) -> tuple[
        AdaptiveStatusIssue,
        ...,
    ]:
        severity_order = {
            "error": 0,
            "warning": 1,
            "info": 2,
        }

        return tuple(
            sorted(
                issues,
                key=lambda issue: (
                    severity_order.get(
                        issue.severity,
                        99,
                    ),
                    issue.category,
                    issue.message,
                ),
            )
        )

    @staticmethod
    def _escape(
        value: str,
    ) -> str:
        if not isinstance(value, str):
            raise TypeError(
                "Markdown table value must "
                "be a string"
            )

        return (
            value
            .replace("\\", "\\\\")
            .replace("|", "\\|")
            .replace("\r", " ")
            .replace("\n", " ")
        )
