from __future__ import annotations

from pathlib import Path

import pytest

from lrp.evolution.feedback import (
    AdaptiveAutomationDoctor,
    AdaptiveAutomationDoctorMarkdownRenderer,
    AdaptiveAutomationDoctorReport,
    AdaptiveAutomationRepository,
)
from lrp.evolution.feedback.doctor import (
    AdaptiveAutomationDoctorReport,
)
from lrp.evolution.feedback.profile_integrity import (
    AdaptiveProfileIntegrityReport,
)
from lrp.evolution.feedback.status import (
    AdaptiveStatusIssue,
    AdaptiveStatusReport,
)


def empty_report(
    tmp_path: Path,
) -> AdaptiveAutomationDoctorReport:
    return AdaptiveAutomationDoctor().inspect(
        AdaptiveAutomationRepository(
            tmp_path / "repository"
        )
    )


def report_with_issues() -> (
    AdaptiveAutomationDoctorReport
):
    repository = AdaptiveStatusReport(
        latest_revision=14,
        repository_ok=False,
        profile_ok=True,
        validation_ok=True,
        rollback_count=1,
        recommendation_count=3,
        issues=(
            AdaptiveStatusIssue(
                category="revision",
                severity="error",
                message="revision gap detected",
            ),
        ),
    )

    profile = AdaptiveProfileIntegrityReport(
        revision_count=3,
        first_revision=12,
        latest_revision=14,
        rollback_count=1,
        duplicate_profile_count=1,
        issues=(
            AdaptiveStatusIssue(
                category="profile_integrity",
                severity="warning",
                message=(
                    "duplicate | profile\nweights"
                ),
            ),
        ),
    )

    return AdaptiveAutomationDoctorReport(
        repository=repository,
        profile=profile,
    )


def test_render_contains_required_sections(
    tmp_path: Path,
) -> None:
    text = (
        AdaptiveAutomationDoctorMarkdownRenderer()
        .render(
            empty_report(tmp_path)
        )
    )

    assert text.startswith(
        "# Adaptive Automation Doctor Report\n"
    )
    assert "## Summary" in text
    assert "## Repository Status" in text
    assert "## Profile Integrity" in text
    assert "## Issues" in text
    assert "## Conclusion" in text


def test_empty_repository_renders_failure(
    tmp_path: Path,
) -> None:
    text = (
        AdaptiveAutomationDoctorMarkdownRenderer()
        .render(
            empty_report(tmp_path)
        )
    )

    assert "- Overall status: FAIL" in text
    assert "- Latest revision: 0" in text
    assert "- First revision: -" in text
    assert "- Latest revision: -" in text
    assert "No issues detected." in text
    assert text.endswith(
        "## Conclusion\n\nFAIL\n"
    )


def test_issue_table_is_rendered() -> None:
    text = (
        AdaptiveAutomationDoctorMarkdownRenderer()
        .render(
            report_with_issues()
        )
    )

    assert (
        "| Severity | Category | Message |"
        in text
    )
    assert (
        "| error | revision | "
        "revision gap detected |"
        in text
    )
    assert (
        "| warning | profile_integrity | "
        "duplicate \\| profile weights |"
        in text
    )


def test_issues_are_sorted_deterministically() -> None:
    report = report_with_issues()

    text = (
        AdaptiveAutomationDoctorMarkdownRenderer()
        .render(report)
    )

    error_position = text.index(
        "| error |"
    )
    warning_position = text.index(
        "| warning |"
    )

    assert error_position < warning_position


def test_render_is_deterministic() -> None:
    report = report_with_issues()
    renderer = (
        AdaptiveAutomationDoctorMarkdownRenderer()
    )

    first = renderer.render(report)
    second = renderer.render(report)

    assert first == second


def test_render_uses_unix_newlines() -> None:
    text = (
        AdaptiveAutomationDoctorMarkdownRenderer()
        .render(
            report_with_issues()
        )
    )

    assert "\r" not in text
    assert text.endswith("\n")


def test_summary_counts_are_rendered() -> None:
    text = (
        AdaptiveAutomationDoctorMarkdownRenderer()
        .render(
            report_with_issues()
        )
    )

    assert "- Latest revision: 14" in text
    assert "- Rollback count: 1" in text
    assert "- Recommendation count: 3" in text
    assert "- Error count: 1" in text
    assert "- Warning count: 1" in text


def test_invalid_report_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="AdaptiveAutomationDoctorReport",
    ):
        AdaptiveAutomationDoctorMarkdownRenderer().render(
            object(),  # type: ignore[arg-type]
        )


def test_public_exports_include_renderer() -> None:
    import lrp.evolution.feedback as feedback

    assert (
        "AdaptiveAutomationDoctorMarkdownRenderer"
        in feedback.__all__
    )
