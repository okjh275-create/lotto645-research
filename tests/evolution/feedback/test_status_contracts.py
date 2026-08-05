from lrp.evolution.feedback import (
    AdaptiveStatusIssue,
    AdaptiveStatusReport,
)


def test_status_report():
    report = AdaptiveStatusReport(
        latest_revision=15,
        repository_ok=True,
        profile_ok=True,
        validation_ok=True,
        rollback_count=1,
        recommendation_count=5,
    )

    assert report.overall_ok is True

    payload = report.as_dict()

    assert payload["overall_ok"] is True


def test_status_issue():
    issue = AdaptiveStatusIssue(
        category="repository",
        severity="warning",
        message="gap",
    )

    payload = issue.as_dict()

    assert payload["category"] == "repository"


def test_public_exports():
    import lrp.evolution.feedback as feedback

    assert (
        "AdaptiveStatusReport"
        in feedback.__all__
    )

    assert (
        "AdaptiveStatusIssue"
        in feedback.__all__
    )
