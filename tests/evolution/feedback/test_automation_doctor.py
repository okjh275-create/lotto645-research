from pathlib import Path

from lrp.evolution.feedback import (
    AdaptiveAutomationDoctor,
    AdaptiveAutomationRepository,
)


def test_empty_repository():

    report = (
        AdaptiveAutomationDoctor()
        .inspect(
            AdaptiveAutomationRepository(
                Path(".")
            )
        )
    )

    assert report.latest_revision == 0
    assert report.rollback_count == 0
    assert report.recommendation_count == 0
    assert report.overall_ok is False


def test_dict_output():

    report = (
        AdaptiveAutomationDoctor()
        .inspect(
            AdaptiveAutomationRepository(
                Path(".")
            )
        )
    )

    payload = report.as_dict()

    assert "repository" in payload
    assert "profile" in payload
    assert "issues" in payload


def test_public_api():

    import lrp.evolution.feedback as feedback

    assert (
        "AdaptiveAutomationDoctor"
        in feedback.__all__
    )

    assert (
        "AdaptiveAutomationDoctorReport"
        in feedback.__all__
    )
