"""Regression tests for Project E E-005B adaptive-weight reporting."""

from __future__ import annotations

from pathlib import Path
import tempfile

from lrp.learning import (
    AdaptiveWeight,
    AdaptiveWeightReport,
    AdaptiveWeightReporter,
    LearningRepository,
    LearningService,
)


REPORT_AT_KST = "2026-07-27T22:40:00+09:00"
REVISION = (12, 12)


def make_weight(
    *,
    strategy_name: str,
    rank_position: int,
    previous_weight: float,
    current_weight: float,
    normalized_weight: float,
) -> AdaptiveWeight:
    return AdaptiveWeight(
        strategy_type="model",
        strategy_name=strategy_name,
        rank_position=rank_position,
        rank_score=0.80 - (rank_position * 0.05),
        target_weight=current_weight,
        previous_weight=previous_weight,
        current_weight=current_weight,
        normalized_weight=normalized_weight,
        confidence=0.85,
        stability=0.75,
        trend="UP" if current_weight > previous_weight else (
            "DOWN" if current_weight < previous_weight else "FLAT"
        ),
        sample_count=20,
        revision=REVISION,
    )


def test_direct_report() -> None:
    reporter = AdaptiveWeightReporter()
    weights = (
        make_weight(
            strategy_name="GPT-v3.3",
            rank_position=1,
            previous_weight=1.00,
            current_weight=1.10,
            normalized_weight=0.44,
        ),
        make_weight(
            strategy_name="Gemini-v7.1",
            rank_position=2,
            previous_weight=1.10,
            current_weight=1.00,
            normalized_weight=0.40,
        ),
        make_weight(
            strategy_name="Baseline",
            rank_position=3,
            previous_weight=0.40,
            current_weight=0.40,
            normalized_weight=0.16,
        ),
    )

    report = reporter.build(
        weights=weights,
        strategy_type="model",
        history_limit=50,
        generated_at_kst=REPORT_AT_KST,
    )

    assert isinstance(report, AdaptiveWeightReport)
    assert report.revision == REVISION
    assert report.strategy_type == "model"
    assert report.history_limit == 50
    assert report.strategy_count == 3
    assert report.raised_count == 1
    assert report.lowered_count == 1
    assert report.unchanged_count == 1
    assert report.normalized_total == 1.0

    gpt = report.get("model", "GPT-v3.3")
    gemini = report.get("model", "Gemini-v7.1")
    baseline = report.get("model", "Baseline")

    assert gpt is not None
    assert gemini is not None
    assert baseline is not None
    assert gpt.direction == "RAISED"
    assert round(gpt.delta, 6) == 0.10
    assert gemini.direction == "LOWERED"
    assert round(gemini.delta, 6) == -0.10
    assert baseline.direction == "UNCHANGED"

    payload = report.as_dict()
    assert payload["revision"] == [12, 12]
    assert payload["raised_count"] == 1
    assert payload["lowered_count"] == 1
    assert payload["unchanged_count"] == 1
    assert payload["normalized_total"] == 1.0
    assert payload["metadata"]["reporter"] == "E-005B"
    assert (
        payload["metadata"]["calculation_owner"]
        == "AdaptiveWeightEngine"
    )

    second = reporter.build(
        weights=weights,
        strategy_type="model",
        history_limit=50,
        generated_at_kst=REPORT_AT_KST,
    )
    assert second == report


def test_validation() -> None:
    reporter = AdaptiveWeightReporter()

    invalid_weights = (
        make_weight(
            strategy_name="A",
            rank_position=1,
            previous_weight=1.00,
            current_weight=1.00,
            normalized_weight=0.70,
        ),
        make_weight(
            strategy_name="B",
            rank_position=2,
            previous_weight=1.00,
            current_weight=1.00,
            normalized_weight=0.20,
        ),
    )

    try:
        reporter.build(
            weights=invalid_weights,
            strategy_type="model",
            generated_at_kst=REPORT_AT_KST,
        )
    except ValueError as exc:
        assert "sum to 1.0" in str(exc)
    else:
        raise AssertionError(
            "invalid normalized total was not rejected"
        )


def test_empty_service_report() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        repository = LearningRepository(
            Path(temporary) / "learning.db"
        )
        service = LearningService(repository)

        before = repository.counts()
        report = service.get_adaptive_weight_report(
            strategy_type="model",
            history_limit=10,
            generated_at_kst=REPORT_AT_KST,
        )
        after = repository.counts()

        assert before == after
        assert report.revision == (0, 0)
        assert report.strategy_count == 0
        assert report.normalized_total == 0.0
        assert report.metadata["storage"] == "memory_only"


def main() -> None:
    test_direct_report()
    test_validation()
    test_empty_service_report()

    print(
        "PASS: Project E E-005B adaptive-weight reporting"
    )
    print("existing_engine_reuse: PASS")
    print("delta_direction_reporting: PASS")
    print("normalized_total_validation: PASS")
    print("revision_consistency: PASS")
    print("deterministic_report: PASS")
    print("memory_only_storage: PASS")
    print("service_integration: PASS")
    print("public_api_compatibility: PASS")


if __name__ == "__main__":
    main()
