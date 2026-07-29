"""Regression tests for Project E E-005B Adaptive Weight Reporting."""

from __future__ import annotations

from lrp.learning import (
    AdaptiveWeight,
    AdaptiveWeightExplanation,
    AdaptiveWeightReport,
    AdaptiveWeightReporter,
)


REPORT_AT_KST = "2026-07-27T22:40:00+09:00"


def make_weight(
    *,
    strategy_name: str,
    rank_position: int,
    previous_weight: float,
    target_weight: float,
    current_weight: float,
    normalized_weight: float,
    confidence: float,
    stability: float,
    trend: str,
) -> AdaptiveWeight:
    return AdaptiveWeight(
        strategy_type="model",
        strategy_name=strategy_name,
        rank_position=rank_position,
        rank_score=max(0.0, 1.0 - rank_position * 0.1),
        target_weight=target_weight,
        previous_weight=previous_weight,
        current_weight=current_weight,
        normalized_weight=normalized_weight,
        confidence=confidence,
        stability=stability,
        trend=trend,
        sample_count=30,
        revision=(16, 16),
    )


def test_empty_report() -> None:
    report = AdaptiveWeightReporter().build(
        (),
        strategy_type="model",
        history_limit=10,
        generated_at_kst=REPORT_AT_KST,
    )
    assert isinstance(report, AdaptiveWeightReport)
    assert report.revision == (0, 0)
    assert report.strategy_count == 0
    assert report.raised_count == 0
    assert report.highest is None
    assert report.lowest is None


def test_report_summary_and_explanations() -> None:
    weights = (
        make_weight(
            strategy_name="GPT-v3.3",
            rank_position=1,
            previous_weight=1.00,
            target_weight=1.30,
            current_weight=1.06,
            normalized_weight=0.40,
            confidence=0.90,
            stability=0.85,
            trend="UP",
        ),
        make_weight(
            strategy_name="Gemini-v7.1",
            rank_position=2,
            previous_weight=1.00,
            target_weight=0.80,
            current_weight=0.96,
            normalized_weight=0.35,
            confidence=0.80,
            stability=0.60,
            trend="DOWN",
        ),
        make_weight(
            strategy_name="Baseline",
            rank_position=3,
            previous_weight=1.00,
            target_weight=1.00,
            current_weight=1.00,
            normalized_weight=0.25,
            confidence=0.50,
            stability=0.80,
            trend="FLAT",
        ),
    )

    report = AdaptiveWeightReporter().build(
        weights,
        strategy_type="model",
        history_limit=100,
        generated_at_kst=REPORT_AT_KST,
    )

    assert report.revision == (16, 16)
    assert report.strategy_count == 3
    assert report.raised_count == 1
    assert report.lowered_count == 1
    assert report.unchanged_count == 1
    assert report.highest is not None
    assert report.highest.strategy_name == "GPT-v3.3"
    assert report.lowest is not None
    assert report.lowest.strategy_name == "Gemini-v7.1"

    gpt = report.get("model", "GPT-v3.3")
    assert isinstance(gpt, AdaptiveWeightExplanation)
    assert gpt.direction == "RAISED"
    assert gpt.delta == 0.06
    assert "최근 성과 추세 상승" in gpt.reasons
    assert "충분한 표본에 따른 높은 신뢰도" in gpt.reasons

    payload = report.as_dict()
    assert payload["revision"] == [16, 16]
    assert payload["raised_count"] == 1
    assert payload["summaries"][0]["strategy_name"] == "GPT-v3.3"


def test_deterministic_output() -> None:
    weights = (
        make_weight(
            strategy_name="B",
            rank_position=2,
            previous_weight=1.0,
            target_weight=0.9,
            current_weight=0.98,
            normalized_weight=0.49,
            confidence=0.7,
            stability=0.7,
            trend="FLAT",
        ),
        make_weight(
            strategy_name="A",
            rank_position=1,
            previous_weight=1.0,
            target_weight=1.1,
            current_weight=1.02,
            normalized_weight=0.51,
            confidence=0.8,
            stability=0.8,
            trend="UP",
        ),
    )
    reporter = AdaptiveWeightReporter()
    first = reporter.build(
        weights,
        history_limit=100,
        generated_at_kst=REPORT_AT_KST,
    )
    second = reporter.build(
        tuple(reversed(weights)),
        history_limit=100,
        generated_at_kst=REPORT_AT_KST,
    )
    assert first == second
    assert [item.strategy_name for item in first.summaries] == ["A", "B"]


def test_mixed_revision_rejected() -> None:
    first = make_weight(
        strategy_name="A",
        rank_position=1,
        previous_weight=1.0,
        target_weight=1.1,
        current_weight=1.02,
        normalized_weight=0.5,
        confidence=0.8,
        stability=0.8,
        trend="UP",
    )
    second = AdaptiveWeight(
        strategy_type="model",
        strategy_name="B",
        rank_position=2,
        rank_score=0.7,
        target_weight=0.9,
        previous_weight=1.0,
        current_weight=0.98,
        normalized_weight=0.5,
        confidence=0.7,
        stability=0.7,
        trend="DOWN",
        sample_count=20,
        revision=(17, 17),
    )
    try:
        AdaptiveWeightReporter().build((first, second))
    except ValueError as exc:
        assert "mixed revisions" in str(exc)
    else:
        raise AssertionError("mixed revisions must be rejected")


def main() -> None:
    test_empty_report()
    test_report_summary_and_explanations()
    test_deterministic_output()
    test_mixed_revision_rejected()

    print("PASS: Project E E-005B adaptive weight reporting")
    print("empty_report: PASS")
    print("summary_counts: PASS")
    print("human_explanations: PASS")
    print("deterministic_output: PASS")
    print("revision_validation: PASS")
    print("public_api_compatibility: PASS")


if __name__ == "__main__":
    main()
