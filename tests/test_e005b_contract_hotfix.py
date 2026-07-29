"""Regression tests for E-005B contract hotfix."""

from lrp.learning import AdaptiveWeight, AdaptiveWeightReporter


def make_weight(name, rank, previous, current, normalized):
    return AdaptiveWeight(
        strategy_type="model", strategy_name=name, rank_position=rank,
        rank_score=1.0 / rank, target_weight=current,
        previous_weight=previous, current_weight=current,
        normalized_weight=normalized, confidence=0.8, stability=0.8,
        trend="FLAT", sample_count=10, revision=(1, 1),
    )


def test_delta_is_contract_stable():
    report = AdaptiveWeightReporter().build(
        (make_weight("A", 1, 1.0, 1.06, 1.0),),
        generated_at_kst="2026-07-29T18:00:00+09:00",
    )
    assert report.summaries[0].delta == 0.06


def test_normalized_total_property():
    report = AdaptiveWeightReporter().build(
        (make_weight("A", 1, 1.0, 1.1, 0.6),
         make_weight("B", 2, 1.0, 0.9, 0.4)),
        generated_at_kst="2026-07-29T18:00:00+09:00",
    )
    assert report.normalized_total == 1.0
    assert report.as_dict()["normalized_total"] == 1.0


def test_invalid_normalized_total_rejected():
    try:
        AdaptiveWeightReporter().build(
            (make_weight("A", 1, 1.0, 1.0, 0.7),
             make_weight("B", 2, 1.0, 1.0, 0.2)),
            generated_at_kst="2026-07-29T18:00:00+09:00",
        )
    except ValueError as exc:
        assert "sum to 1.0" in str(exc)
    else:
        raise AssertionError("invalid normalized total was not rejected")


def test_empty_report_remains_valid():
    report = AdaptiveWeightReporter().build(
        (), generated_at_kst="2026-07-29T18:00:00+09:00"
    )
    assert report.normalized_total == 0.0


def main():
    test_delta_is_contract_stable()
    test_normalized_total_property()
    test_invalid_normalized_total_rejected()
    test_empty_report_remains_valid()
    print("PASS: E-005B contract hotfix")


if __name__ == "__main__":
    main()
