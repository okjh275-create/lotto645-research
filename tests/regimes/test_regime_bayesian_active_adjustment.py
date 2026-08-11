from __future__ import annotations

from lrp.evolution.contracts.bayesian import (
    BayesianPosterior,
)
from lrp.prediction.probability import (
    NumberProbability,
    ProbabilityVector,
)
from lrp.regimes import (
    RegimeDecision,
    RegimeFeatureSnapshot,
)
from lrp.regimes.bayesian_state import (
    RegimeBayesianState,
)
from lrp.regimes.integration import (
    ActiveGlobalRegimeAdjustmentAdapter,
    StaticRegimeBayesianProvider,
)


def make_vector() -> ProbabilityVector:
    probabilities = tuple(
        NumberProbability(
            number=number,
            probability=1.0 / 45.0,
            raw_score=1.0,
            rank=number,
            components={
                "gap": (
                    1.0
                    if number == 45
                    else 0.2
                ),
                "transition": 0.0,
            },
            metadata={},
        )
        for number in range(1, 46)
    )

    return ProbabilityVector(
        round_no=1220,
        generated_at_kst=(
            "2026-08-11T12:00:00+09:00"
        ),
        probabilities=probabilities,
        metadata={},
    )


def make_gap_regime() -> RegimeDecision:
    return RegimeDecision(
        primary="gap_recovery",
        confidence=1.0,
        features=RegimeFeatureSnapshot(
            average_recency=0.5,
            average_frequency=0.5,
            average_gap_reversion=0.8,
            pair_density=0.5,
            frequency_dispersion=0.5,
            recency_variance=0.5,
            pair_variance=0.5,
            low_band_ratio=0.3,
            high_band_ratio=0.3,
        ),
        scores={
            "neutral": 0.0,
            "mixed": 0.0,
            "gap_recovery": 1.0,
            "cluster_rotation": 0.0,
            "high_band_expansion": 0.0,
            "low_band_expansion": 0.0,
        },
    )


def make_positive_state() -> RegimeBayesianState:
    return RegimeBayesianState.from_posteriors(
        {
            "gap_recovery": BayesianPosterior(
                alpha=5.0,
                beta=1.0,
            ),
            "cluster_rotation": BayesianPosterior(
                alpha=1.0,
                beta=1.0,
            ),
            "high_band_expansion": BayesianPosterior(
                alpha=1.0,
                beta=1.0,
            ),
            "low_band_expansion": BayesianPosterior(
                alpha=1.0,
                beta=1.0,
            ),
        }
    )


def test_adapter_accepts_bayesian_provider() -> None:
    provider = StaticRegimeBayesianProvider(
        RegimeBayesianState.default()
    )

    adapter = ActiveGlobalRegimeAdjustmentAdapter(
        bayesian_provider=provider
    )

    assert adapter.bayesian_provider is provider


def test_neutral_bayesian_state_preserves_active_adjustment() -> None:
    vector = make_vector()
    regime = make_gap_regime()

    baseline = ActiveGlobalRegimeAdjustmentAdapter()

    neutral = ActiveGlobalRegimeAdjustmentAdapter(
        bayesian_provider=(
            StaticRegimeBayesianProvider(
                RegimeBayesianState.default()
            )
        )
    )

    baseline_adjusted = baseline.adjust(
        vector,
        global_regime=regime,
        round_no=1220,
        seed=20260811,
    )

    neutral_adjusted = neutral.adjust(
        vector,
        global_regime=regime,
        round_no=1220,
        seed=20260811,
    )

    baseline_scores = tuple(
        item.raw_score
        for item
        in baseline_adjusted.probabilities
    )
    neutral_scores = tuple(
        item.raw_score
        for item
        in neutral_adjusted.probabilities
    )

    assert neutral_scores == baseline_scores


def test_positive_bayesian_signal_strengthens_target_regime() -> None:
    vector = make_vector()
    regime = make_gap_regime()

    baseline = ActiveGlobalRegimeAdjustmentAdapter()

    learned = ActiveGlobalRegimeAdjustmentAdapter(
        bayesian_provider=(
            StaticRegimeBayesianProvider(
                make_positive_state()
            )
        )
    )

    baseline_adjusted = baseline.adjust(
        vector,
        global_regime=regime,
        round_no=1220,
        seed=20260811,
    )

    learned_adjusted = learned.adjust(
        vector,
        global_regime=regime,
        round_no=1220,
        seed=20260811,
    )

    assert (
        learned_adjusted.get(45).raw_score
        > baseline_adjusted.get(45).raw_score
    )


def test_none_bayesian_state_preserves_active_adjustment() -> None:
    vector = make_vector()
    regime = make_gap_regime()

    baseline = ActiveGlobalRegimeAdjustmentAdapter()

    unavailable = ActiveGlobalRegimeAdjustmentAdapter(
        bayesian_provider=(
            StaticRegimeBayesianProvider(None)
        )
    )

    baseline_adjusted = baseline.adjust(
        vector,
        global_regime=regime,
        round_no=1220,
        seed=20260811,
    )

    unavailable_adjusted = unavailable.adjust(
        vector,
        global_regime=regime,
        round_no=1220,
        seed=20260811,
    )

    baseline_scores = tuple(
        item.raw_score
        for item
        in baseline_adjusted.probabilities
    )
    unavailable_scores = tuple(
        item.raw_score
        for item
        in unavailable_adjusted.probabilities
    )

    assert unavailable_scores == baseline_scores


def make_negative_state() -> RegimeBayesianState:
    return RegimeBayesianState.from_posteriors(
        {
            "gap_recovery": BayesianPosterior(
                alpha=1.0,
                beta=5.0,
            ),
            "cluster_rotation": BayesianPosterior(
                alpha=1.0,
                beta=1.0,
            ),
            "high_band_expansion": BayesianPosterior(
                alpha=1.0,
                beta=1.0,
            ),
            "low_band_expansion": BayesianPosterior(
                alpha=1.0,
                beta=1.0,
            ),
        }
    )


def make_extreme_positive_state() -> RegimeBayesianState:
    return RegimeBayesianState.from_posteriors(
        {
            "gap_recovery": BayesianPosterior(
                alpha=1000.0,
                beta=1.0,
            ),
            "cluster_rotation": BayesianPosterior(
                alpha=1.0,
                beta=1.0,
            ),
            "high_band_expansion": BayesianPosterior(
                alpha=1.0,
                beta=1.0,
            ),
            "low_band_expansion": BayesianPosterior(
                alpha=1.0,
                beta=1.0,
            ),
        }
    )


def make_extreme_negative_state() -> RegimeBayesianState:
    return RegimeBayesianState.from_posteriors(
        {
            "gap_recovery": BayesianPosterior(
                alpha=1.0,
                beta=1000.0,
            ),
            "cluster_rotation": BayesianPosterior(
                alpha=1.0,
                beta=1.0,
            ),
            "high_band_expansion": BayesianPosterior(
                alpha=1.0,
                beta=1.0,
            ),
            "low_band_expansion": BayesianPosterior(
                alpha=1.0,
                beta=1.0,
            ),
        }
    )


def test_negative_bayesian_signal_weakens_target_regime() -> None:
    vector = make_vector()
    regime = make_gap_regime()

    baseline = ActiveGlobalRegimeAdjustmentAdapter()

    learned = ActiveGlobalRegimeAdjustmentAdapter(
        bayesian_provider=(
            StaticRegimeBayesianProvider(
                make_negative_state()
            )
        )
    )

    baseline_adjusted = baseline.adjust(
        vector,
        global_regime=regime,
        round_no=1220,
        seed=20260811,
    )

    learned_adjusted = learned.adjust(
        vector,
        global_regime=regime,
        round_no=1220,
        seed=20260811,
    )

    assert (
        learned_adjusted.get(45).raw_score
        < baseline_adjusted.get(45).raw_score
    )


def test_positive_bayesian_signal_is_clamped() -> None:
    vector = make_vector()
    regime = make_gap_regime()

    extreme = ActiveGlobalRegimeAdjustmentAdapter(
        bayesian_provider=(
            StaticRegimeBayesianProvider(
                make_extreme_positive_state()
            )
        )
    )

    bounded = ActiveGlobalRegimeAdjustmentAdapter(
        bayesian_provider=(
            StaticRegimeBayesianProvider(
                RegimeBayesianState.from_posteriors(
                    {
                        "gap_recovery": BayesianPosterior(
                            alpha=5.0,
                            beta=3.0,
                        ),
                        "cluster_rotation": BayesianPosterior(
                            alpha=1.0,
                            beta=1.0,
                        ),
                        "high_band_expansion": BayesianPosterior(
                            alpha=1.0,
                            beta=1.0,
                        ),
                        "low_band_expansion": BayesianPosterior(
                            alpha=1.0,
                            beta=1.0,
                        ),
                    }
                )
            )
        )
    )

    extreme_adjusted = extreme.adjust(
        vector,
        global_regime=regime,
        round_no=1220,
        seed=20260811,
    )

    bounded_adjusted = bounded.adjust(
        vector,
        global_regime=regime,
        round_no=1220,
        seed=20260811,
    )

    assert (
        extreme_adjusted.get(45).raw_score
        == bounded_adjusted.get(45).raw_score
    )


def test_negative_bayesian_signal_is_clamped() -> None:
    vector = make_vector()
    regime = make_gap_regime()

    extreme = ActiveGlobalRegimeAdjustmentAdapter(
        bayesian_provider=(
            StaticRegimeBayesianProvider(
                make_extreme_negative_state()
            )
        )
    )

    bounded = ActiveGlobalRegimeAdjustmentAdapter(
        bayesian_provider=(
            StaticRegimeBayesianProvider(
                RegimeBayesianState.from_posteriors(
                    {
                        "gap_recovery": BayesianPosterior(
                            alpha=3.0,
                            beta=5.0,
                        ),
                        "cluster_rotation": BayesianPosterior(
                            alpha=1.0,
                            beta=1.0,
                        ),
                        "high_band_expansion": BayesianPosterior(
                            alpha=1.0,
                            beta=1.0,
                        ),
                        "low_band_expansion": BayesianPosterior(
                            alpha=1.0,
                            beta=1.0,
                        ),
                    }
                )
            )
        )
    )

    extreme_adjusted = extreme.adjust(
        vector,
        global_regime=regime,
        round_no=1220,
        seed=20260811,
    )

    bounded_adjusted = bounded.adjust(
        vector,
        global_regime=regime,
        round_no=1220,
        seed=20260811,
    )

    assert (
        extreme_adjusted.get(45).raw_score
        == bounded_adjusted.get(45).raw_score
    )
