from __future__ import annotations

import math

from lrp.prediction.probability import (
    NumberProbability,
    ProbabilityVector,
)
from lrp.regimes.contracts import (
    RegimeDecision,
    RegimeFeatureSnapshot,
)
from lrp.regimes.integration.active_adjustment import (
    ActiveGlobalRegimeAdjustmentAdapter,
)


def make_vector() -> ProbabilityVector:
    raw_scores = {
        number: 1.0
        for number in range(1, 46)
    }

    total = sum(raw_scores.values())

    probabilities = tuple(
        NumberProbability(
            number=number,
            probability=raw_scores[number] / total,
            raw_score=raw_scores[number],
            rank=number,
            components={
                "gap": number / 45.0,
                "transition": (46 - number) / 45.0,
            },
            metadata={
                "source": "test",
            },
        )
        for number in range(1, 46)
    )

    return ProbabilityVector(
        round_no=1220,
        generated_at_kst="2026-08-09T22:00:00+09:00",
        probabilities=probabilities,
        metadata={
            "source": "unit-test",
        },
    )


def make_regime(
    primary: str,
    *,
    confidence: float = 1.0,
) -> RegimeDecision:
    scores = {
        "neutral": 0.0,
        "mixed": 0.0,
        "gap_recovery": 0.0,
        "cluster_rotation": 0.0,
        "high_band_expansion": 0.0,
        "low_band_expansion": 0.0,
    }
    scores[primary] = confidence

    return RegimeDecision(
        primary=primary,
        confidence=confidence,
        features=RegimeFeatureSnapshot(
            average_recency=0.5,
            average_frequency=0.5,
            average_gap_reversion=0.5,
            pair_density=0.5,
            frequency_dispersion=0.5,
            recency_variance=0.5,
            pair_variance=0.5,
            low_band_ratio=0.5,
            high_band_ratio=0.5,
        ),
        scores=scores,
    )


def test_none_regime_preserves_identity() -> None:
    vector = make_vector()
    adapter = ActiveGlobalRegimeAdjustmentAdapter()

    adjusted = adapter.adjust(
        vector,
        global_regime=None,
        round_no=1220,
        seed=20260809,
    )

    assert adjusted is vector


def test_neutral_regime_preserves_identity() -> None:
    vector = make_vector()
    adapter = ActiveGlobalRegimeAdjustmentAdapter()

    adjusted = adapter.adjust(
        vector,
        global_regime=make_regime("neutral"),
        round_no=1220,
        seed=20260809,
    )

    assert adjusted is vector


def test_mixed_regime_preserves_identity() -> None:
    vector = make_vector()
    adapter = ActiveGlobalRegimeAdjustmentAdapter()

    adjusted = adapter.adjust(
        vector,
        global_regime=make_regime("mixed"),
        round_no=1220,
        seed=20260809,
    )

    assert adjusted is vector


def test_high_band_expansion_boosts_31_to_45() -> None:
    vector = make_vector()
    adapter = ActiveGlobalRegimeAdjustmentAdapter()

    adjusted = adapter.adjust(
        vector,
        global_regime=make_regime(
            "high_band_expansion",
            confidence=1.0,
        ),
        round_no=1220,
        seed=20260809,
    )

    assert adjusted.get(31).raw_score > vector.get(31).raw_score
    assert adjusted.get(45).raw_score > vector.get(45).raw_score
    assert adjusted.get(30).raw_score == vector.get(30).raw_score


def test_low_band_expansion_boosts_1_to_15() -> None:
    vector = make_vector()
    adapter = ActiveGlobalRegimeAdjustmentAdapter()

    adjusted = adapter.adjust(
        vector,
        global_regime=make_regime(
            "low_band_expansion",
            confidence=1.0,
        ),
        round_no=1220,
        seed=20260809,
    )

    assert adjusted.get(1).raw_score > vector.get(1).raw_score
    assert adjusted.get(15).raw_score > vector.get(15).raw_score
    assert adjusted.get(16).raw_score == vector.get(16).raw_score


def test_gap_recovery_uses_gap_component_strength() -> None:
    vector = make_vector()
    adapter = ActiveGlobalRegimeAdjustmentAdapter()

    adjusted = adapter.adjust(
        vector,
        global_regime=make_regime(
            "gap_recovery",
            confidence=1.0,
        ),
        round_no=1220,
        seed=20260809,
    )

    assert adjusted.get(45).raw_score > adjusted.get(1).raw_score


def test_cluster_rotation_uses_transition_component_strength() -> None:
    vector = make_vector()
    adapter = ActiveGlobalRegimeAdjustmentAdapter()

    adjusted = adapter.adjust(
        vector,
        global_regime=make_regime(
            "cluster_rotation",
            confidence=1.0,
        ),
        round_no=1220,
        seed=20260809,
    )

    assert adjusted.get(1).raw_score > adjusted.get(45).raw_score


def test_zero_confidence_produces_no_score_change() -> None:
    vector = make_vector()
    adapter = ActiveGlobalRegimeAdjustmentAdapter()

    adjusted = adapter.adjust(
        vector,
        global_regime=make_regime(
            "high_band_expansion",
            confidence=0.0,
        ),
        round_no=1220,
        seed=20260809,
    )

    for original, result in zip(
        vector.probabilities,
        adjusted.probabilities,
        strict=True,
    ):
        assert result.raw_score == original.raw_score


def test_active_adjustment_remains_normalized() -> None:
    adjusted = ActiveGlobalRegimeAdjustmentAdapter().adjust(
        make_vector(),
        global_regime=make_regime(
            "high_band_expansion",
            confidence=0.9,
        ),
        round_no=1220,
        seed=20260809,
    )

    total = sum(
        item.probability
        for item in adjusted.probabilities
    )

    assert math.isclose(
        total,
        1.0,
        rel_tol=1e-9,
        abs_tol=1e-9,
    )
    assert (
        adjusted.metadata["global_regime_adjusted"]
        is True
    )


def test_same_input_is_deterministic() -> None:
    vector = make_vector()
    regime = make_regime(
        "gap_recovery",
        confidence=0.8,
    )
    adapter = ActiveGlobalRegimeAdjustmentAdapter()

    first = adapter.adjust(
        vector,
        global_regime=regime,
        round_no=1220,
        seed=20260809,
    )
    second = adapter.adjust(
        vector,
        global_regime=regime,
        round_no=1220,
        seed=20260809,
    )

    assert first.as_dict() == second.as_dict()
