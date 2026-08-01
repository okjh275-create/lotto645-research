from __future__ import annotations

from datetime import datetime, timezone

import pytest

from lrp.evolution.contracts.models import (
    AdaptiveWeightProfile,
)
from lrp.evolution.integration import (
    ProviderEvolutionWeightAdapter,
    StaticProfileProvider,
)
from lrp.prediction.probability import (
    NumberProbability,
    ProbabilityVector,
)


def make_vector() -> ProbabilityVector:
    records = tuple(
        NumberProbability(
            number=number,
            probability=1.0 / 45.0,
            raw_score=1.0,
            rank=number,
            components={
                "hot": number / 45.0,
                "cold": 1.0 - number / 45.0,
                "gap": 0.5,
                "trend": 0.5,
                "transition": 0.5,
                "learning": 0.5,
                "adaptive": 0.5,
            },
            metadata={},
        )
        for number in range(1, 46)
    )

    return ProbabilityVector(
        round_no=1220,
        generated_at_kst=(
            "2026-08-02T09:00:00+09:00"
        ),
        probabilities=records,
        metadata={},
    )


def make_profile() -> AdaptiveWeightProfile:
    return AdaptiveWeightProfile(
        hot_weight=0.50,
        cold_weight=0.10,
        gap_weight=0.10,
        trend_weight=0.10,
        transition_weight=0.10,
        learning_weight=0.05,
        adaptive_weight=0.05,
        confidence=0.80,
        sample_size=40,
        revision=3,
        generated_at=datetime(
            2026,
            8,
            2,
            tzinfo=timezone.utc,
        ),
    )


def test_returns_original_when_profile_missing() -> None:
    vector = make_vector()

    adapter = ProviderEvolutionWeightAdapter(
        StaticProfileProvider(None)
    )

    result = adapter.adjust(
        vector,
        round_no=1220,
        seed=20260802,
    )

    assert result is vector


def test_applies_provider_profile() -> None:
    vector = make_vector()

    adapter = ProviderEvolutionWeightAdapter(
        StaticProfileProvider(
            make_profile()
        )
    )

    result = adapter.adjust(
        vector,
        round_no=1220,
        seed=20260802,
    )

    assert result is not vector
    assert result.metadata[
        "evolution_revision"
    ] == 3
    assert result.get(45).rank == 1


def test_invalid_provider_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="AdaptiveWeightProfileProvider",
    ):
        ProviderEvolutionWeightAdapter(
            object(),  # type: ignore[arg-type]
        )
