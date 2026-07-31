from __future__ import annotations

from datetime import datetime, timezone

import pytest

from lrp.evolution.algorithms.adaptive import (
    AdaptiveWeightCalculator,
)
from lrp.evolution.contracts.models import (
    AdaptiveWeightProfile,
)
from lrp.evolution.contracts.pipeline import (
    EvolutionPipelineRequest,
)
from lrp.evolution.services import (
    AdaptiveEvolutionPipeline,
    EvolutionPipeline,
)


FIXED_TIME = datetime(
    2026,
    7,
    31,
    12,
    0,
    0,
    tzinfo=timezone.utc,
)


def make_request(
    *,
    signals: dict[str, float] | None = None,
    confidence: float = 0.80,
    sample_size: int = 40,
    revision: int = 2,
    previous_profile: AdaptiveWeightProfile | None = None,
) -> EvolutionPipelineRequest:
    return EvolutionPipelineRequest(
        signals=signals or {"hot": 0.5},
        confidence=confidence,
        sample_size=sample_size,
        revision=revision,
        generated_at=FIXED_TIME,
        previous_profile=previous_profile,
    )


def test_default_calculator_is_created() -> None:
    pipeline = AdaptiveEvolutionPipeline()

    assert isinstance(
        pipeline.calculator,
        AdaptiveWeightCalculator,
    )


def test_custom_calculator_is_preserved() -> None:
    calculator = AdaptiveWeightCalculator(
        adjustment_scale=0.40,
    )

    pipeline = AdaptiveEvolutionPipeline(
        calculator=calculator,
    )

    assert pipeline.calculator is calculator


def test_invalid_calculator_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="AdaptiveWeightCalculator or None",
    ):
        AdaptiveEvolutionPipeline(
            calculator=object(),  # type: ignore[arg-type]
        )


def test_pipeline_satisfies_protocol() -> None:
    pipeline = AdaptiveEvolutionPipeline()

    assert isinstance(
        pipeline,
        EvolutionPipeline,
    )


def test_calculate_returns_adaptive_profile() -> None:
    pipeline = AdaptiveEvolutionPipeline()

    profile = pipeline.calculate(
        make_request()
    )

    assert isinstance(
        profile,
        AdaptiveWeightProfile,
    )


def test_request_metadata_is_forwarded() -> None:
    pipeline = AdaptiveEvolutionPipeline()

    profile = pipeline.calculate(
        make_request(
            confidence=0.75,
            sample_size=36,
            revision=5,
        )
    )

    assert profile.confidence == 0.75
    assert profile.sample_size == 36
    assert profile.revision == 5
    assert profile.generated_at == FIXED_TIME


def test_signals_are_forwarded() -> None:
    pipeline = AdaptiveEvolutionPipeline()

    profile = pipeline.calculate(
        make_request(
            signals={"hot": 1.0},
            confidence=1.0,
        )
    )

    baseline = AdaptiveWeightProfile.default(
        generated_at=FIXED_TIME,
    )

    assert profile.hot_weight > baseline.hot_weight


def test_previous_profile_is_used_as_baseline() -> None:
    previous = AdaptiveWeightProfile(
        hot_weight=0.20,
        cold_weight=0.20,
        gap_weight=0.20,
        trend_weight=0.10,
        transition_weight=0.10,
        learning_weight=0.10,
        adaptive_weight=0.10,
        confidence=0.70,
        sample_size=30,
        revision=3,
        generated_at=FIXED_TIME,
    )

    pipeline = AdaptiveEvolutionPipeline()

    profile = pipeline.calculate(
        make_request(
            signals={"hot": 0.5},
            confidence=0.0,
            sample_size=31,
            revision=4,
            previous_profile=previous,
        )
    )

    assert profile.to_probability_weights() == (
        pytest.approx(
            previous.to_probability_weights()
        )
    )


def test_invalid_request_is_rejected() -> None:
    pipeline = AdaptiveEvolutionPipeline()

    with pytest.raises(
        TypeError,
        match="EvolutionPipelineRequest",
    ):
        pipeline.calculate(  # type: ignore[arg-type]
            {"hot": 0.5}
        )


def test_calculator_configuration_affects_result() -> None:
    low_adjustment = AdaptiveEvolutionPipeline(
        AdaptiveWeightCalculator(
            adjustment_scale=0.10,
        )
    )
    high_adjustment = AdaptiveEvolutionPipeline(
        AdaptiveWeightCalculator(
            adjustment_scale=0.80,
        )
    )

    request = make_request(
        signals={"hot": 1.0},
        confidence=1.0,
    )

    low_profile = low_adjustment.calculate(request)
    high_profile = high_adjustment.calculate(request)

    assert (
        high_profile.hot_weight
        > low_profile.hot_weight
    )


def test_services_public_api_exports_pipeline() -> None:
    pipeline = AdaptiveEvolutionPipeline()

    assert isinstance(
        pipeline.calculator,
        AdaptiveWeightCalculator,
    )
