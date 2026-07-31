from __future__ import annotations

from datetime import datetime, timezone
from math import inf, nan
from typing import Any

import pytest

from lrp.evolution import (
    AdaptiveWeightProfile,
    CallableEvolutionPipeline,
    EvolutionPipeline,
    EvolutionPipelineRequest,
)


FIXED_TIME = datetime(
    2026,
    7,
    31,
    13,
    0,
    0,
    tzinfo=timezone.utc,
)


def make_request(
    **changes: Any,
) -> EvolutionPipelineRequest:
    values: dict[str, Any] = {
        "signals": {
            "hot": 0.30,
            "gap": 0.20,
            "trend": -0.10,
        },
        "confidence": 0.80,
        "sample_size": 40,
        "revision": 2,
        "generated_at": FIXED_TIME,
        "previous_profile": None,
    }
    values.update(changes)

    return EvolutionPipelineRequest(**values)


def make_profile(
    revision: int = 2,
) -> AdaptiveWeightProfile:
    return AdaptiveWeightProfile.default(
        revision=revision,
        generated_at=FIXED_TIME,
    )


def test_request_preserves_valid_values() -> None:
    previous = make_profile(1)

    request = make_request(
        previous_profile=previous
    )

    assert request.signals["hot"] == 0.30
    assert request.confidence == 0.80
    assert request.sample_size == 40
    assert request.revision == 2
    assert request.generated_at == FIXED_TIME
    assert request.previous_profile == previous


def test_request_copies_signals_immutably() -> None:
    signals = {"hot": 0.30}

    request = make_request(signals=signals)
    signals["hot"] = 0.90

    assert request.signals["hot"] == 0.30

    with pytest.raises(TypeError):
        request.signals["hot"] = 0.10  # type: ignore[index]


def test_empty_signals_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="at least one item",
    ):
        make_request(signals={})


def test_empty_signal_name_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        make_request(signals={"  ": 0.2})


@pytest.mark.parametrize(
    "value",
    [nan, inf, -inf],
)
def test_non_finite_signal_is_rejected(
    value: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="must be finite",
    ):
        make_request(signals={"hot": value})


@pytest.mark.parametrize(
    "confidence",
    [-0.01, 1.01],
)
def test_confidence_range_is_validated(
    confidence: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="between 0.0 and 1.0",
    ):
        make_request(confidence=confidence)


def test_boolean_sample_size_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="sample_size must be an integer",
    ):
        make_request(sample_size=True)


def test_revision_must_be_positive() -> None:
    with pytest.raises(
        ValueError,
        match="greater than or equal to 1",
    ):
        make_request(revision=0)


def test_generated_at_must_be_timezone_aware() -> None:
    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        make_request(
            generated_at=datetime(
                2026,
                7,
                31,
                13,
                0,
                0,
            )
        )


def test_previous_profile_type_is_validated() -> None:
    with pytest.raises(
        TypeError,
        match="previous_profile",
    ):
        make_request(
            previous_profile={}  # type: ignore[arg-type]
        )


def test_callable_pipeline_receives_request() -> None:
    received: list[EvolutionPipelineRequest] = []
    expected = make_profile(2)

    def calculator(
        request: EvolutionPipelineRequest,
    ) -> AdaptiveWeightProfile:
        received.append(request)
        return expected

    pipeline = CallableEvolutionPipeline(
        calculator
    )
    request = make_request()

    result = pipeline.calculate(request)

    assert result == expected
    assert received == [request]


def test_callable_pipeline_rejects_wrong_request() -> None:
    pipeline = CallableEvolutionPipeline(
        lambda request: make_profile()
    )

    with pytest.raises(
        TypeError,
        match="EvolutionPipelineRequest",
    ):
        pipeline.calculate(  # type: ignore[arg-type]
            {}
        )


def test_callable_pipeline_validates_return_type() -> None:
    pipeline = CallableEvolutionPipeline(
        lambda request: {}  # type: ignore[return-value]
    )

    with pytest.raises(
        TypeError,
        match="must return an AdaptiveWeightProfile",
    ):
        pipeline.calculate(make_request())


def test_pipeline_protocol_is_runtime_checkable() -> None:
    pipeline = CallableEvolutionPipeline(
        lambda request: make_profile()
    )

    assert isinstance(
        pipeline,
        EvolutionPipeline,
    )


def test_non_callable_calculator_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="calculator must be callable",
    ):
        CallableEvolutionPipeline(  # type: ignore[arg-type]
            None
        )