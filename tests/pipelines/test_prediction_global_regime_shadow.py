from __future__ import annotations

from dataclasses import dataclass

import pytest

import lrp.pipelines.prediction as prediction_module
from lrp.pipelines import (
    PredictionPipeline,
    PredictionRequest,
    prediction_to_dict,
)


def build_draws(module: object) -> tuple[object, ...]:
    draw_type = getattr(module, "DrawRecord")
    draws: list[object] = []

    for round_no in range(1, 81):
        start = ((round_no - 1) * 7) % 45

        numbers = tuple(
            sorted(
                {
                    ((start + offset * 6) % 45) + 1
                    for offset in range(6)
                }
            )
        )

        if len(numbers) != 6:
            raise AssertionError(
                f"invalid synthetic draw: {numbers}"
            )

        draws.append(
            draw_type(
                round=round_no,
                numbers=numbers,
                bonus=None,
            )
        )

    return tuple(draws)


def make_request(
    previous_numbers: frozenset[int],
) -> PredictionRequest:
    return PredictionRequest(
        round_no=81,
        seed=20260721,
        temperature=0.85,
        candidate_count=200,
        max_attempts_multiplier=100,
        top_k=10,
        practical_k=5,
        previous_numbers=previous_numbers,
        long_gap_numbers=frozenset(range(1, 46)),
    )


def test_pipeline_serializes_global_regime_in_shadow_mode() -> None:
    pipeline = PredictionPipeline.load()
    draws = build_draws(pipeline.statistics.module)

    request = make_request(
        frozenset(draws[-1].numbers)
    )

    result = pipeline.run(draws, request)
    payload = prediction_to_dict(result)

    context = result.generation.global_regime_context

    assert context is not None

    regime = payload["metadata"]["global_regime"]

    assert regime is not None
    assert regime["mode"] == "shadow"
    assert regime["primary"] in regime["scores"]
    assert 0.0 <= regime["confidence"] <= 1.0
    assert isinstance(regime["scores"], dict)
    assert isinstance(regime["features"], dict)

    assert set(regime["features"]) == {
        "average_recency",
        "average_frequency",
        "average_gap_reversion",
        "pair_density",
        "frequency_dispersion",
        "recency_variance",
        "pair_variance",
        "low_band_ratio",
        "high_band_ratio",
    }


@dataclass(frozen=True)
class StubGlobalRegimeContext:
    primary: str
    confidence: float

    def as_dict(self) -> dict[str, object]:
        return {
            "primary": self.primary,
            "confidence": self.confidence,
            "secondary": None,
            "secondary_confidence": None,
            "scores": {
                self.primary: self.confidence,
            },
            "features": {
                "marker": self.primary,
            },
        }


class StubFeatureExtractor:
    def extract(self, snapshot: object) -> object:
        return snapshot


class StubStabilityPolicy:
    def __init__(
        self,
        context: StubGlobalRegimeContext,
    ) -> None:
        self.context = context

    def decide(self, features: object) -> StubGlobalRegimeContext:
        return self.context


def run_with_shadow_context(
    *,
    pipeline: PredictionPipeline,
    draws: tuple[object, ...],
    request: PredictionRequest,
    monkeypatch: pytest.MonkeyPatch,
    context: StubGlobalRegimeContext,
) -> dict[str, object]:
    monkeypatch.setattr(
        prediction_module,
        "GlobalRegimeFeatureExtractor",
        lambda: StubFeatureExtractor(),
    )
    monkeypatch.setattr(
        prediction_module,
        "GlobalRegimeStabilityPolicy",
        lambda: StubStabilityPolicy(context),
    )

    result = pipeline.run(draws, request)
    return prediction_to_dict(result)


def test_global_regime_shadow_does_not_change_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = PredictionPipeline.load()
    draws = build_draws(pipeline.statistics.module)

    request = make_request(
        frozenset(draws[-1].numbers)
    )

    first = run_with_shadow_context(
        pipeline=pipeline,
        draws=draws,
        request=request,
        monkeypatch=monkeypatch,
        context=StubGlobalRegimeContext(
            primary="gap_recovery",
            confidence=0.91,
        ),
    )

    second = run_with_shadow_context(
        pipeline=pipeline,
        draws=draws,
        request=request,
        monkeypatch=monkeypatch,
        context=StubGlobalRegimeContext(
            primary="high_band_expansion",
            confidence=0.37,
        ),
    )

    assert (
        first["metadata"]["global_regime"]
        != second["metadata"]["global_regime"]
    )

    first_probability_vector = dict(
        first["probability_vector"]
    )
    second_probability_vector = dict(
        second["probability_vector"]
    )

    first_probability_vector.pop(
        "generated_at_kst",
        None,
    )
    second_probability_vector.pop(
        "generated_at_kst",
        None,
    )

    assert (
        first_probability_vector
        == second_probability_vector
    )
    assert first["sets"] == second["sets"]
    assert (
        first["top5_practical"]
        == second["top5_practical"]
    )
    assert first["diversity"] == second["diversity"]

    first_metadata = dict(first["metadata"])
    second_metadata = dict(second["metadata"])

    first_metadata.pop("global_regime")
    second_metadata.pop("global_regime")

    assert first_metadata == second_metadata
