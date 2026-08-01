from __future__ import annotations

from dataclasses import dataclass

import pytest

from lrp.contracts import ContractError
from lrp.evolution.integration import (
    NoOpEvolutionWeightAdapter,
)
from lrp.pipelines.models import (
    PredictionRequest,
)
from lrp.pipelines.prediction import (
    PredictionPipeline,
)


def make_request() -> PredictionRequest:
    return PredictionRequest(
        round_no=1220,
        seed=20260802,
        candidate_count=10,
        top_k=5,
        practical_k=3,
        previous_numbers=frozenset(
            {1, 2, 3, 4, 5, 6}
        ),
        long_gap_numbers=frozenset(
            {7, 8, 9}
        ),
    )


@dataclass
class RecordingEvolutionAdapter:
    result: object
    calls: list[
        tuple[object, int, int]
    ]

    def adjust(
        self,
        probability_vector: object,
        *,
        round_no: int,
        seed: int,
    ) -> object:
        self.calls.append(
            (
                probability_vector,
                round_no,
                seed,
            )
        )
        return self.result


class NoneEvolutionAdapter:
    def adjust(
        self,
        probability_vector: object,
        *,
        round_no: int,
        seed: int,
    ) -> object:
        return None


def test_pipeline_uses_noop_adapter_by_default() -> None:
    pipeline = PredictionPipeline(
        statistics=object(),  # type: ignore[arg-type]
        candidate=object(),  # type: ignore[arg-type]
    )

    assert isinstance(
        pipeline.evolution,
        NoOpEvolutionWeightAdapter,
    )


def test_pipeline_accepts_structural_adapter() -> None:
    adapter = RecordingEvolutionAdapter(
        result=object(),
        calls=[],
    )

    pipeline = PredictionPipeline(
        statistics=object(),  # type: ignore[arg-type]
        candidate=object(),  # type: ignore[arg-type]
        evolution=adapter,
    )

    assert pipeline.evolution is adapter


def test_pipeline_rejects_invalid_adapter() -> None:
    with pytest.raises(
        ContractError,
        match="EvolutionWeightAdapter",
    ):
        PredictionPipeline(
            statistics=object(),  # type: ignore[arg-type]
            candidate=object(),  # type: ignore[arg-type]
            evolution=object(),  # type: ignore[arg-type]
        )


def test_adjustment_receives_round_and_seed() -> None:
    original = object()
    adjusted = object()
    adapter = RecordingEvolutionAdapter(
        result=adjusted,
        calls=[],
    )
    pipeline = PredictionPipeline(
        statistics=object(),  # type: ignore[arg-type]
        candidate=object(),  # type: ignore[arg-type]
        evolution=adapter,
    )

    result = pipeline._adjust_probability_vector(
        original,
        request=make_request(),
    )

    assert result is adjusted
    assert adapter.calls == [
        (
            original,
            1220,
            20260802,
        )
    ]


def test_default_adjustment_preserves_identity() -> None:
    original = object()
    pipeline = PredictionPipeline(
        statistics=object(),  # type: ignore[arg-type]
        candidate=object(),  # type: ignore[arg-type]
    )

    result = pipeline._adjust_probability_vector(
        original,
        request=make_request(),
    )

    assert result is original


def test_none_adjustment_is_rejected() -> None:
    pipeline = PredictionPipeline(
        statistics=object(),  # type: ignore[arg-type]
        candidate=object(),  # type: ignore[arg-type]
        evolution=NoneEvolutionAdapter(),
    )

    with pytest.raises(
        ContractError,
        match="returned None",
    ):
        pipeline._adjust_probability_vector(
            object(),
            request=make_request(),
        )
