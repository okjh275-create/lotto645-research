from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest

from lrp.contracts.exceptions import ContractError
from lrp.io.draws import HistoryRow
from lrp.pipelines.models import (
    PredictionGenerationResult,
    PredictionRequest,
    PredictionResult,
)
from lrp.evaluation.topk_live_evaluation_snapshot_factory import (
    TopKLiveEvaluationSnapshotBuildRequest,
    TopKLiveEvaluationSnapshotFactory,
)


class _SelectedItem:
    def __init__(
        self,
        numbers: tuple[int, ...],
    ) -> None:
        self.numbers = numbers


class _Diversity:
    def __init__(
        self,
        selected: tuple[_SelectedItem, ...],
    ) -> None:
        self.selected = selected


def _prediction_result() -> PredictionResult:
    request = PredictionRequest(
        round_no=1233,
        seed=20260821,
        top_k=2,
        practical_k=1,
        long_gap_numbers=frozenset({45}),
    )

    generation = PredictionGenerationResult(
        request=request,
        windows=(10, 20, 50),
        probabilities={},
        statistics_contract=object(),
        number_signals={},
        candidates=(),
        statistics_version="stats-v1",
        candidate_version="candidate-v1",
    )

    return PredictionResult(
        generation=generation,
        scored_candidates=(),
        ranking=object(),
        diversity=_Diversity(
            (
                _SelectedItem(
                    (41, 1, 32, 7, 24, 13),
                ),
                _SelectedItem(
                    (44, 35, 27, 18, 9, 3),
                ),
            )
        ),
        practical=object(),
        generated_at_kst=datetime.fromisoformat(
            "2026-08-21T17:00:00+09:00"
        ),
    )


def _history_rows() -> tuple[HistoryRow, ...]:
    return (
        HistoryRow(
            round_no=1230,
            numbers=(1, 2, 3, 4, 5, 6),
            bonus=7,
        ),
        HistoryRow(
            round_no=1231,
            numbers=(8, 9, 10, 11, 12, 13),
            bonus=14,
        ),
        HistoryRow(
            round_no=1232,
            numbers=(15, 16, 17, 18, 19, 20),
            bonus=21,
        ),
    )


def _request(
    *,
    prediction_result: PredictionResult | None = None,
    history_rows: tuple[HistoryRow, ...] | None = None,
) -> TopKLiveEvaluationSnapshotBuildRequest:
    return TopKLiveEvaluationSnapshotBuildRequest(
        prediction_result=(
            prediction_result
            if prediction_result is not None
            else _prediction_result()
        ),
        history_rows=(
            history_rows
            if history_rows is not None
            else _history_rows()
        ),
        model_name="candidate-v1",
        source_artifact_sha256="a" * 64,
        regime_id="regime-a",
        strategy_name="strategy-a",
    )


def test_build_request_is_immutable() -> None:
    request = _request()

    with pytest.raises(FrozenInstanceError):
        request.model_name = "changed"


def test_factory_does_not_mutate_history_rows() -> None:
    rows = _history_rows()
    before = tuple(rows)

    TopKLiveEvaluationSnapshotFactory().build(
        request=_request(
            history_rows=rows,
        )
    )

    assert rows == before


def test_factory_does_not_mutate_prediction_result() -> None:
    prediction = _prediction_result()

    generation_before = prediction.generation
    diversity_before = prediction.diversity
    timestamp_before = prediction.generated_at_kst

    TopKLiveEvaluationSnapshotFactory().build(
        request=_request(
            prediction_result=prediction,
        )
    )

    assert prediction.generation is generation_before
    assert prediction.diversity is diversity_before
    assert prediction.generated_at_kst == timestamp_before


def test_factory_rejects_reverse_history_end_to_end() -> None:
    rows = tuple(
        reversed(
            _history_rows()
        )
    )

    request = _request(
        history_rows=rows,
    )

    with pytest.raises(ContractError):
        TopKLiveEvaluationSnapshotFactory().build(
            request=request
        )


def test_factory_rejects_invalid_build_request_type() -> None:
    with pytest.raises(ContractError):
        TopKLiveEvaluationSnapshotFactory().build(
            request=object(),
        )


def test_factory_repeated_build_is_exactly_stable() -> None:
    request = _request()
    factory = TopKLiveEvaluationSnapshotFactory()

    first = factory.build(
        request=request
    )

    second = factory.build(
        request=request
    )

    assert first == second


def test_factory_product_has_no_runtime_nondeterminism_dependency() -> None:
    from pathlib import Path

    source = Path(
        "lrp/evaluation/"
        "topk_live_evaluation_snapshot_factory.py"
    ).read_text(
        encoding="utf-8-sig"
    )

    forbidden = (
        "datetime.now",
        "datetime.utcnow",
        "random",
        "secrets",
        "uuid",
        "time.time",
    )

    assert not any(
        token in source
        for token in forbidden
    )


def test_factory_product_has_no_persistence_dependency() -> None:
    from pathlib import Path

    source = Path(
        "lrp/evaluation/"
        "topk_live_evaluation_snapshot_factory.py"
    ).read_text(
        encoding="utf-8-sig"
    )

    forbidden = (
        "open(",
        "write_text(",
        "write_bytes(",
        "sqlite3",
        "write_prediction_artifacts",
        "round_completion",
        "subprocess",
    )

    assert not any(
        token in source
        for token in forbidden
    )