from __future__ import annotations

from types import SimpleNamespace
from datetime import datetime, timezone

import pytest

from lrp.contracts.exceptions import ContractError
from lrp.pipelines.models import (
    PredictionGenerationResult,
    PredictionRequest,
    PredictionResult,
)
from lrp.evaluation.topk_replay_adapter import (
    TopKReplayAdapter,
    TopKReplayPrediction,
)
from lrp.evaluation.topk_prediction_source_adapter import (
    TopKPredictionSourceAdapter,
    TopKPredictionSourceRecord,
)


def _candidate(
    numbers: object,
) -> object:
    return SimpleNamespace(
        numbers=numbers,
    )


def _ten_candidates(
    first: tuple[int, ...],
    second: tuple[int, ...],
) -> tuple[object, ...]:
    tail = tuple(
        _candidate(
            (
                start,
                start + 1,
                start + 2,
                start + 3,
                start + 4,
                start + 5,
            )
        )
        for start in (
            13,
            17,
            21,
            25,
            29,
            33,
            37,
            39,
        )
    )

    return (
        _candidate(
            first
        ),
        _candidate(
            second
        ),
        *tail,
    )


def _prediction_result(
    *,
    round_no: int = 1200,
    top_k: int = 10,
    selected: object | None = None,
    diversity: object | None = None,
) -> PredictionResult:
    request = PredictionRequest(
        round_no=round_no,
        seed=20260821,
        long_gap_numbers=frozenset({45}),
        top_k=top_k,
        practical_k=min(
            5,
            top_k,
        ),
    )

    generation = PredictionGenerationResult(
        request=request,
        windows=(10, 20, 50),
        probabilities={1: 1.0},
        statistics_contract=object(),
        number_signals={},
        candidates=(),
        statistics_version="test-statistics",
        candidate_version="test-candidate",
    )

    if diversity is None:
        if selected is None:
            selected = tuple(
                _candidate(
                    (
                        start + 5,
                        start,
                        start + 3,
                        start + 2,
                        start + 4,
                        start + 1,
                    )
                )
                for start in (
                    1,
                    5,
                    9,
                    13,
                    17,
                    21,
                    25,
                    29,
                    33,
                    37,
                )[:top_k]
            )

        diversity = SimpleNamespace(
            selected=selected,
        )

    return PredictionResult(
        generation=generation,
        scored_candidates=(),
        ranking=object(),
        diversity=diversity,
        practical=object(),
        generated_at_kst=datetime.now(
            timezone.utc
        ),
    )


def _source(
    *,
    prediction_result: PredictionResult | None = None,
    model_name: str = "candidate",
    history_rounds: tuple[int, ...] = (
        1197,
        1198,
        1199,
    ),
    regime_id: str | None = "regime-a",
    strategy_name: str | None = "strategy-a",
) -> TopKPredictionSourceRecord:
    return TopKPredictionSourceRecord(
        prediction_result=(
            _prediction_result()
            if prediction_result is None
            else prediction_result
        ),
        model_name=model_name,
        history_rounds=history_rounds,
        regime_id=regime_id,
        strategy_name=strategy_name,
    )


def _adapt(
    source: TopKPredictionSourceRecord | None = None,
) -> TopKReplayPrediction:
    return TopKPredictionSourceAdapter().adapt(
        source=(
            _source()
            if source is None
            else source
        )
    )


def test_adapter_maps_round_identity() -> None:
    result = _prediction_result(
        round_no=1200,
    )

    replay = _adapt(
        _source(
            prediction_result=result,
        )
    )

    assert replay.round_no == 1200


def test_adapter_maps_model_identity() -> None:
    replay = _adapt(
        _source(
            model_name="candidate-v2",
        )
    )

    assert replay.model_name == "candidate-v2"


def test_adapter_maps_history_rounds() -> None:
    history = (
        1195,
        1196,
        1197,
        1198,
        1199,
    )

    replay = _adapt(
        _source(
            history_rounds=history,
        )
    )

    assert replay.history_rounds == history


def test_adapter_preserves_optional_provenance() -> None:
    replay = _adapt(
        _source(
            regime_id="regime-x",
            strategy_name="strategy-x",
        )
    )

    assert replay.regime_id == "regime-x"
    assert replay.strategy_name == "strategy-x"


def test_adapter_preserves_none_optional_provenance() -> None:
    replay = _adapt(
        _source(
            regime_id=None,
            strategy_name=None,
        )
    )

    assert replay.regime_id is None
    assert replay.strategy_name is None


def test_adapter_extracts_diversity_selected_predictions() -> None:
    result = _prediction_result(
        selected=_ten_candidates(
            (1, 2, 3, 4, 5, 6),
            (7, 8, 9, 10, 11, 12),
        )
    )

    replay = _adapt(
        _source(
            prediction_result=result,
        )
    )

    assert len(
        replay.predictions
    ) == 10

    assert replay.predictions[:2] == (
        (1, 2, 3, 4, 5, 6),
        (7, 8, 9, 10, 11, 12),
    )


def test_adapter_preserves_selected_prediction_order() -> None:
    result = _prediction_result(
        selected=_ten_candidates(
            (20, 19, 18, 17, 16, 15),
            (6, 5, 4, 3, 2, 1),
        )
    )

    replay = _adapt(
        _source(
            prediction_result=result,
        )
    )

    assert len(
        replay.predictions
    ) == 10

    assert replay.predictions[:2] == (
        (15, 16, 17, 18, 19, 20),
        (1, 2, 3, 4, 5, 6),
    )


def test_adapter_normalizes_numbers_ascending() -> None:
    result = _prediction_result(
        selected=_ten_candidates(
            (6, 1, 5, 2, 4, 3),
            (12, 7, 11, 8, 10, 9),
        )
    )

    replay = _adapt(
        _source(
            prediction_result=result,
        )
    )

    assert replay.predictions[0] == (
        1,
        2,
        3,
        4,
        5,
        6,
    )

    assert replay.predictions[1] == (
        7,
        8,
        9,
        10,
        11,
        12,
    )


def test_adapter_rejects_missing_selected_output() -> None:
    result = _prediction_result(
        diversity=SimpleNamespace()
    )

    source = _source(
        prediction_result=result,
    )

    with pytest.raises(
        ContractError
    ):
        TopKPredictionSourceAdapter().adapt(
            source=source
        )


def test_adapter_rejects_wrong_prediction_count() -> None:
    result = _prediction_result(
        top_k=10,
        selected=(
            _candidate(
                (1, 2, 3, 4, 5, 6)
            ),
        ),
    )

    with pytest.raises(
        ContractError
    ):
        _adapt(
            _source(
                prediction_result=result,
            )
        )


def test_adapter_rejects_duplicate_prediction_sets() -> None:
    result = _prediction_result(
        selected=_ten_candidates(
            (1, 2, 3, 4, 5, 6),
            (6, 5, 4, 3, 2, 1),
        )
    )

    with pytest.raises(
        ContractError
    ):
        _adapt(
            _source(
                prediction_result=result,
            )
        )


def test_adapter_rejects_invalid_candidate_numbers() -> None:
    invalid_number_sets = (
        None,
        (1, 2, 3, 4, 5),
        (1, 2, 3, 4, 5, 5),
        (0, 1, 2, 3, 4, 5),
        (1, 2, 3, 4, 5, 46),
        (1, 2, 3, 4, 5, True),
        (1, 2, 3, 4, 5, "6"),
    )

    for invalid in invalid_number_sets:
        result = _prediction_result(
            selected=_ten_candidates(
                invalid,
                (7, 8, 9, 10, 11, 12),
            )
        )

        with pytest.raises(
            ContractError
        ):
            _adapt(
                _source(
                    prediction_result=result,
                )
            )


def test_adapter_rejects_history_round_at_target() -> None:
    with pytest.raises(
        ContractError
    ):
        _source(
            history_rounds=(
                1199,
                1200,
            )
        )


def test_adapter_rejects_future_history_round() -> None:
    with pytest.raises(
        ContractError
    ):
        _source(
            history_rounds=(
                1199,
                1201,
            )
        )


def test_adapter_result_is_directly_consumable_by_topk_replay_adapter() -> None:
    prediction = _adapt(
        _source(
            regime_id=None,
            strategy_name=None,
        )
    )

    rows = TopKReplayAdapter().adapt(
        prediction_rows=(
            prediction,
        ),
        actual_draws=(
            SimpleNamespace(
                round_no=1200,
                numbers=(
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                ),
            ),
        ),
    )

    assert len(rows) == 1
    assert rows[0].round_no == 1200
    assert rows[0].model_name == "candidate"
