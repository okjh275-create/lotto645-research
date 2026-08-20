from __future__ import annotations

from dataclasses import dataclass

import pytest

from lrp.contracts import ContractError
from lrp.evaluation import EvaluationWindow
from lrp.evaluation.topk_replay_adapter import (
    TopKReplayAdapter,
    TopKReplayBaselineProvider,
    TopKReplayPrediction,
)
from lrp.evaluation.topk_walkforward import (
    TopKWalkForwardEvaluator,
)


@dataclass(frozen=True)
class Draw:
    round_no: int
    numbers: tuple[int, ...]
    bonus: int = 45


def _prediction_sets() -> tuple[tuple[int, ...], ...]:
    return (
        (6, 5, 4, 3, 2, 1),
        (12, 11, 10, 9, 8, 7),
        (18, 17, 16, 15, 14, 13),
        (24, 23, 22, 21, 20, 19),
        (30, 29, 28, 27, 26, 25),
        (36, 35, 34, 33, 32, 31),
        (42, 41, 40, 39, 38, 37),
        (45, 44, 43, 3, 2, 1),
        (15, 14, 13, 12, 11, 10),
        (25, 24, 23, 22, 21, 20),
    )


def _prediction(
    round_no: int = 1200,
    *,
    history_rounds: tuple[int, ...] | None = None,
    predictions: tuple[tuple[int, ...], ...] | None = None,
    model_name: str = "combined",
    regime_id: str | None = "R1",
    strategy_name: str | None = "S1",
) -> TopKReplayPrediction:
    if history_rounds is None:
        history_rounds = (
            round_no - 3,
            round_no - 2,
            round_no - 1,
        )

    if predictions is None:
        predictions = _prediction_sets()

    return TopKReplayPrediction(
        round_no=round_no,
        history_rounds=history_rounds,
        predictions=predictions,
        model_name=model_name,
        regime_id=regime_id,
        strategy_name=strategy_name,
    )


def _draw(
    round_no: int = 1200,
    numbers: tuple[int, ...] = (
        6,
        5,
        4,
        3,
        2,
        1,
    ),
) -> Draw:
    return Draw(
        round_no=round_no,
        numbers=numbers,
    )


def _adapt(
    predictions: tuple[TopKReplayPrediction, ...],
    draws: tuple[Draw, ...],
):
    return TopKReplayAdapter().adapt(
        prediction_rows=predictions,
        actual_draws=draws,
    )


def test_adapter_builds_canonical_replay_rows() -> None:
    rows = _adapt(
        (
            _prediction(),
        ),
        (
            _draw(),
        ),
    )

    assert len(rows) == 1

    row = rows[0]

    assert row.round_no == 1200
    assert row.history_rounds == (1197, 1198, 1199)
    assert row.actual_numbers == (1, 2, 3, 4, 5, 6)
    assert row.predictions[0] == (1, 2, 3, 4, 5, 6)
    assert row.model_name == "combined"


def test_adapter_orders_rows_by_round() -> None:
    rows = _adapt(
        (
            _prediction(1202),
            _prediction(1200),
            _prediction(1201),
        ),
        (
            _draw(1201),
            _draw(1202),
            _draw(1200),
        ),
    )

    assert tuple(
        row.round_no
        for row in rows
    ) == (
        1200,
        1201,
        1202,
    )


def test_adapter_preserves_prediction_set_order() -> None:
    predictions = _prediction_sets()

    rows = _adapt(
        (
            _prediction(
                predictions=predictions
            ),
        ),
        (
            _draw(),
        ),
    )

    expected = tuple(
        tuple(sorted(values))
        for values in predictions
    )

    assert rows[0].predictions == expected


def test_adapter_normalizes_numbers_within_sets() -> None:
    rows = _adapt(
        (
            _prediction(),
        ),
        (
            _draw(),
        ),
    )

    for numbers in rows[0].predictions:
        assert numbers == tuple(
            sorted(numbers)
        )


def test_adapter_attaches_actual_draw() -> None:
    rows = _adapt(
        (
            _prediction(),
        ),
        (
            _draw(
                numbers=(45, 44, 43, 42, 41, 40)
            ),
        ),
    )

    assert rows[0].actual_numbers == (
        40,
        41,
        42,
        43,
        44,
        45,
    )


def test_adapter_preserves_model_name() -> None:
    rows = _adapt(
        (
            _prediction(
                model_name="candidate-x"
            ),
        ),
        (
            _draw(),
        ),
    )

    assert rows[0].model_name == "candidate-x"


def test_adapter_preserves_optional_regime_and_strategy() -> None:
    rows = _adapt(
        (
            _prediction(
                regime_id="REGIME-A",
                strategy_name="STRATEGY-B",
            ),
        ),
        (
            _draw(),
        ),
    )

    assert rows[0].regime_id == "REGIME-A"
    assert rows[0].strategy_name == "STRATEGY-B"


def test_adapter_rejects_blank_model_name() -> None:
    with pytest.raises(ContractError):
        _prediction(
            model_name=""
        )


def test_adapter_rejects_empty_history() -> None:
    with pytest.raises(ContractError):
        _prediction(
            history_rounds=(),
        )


def test_adapter_rejects_unsorted_history() -> None:
    with pytest.raises(ContractError):
        _prediction(
            history_rounds=(
                1198,
                1197,
                1199,
            ),
        )


def test_adapter_rejects_duplicate_history_round() -> None:
    with pytest.raises(ContractError):
        _prediction(
            history_rounds=(
                1197,
                1198,
                1198,
            ),
        )


def test_adapter_rejects_prediction_round_in_history() -> None:
    with pytest.raises(ContractError):
        _prediction(
            history_rounds=(
                1198,
                1199,
                1200,
            ),
        )


def test_adapter_rejects_future_history_round() -> None:
    with pytest.raises(ContractError):
        _prediction(
            history_rounds=(
                1199,
                1201,
            ),
        )


def test_adapter_rejects_fewer_than_ten_predictions() -> None:
    with pytest.raises(ContractError):
        _prediction(
            predictions=_prediction_sets()[:9],
        )


def test_adapter_rejects_invalid_prediction_numbers() -> None:
    bad = list(
        _prediction_sets()
    )

    bad[0] = (
        1,
        2,
        3,
        4,
        5,
        46,
    )

    with pytest.raises(ContractError):
        _prediction(
            predictions=tuple(bad),
        )


def test_adapter_rejects_duplicate_prediction_round() -> None:
    with pytest.raises(ContractError):
        _adapt(
            (
                _prediction(1200),
                _prediction(1200),
            ),
            (
                _draw(1200),
            ),
        )


def test_adapter_rejects_missing_actual_draw() -> None:
    with pytest.raises(ContractError):
        _adapt(
            (
                _prediction(1200),
            ),
            (),
        )


def test_adapter_rejects_duplicate_actual_draw_round() -> None:
    with pytest.raises(ContractError):
        _adapt(
            (
                _prediction(1200),
            ),
            (
                _draw(1200),
                _draw(1200),
            ),
        )


def test_adapter_rejects_invalid_actual_numbers() -> None:
    with pytest.raises(ContractError):
        _adapt(
            (
                _prediction(),
            ),
            (
                _draw(
                    numbers=(
                        1,
                        1,
                        2,
                        3,
                        4,
                        5,
                    )
                ),
            ),
        )


def test_baseline_provider_resolves_exact_round() -> None:
    rows = _adapt(
        (
            _prediction(
                1200,
                model_name="baseline",
            ),
            _prediction(
                1201,
                model_name="baseline",
            ),
        ),
        (
            _draw(1200),
            _draw(1201),
        ),
    )

    provider = TopKReplayBaselineProvider(
        rows
    )

    assert provider.get(1200) == rows[0]
    assert provider.get(1201) == rows[1]


def test_baseline_provider_rejects_missing_round() -> None:
    rows = _adapt(
        (
            _prediction(
                1200,
                model_name="baseline",
            ),
        ),
        (
            _draw(1200),
        ),
    )

    provider = TopKReplayBaselineProvider(
        rows
    )

    with pytest.raises(ContractError):
        provider.get(1201)


def test_baseline_provider_rejects_duplicate_round() -> None:
    first = _adapt(
        (
            _prediction(
                1200,
                model_name="baseline-a",
            ),
        ),
        (
            _draw(1200),
        ),
    )[0]

    second = _adapt(
        (
            _prediction(
                1200,
                model_name="baseline-b",
            ),
        ),
        (
            _draw(1200),
        ),
    )[0]

    with pytest.raises(ContractError):
        TopKReplayBaselineProvider(
            (
                first,
                second,
            )
        )


def test_baseline_provider_preserves_history_boundary() -> None:
    rows = _adapt(
        (
            _prediction(
                1200,
                model_name="baseline",
                history_rounds=(
                    1195,
                    1196,
                    1197,
                    1198,
                    1199,
                ),
            ),
        ),
        (
            _draw(1200),
        ),
    )

    provider = TopKReplayBaselineProvider(
        rows
    )

    assert (
        provider.get(
            1200
        ).history_rounds
        == (
            1195,
            1196,
            1197,
            1198,
            1199,
        )
    )


def test_adapter_result_is_directly_consumable_by_topk_evaluator() -> None:
    candidate_rows = _adapt(
        (
            _prediction(
                1200,
                model_name="candidate",
            ),
        ),
        (
            _draw(1200),
        ),
    )

    baseline_rows = _adapt(
        (
            _prediction(
                1200,
                model_name="baseline",
            ),
        ),
        (
            _draw(1200),
        ),
    )

    evaluator = TopKWalkForwardEvaluator(
        baseline_provider=TopKReplayBaselineProvider(
            baseline_rows
        )
    )

    result = evaluator.evaluate(
        window=EvaluationWindow(
            name="ab04",
            start_round=1200,
            end_round=1200,
        ),
        replay_rows=candidate_rows,
    )

    assert result.model_name == "candidate"
    assert len(result.rounds) == 1
