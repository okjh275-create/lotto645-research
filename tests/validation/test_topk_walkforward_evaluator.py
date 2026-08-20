from __future__ import annotations

import importlib
from dataclasses import dataclass

import pytest

from lrp.contracts import ContractError
from lrp.evaluation import EvaluationWindow


@dataclass(frozen=True)
class ReplayRow:
    round_no: int
    history_rounds: tuple[int, ...]
    actual_numbers: tuple[int, ...]
    predictions: tuple[tuple[int, ...], ...]
    model_name: str = "combined"
    regime_id: str | None = None
    strategy_name: str | None = None


class BaselineProvider:
    def __init__(
        self,
        rows: dict[int, ReplayRow],
    ) -> None:
        self.rows = dict(rows)

    def get(
        self,
        round_no: int,
    ) -> ReplayRow:
        try:
            return self.rows[round_no]
        except KeyError as exc:
            raise ContractError(
                "baseline round missing"
            ) from exc


def _api():
    module = importlib.import_module(
        "lrp.evaluation.topk_walkforward"
    )

    return module.TopKWalkForwardEvaluator


def _predictions() -> tuple[tuple[int, ...], ...]:
    return (
        (1, 2, 3, 20, 21, 22),
        (1, 2, 7, 8, 9, 10),
        (1, 11, 12, 13, 14, 15),
        (1, 2, 3, 4, 16, 17),
        (1, 2, 3, 4, 5, 6),
        (7, 8, 9, 22, 23, 24),
        (10, 11, 12, 25, 26, 27),
        (13, 14, 15, 28, 29, 30),
        (16, 17, 18, 31, 32, 33),
        (19, 20, 21, 34, 35, 36),
    )


def _baseline_predictions() -> tuple[tuple[int, ...], ...]:
    return (
        (1, 7, 8, 9, 10, 11),
        (2, 12, 13, 14, 15, 16),
        (3, 17, 18, 19, 20, 21),
        (4, 22, 23, 24, 25, 26),
        (5, 27, 28, 29, 30, 31),
        (6, 32, 33, 34, 35, 36),
        (7, 8, 37, 38, 39, 40),
        (9, 10, 11, 41, 42, 43),
        (12, 13, 14, 15, 44, 45),
        (16, 17, 18, 19, 20, 21),
    )


def _row(
    round_no: int,
    *,
    history_end: int | None = None,
    predictions: tuple[tuple[int, ...], ...] | None = None,
    regime_id: str | None = "R1",
    strategy_name: str | None = "balanced",
) -> ReplayRow:
    if history_end is None:
        history_end = round_no - 1

    return ReplayRow(
        round_no=round_no,
        history_rounds=tuple(
            range(
                max(1, history_end - 4),
                history_end + 1,
            )
        ),
        actual_numbers=(1, 2, 3, 4, 5, 6),
        predictions=(
            _predictions()
            if predictions is None
            else predictions
        ),
        model_name="combined",
        regime_id=regime_id,
        strategy_name=strategy_name,
    )


def _baseline_row(
    round_no: int,
) -> ReplayRow:
    return _row(
        round_no,
        predictions=_baseline_predictions(),
        regime_id=None,
        strategy_name=None,
    )


def _evaluate(
    rows: tuple[ReplayRow, ...],
):
    Evaluator = _api()

    provider = BaselineProvider(
        {
            row.round_no: _baseline_row(
                row.round_no
            )
            for row in rows
        }
    )

    evaluator = Evaluator(
        baseline_provider=provider
    )

    window = EvaluationWindow(
        name="w1",
        start_round=rows[0].round_no,
        end_round=rows[-1].round_no,
    )

    return evaluator.evaluate(
        window=window,
        replay_rows=rows,
    )


def test_evaluator_top3_metrics() -> None:
    result = _evaluate(
        (
            _row(1200),
            _row(1201),
        )
    )

    assert result.top3.k == 3
    assert result.top3.round_count == 2
    assert result.top3.mean_best_hits == pytest.approx(3.0)


def test_evaluator_top5_metrics() -> None:
    result = _evaluate(
        (
            _row(1200),
            _row(1201),
        )
    )

    assert result.top5.k == 5
    assert result.top5.set_count == 10
    assert result.top5.mean_best_hits >= result.top3.mean_best_hits


def test_evaluator_top10_metrics() -> None:
    result = _evaluate(
        (
            _row(1200),
            _row(1201),
        )
    )

    assert result.top10.k == 10
    assert result.top10.set_count == 20
    assert result.top10.mean_best_hits >= result.top5.mean_best_hits


def test_evaluator_best_hit_distribution() -> None:
    result = _evaluate(
        (
            _row(1200),
            _row(1201),
        )
    )

    assert result.top3.best_hit_distribution.total_count == 2
    assert result.top5.best_hit_distribution.total_count == 2
    assert result.top10.best_hit_distribution.total_count == 2


def test_evaluator_set_hit_distribution() -> None:
    result = _evaluate(
        (
            _row(1200),
            _row(1201),
        )
    )

    assert result.top3.set_hit_distribution.total_count == 6
    assert result.top5.set_hit_distribution.total_count == 10
    assert result.top10.set_hit_distribution.total_count == 20


def test_evaluator_3plus_rate() -> None:
    result = _evaluate(
        (
            _row(1200),
            _row(1201),
        )
    )

    assert (
        result.top3
        .best_hit_distribution
        .at_least_3_rate
        == pytest.approx(1.0)
    )


def test_evaluator_4plus_rate() -> None:
    result = _evaluate(
        (
            _row(1200),
            _row(1201),
        )
    )

    assert (
        result.top5
        .best_hit_distribution
        .at_least_4_rate
        == pytest.approx(1.0)
    )


def test_evaluator_5plus_and_6hit_rates() -> None:
    result = _evaluate(
        (
            _row(1200),
            _row(1201),
        )
    )

    assert (
        result.top5
        .best_hit_distribution
        .at_least_5_rate
        == pytest.approx(1.0)
    )

    assert (
        result.top5
        .best_hit_distribution
        .six_hit_rate
        == pytest.approx(1.0)
    )


def test_evaluator_baseline_deltas() -> None:
    result = _evaluate(
        (
            _row(1200),
            _row(1201),
        )
    )

    assert result.top3.baseline_delta_mean_best_hits > 0.0
    assert result.top5.baseline_delta_3plus_rate >= 0.0
    assert result.top10.baseline_delta_4plus_rate >= 0.0


def test_evaluator_rejects_insufficient_prediction_sets() -> None:
    row = _row(
        1200,
        predictions=_predictions()[:5],
    )

    with pytest.raises(ContractError):
        _evaluate(
            (row,)
        )


def test_evaluator_rejects_duplicate_rounds() -> None:
    row = _row(1200)

    with pytest.raises(ContractError):
        _evaluate(
            (
                row,
                row,
            )
        )


def test_evaluator_rejects_out_of_order_rounds() -> None:
    with pytest.raises(ContractError):
        _evaluate(
            (
                _row(1201),
                _row(1200),
            )
        )


def test_evaluator_rejects_future_history() -> None:
    row = ReplayRow(
        round_no=1200,
        history_rounds=(1199, 1201),
        actual_numbers=(1, 2, 3, 4, 5, 6),
        predictions=_predictions(),
        model_name="combined",
        regime_id=None,
        strategy_name=None,
    )

    with pytest.raises(ContractError):
        _evaluate(
            (row,)
        )


def test_evaluator_rejects_prediction_round_in_history() -> None:
    row = ReplayRow(
        round_no=1200,
        history_rounds=(1199, 1200),
        actual_numbers=(1, 2, 3, 4, 5, 6),
        predictions=_predictions(),
        model_name="combined",
        regime_id=None,
        strategy_name=None,
    )

    with pytest.raises(ContractError):
        _evaluate(
            (row,)
        )


def test_evaluator_baseline_uses_same_round_boundary() -> None:
    Evaluator = _api()

    candidate = _row(1200)

    bad_baseline = ReplayRow(
        round_no=1200,
        history_rounds=(1198,),
        actual_numbers=(1, 2, 3, 4, 5, 6),
        predictions=_baseline_predictions(),
        model_name="baseline",
        regime_id=None,
        strategy_name=None,
    )

    evaluator = Evaluator(
        baseline_provider=BaselineProvider(
            {
                1200: bad_baseline,
            }
        )
    )

    window = EvaluationWindow(
        name="w1",
        start_round=1200,
        end_round=1200,
    )

    with pytest.raises(ContractError):
        evaluator.evaluate(
            window=window,
            replay_rows=(candidate,),
        )


def test_evaluator_repeated_result_is_deterministic() -> None:
    rows = (
        _row(1200),
        _row(1201),
    )

    first = _evaluate(rows)
    second = _evaluate(rows)

    assert first.as_dict() == second.as_dict()


def test_evaluator_regime_slice_when_provenance_present() -> None:
    result = _evaluate(
        (
            _row(
                1200,
                regime_id="R2",
            ),
            _row(
                1201,
                regime_id="R1",
            ),
        )
    )

    names = tuple(
        slice_value.name
        for slice_value in result.regime_slices
    )

    assert names == (
        "R1",
        "R2",
    )


def test_evaluator_strategy_slice_when_provenance_present() -> None:
    result = _evaluate(
        (
            _row(
                1200,
                strategy_name="B",
            ),
            _row(
                1201,
                strategy_name="A",
            ),
        )
    )

    names = tuple(
        slice_value.name
        for slice_value in result.strategy_slices
    )

    assert names == (
        "A",
        "B",
    )


def test_evaluator_missing_optional_slice_does_not_fail() -> None:
    result = _evaluate(
        (
            _row(
                1200,
                regime_id=None,
                strategy_name=None,
            ),
        )
    )

    assert result.regime_slices == ()
    assert result.strategy_slices == ()
