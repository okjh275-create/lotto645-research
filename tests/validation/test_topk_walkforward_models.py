from __future__ import annotations

import importlib

import pytest

from lrp.contracts import ContractError
from lrp.evaluation import EvaluationWindow


def _api():
    module = importlib.import_module(
        "lrp.evaluation.topk_walkforward"
    )

    return (
        module.HitDistribution,
        module.TopKEvaluation,
        module.WalkForwardRoundEvaluation,
        module.WalkForwardEvaluation,
    )


def _distribution(*counts: int):
    Distribution, _, _, _ = _api()

    return Distribution(
        hit_0=counts[0],
        hit_1=counts[1],
        hit_2=counts[2],
        hit_3=counts[3],
        hit_4=counts[4],
        hit_5=counts[5],
        hit_6=counts[6],
    )


def _topk(k: int):
    _, TopK, _, _ = _api()

    best = _distribution(
        0, 0, 1, 1, 0, 0, 0
    )

    sets = _distribution(
        1, 2, 2, 1, 0, 0, 0
    )

    return TopK(
        k=k,
        round_count=2,
        set_count=2 * k,
        mean_best_hits=2.5,
        mean_set_hits=1.5,
        best_hit_distribution=best,
        set_hit_distribution=sets,
        baseline_delta_mean_best_hits=0.25,
        baseline_delta_3plus_rate=0.10,
        baseline_delta_4plus_rate=0.00,
    )


def test_hit_distribution_contract() -> None:
    Distribution, _, _, _ = _api()

    value = Distribution(
        hit_0=1,
        hit_1=2,
        hit_2=3,
        hit_3=4,
        hit_4=5,
        hit_5=6,
        hit_6=7,
    )

    assert value.hit_0 == 1
    assert value.hit_6 == 7
    assert value.total_count == 28

    payload = value.as_dict()

    assert payload["hit_0"] == 1
    assert payload["hit_6"] == 7
    assert payload["total_count"] == 28

    with pytest.raises(ContractError):
        Distribution(
            hit_0=-1,
            hit_1=0,
            hit_2=0,
            hit_3=0,
            hit_4=0,
            hit_5=0,
            hit_6=0,
        )


def test_hit_distribution_derived_rates() -> None:
    value = _distribution(
        1, 1, 1, 2, 2, 1, 2
    )

    assert value.total_count == 10
    assert value.at_least_3_count == 7
    assert value.at_least_4_count == 5
    assert value.at_least_5_count == 3
    assert value.six_hit_count == 2

    assert value.at_least_3_rate == pytest.approx(0.7)
    assert value.at_least_4_rate == pytest.approx(0.5)
    assert value.at_least_5_rate == pytest.approx(0.3)
    assert value.six_hit_rate == pytest.approx(0.2)


def test_topk_evaluation_contract() -> None:
    value = _topk(5)

    assert value.k == 5
    assert value.round_count == 2
    assert value.set_count == 10
    assert value.mean_best_hits == pytest.approx(2.5)

    payload = value.as_dict()

    assert payload["k"] == 5
    assert payload["best_hit_distribution"]["total_count"] == 2
    assert payload["set_hit_distribution"]["total_count"] == 6


def test_topk_evaluation_rejects_unsupported_k() -> None:
    _, TopK, _, _ = _api()

    with pytest.raises(ContractError):
        TopK(
            k=4,
            round_count=1,
            set_count=4,
            mean_best_hits=1.0,
            mean_set_hits=1.0,
            best_hit_distribution=_distribution(
                0, 1, 0, 0, 0, 0, 0
            ),
            set_hit_distribution=_distribution(
                0, 4, 0, 0, 0, 0, 0
            ),
            baseline_delta_mean_best_hits=0.0,
            baseline_delta_3plus_rate=0.0,
            baseline_delta_4plus_rate=0.0,
        )


def test_round_evaluation_contract() -> None:
    _, _, RoundEvaluation, _ = _api()

    value = RoundEvaluation(
        round_no=1200,
        history_end_round=1199,
        actual_numbers=(1, 2, 3, 4, 5, 6),
        model_name="combined",
        regime_id="R1",
        strategy_name="balanced",
        top3=_topk(3),
        top5=_topk(5),
        top10=_topk(10),
    )

    assert value.round_no == 1200
    assert value.history_end_round == 1199
    assert value.top3.k == 3
    assert value.top10.k == 10


def test_round_evaluation_rejects_future_history_boundary() -> None:
    _, _, RoundEvaluation, _ = _api()

    with pytest.raises(ContractError):
        RoundEvaluation(
            round_no=1200,
            history_end_round=1200,
            actual_numbers=(1, 2, 3, 4, 5, 6),
            model_name="combined",
            regime_id=None,
            strategy_name=None,
            top3=_topk(3),
            top5=_topk(5),
            top10=_topk(10),
        )


def test_walkforward_evaluation_contract() -> None:
    _, _, RoundEvaluation, Evaluation = _api()

    window = EvaluationWindow(
        name="w1",
        start_round=1200,
        end_round=1201,
    )

    rounds = (
        RoundEvaluation(
            round_no=1200,
            history_end_round=1199,
            actual_numbers=(1, 2, 3, 4, 5, 6),
            model_name="combined",
            regime_id="R1",
            strategy_name="balanced",
            top3=_topk(3),
            top5=_topk(5),
            top10=_topk(10),
        ),
        RoundEvaluation(
            round_no=1201,
            history_end_round=1200,
            actual_numbers=(7, 8, 9, 10, 11, 12),
            model_name="combined",
            regime_id="R2",
            strategy_name="balanced",
            top3=_topk(3),
            top5=_topk(5),
            top10=_topk(10),
        ),
    )

    value = Evaluation(
        window=window,
        rounds=rounds,
        top3=_topk(3),
        top5=_topk(5),
        top10=_topk(10),
        model_name="combined",
        regime_slices=(),
        strategy_slices=(),
    )

    assert value.window == window
    assert len(value.rounds) == 2
    assert value.top5.k == 5


def test_walkforward_requires_chronological_unique_rounds() -> None:
    _, _, RoundEvaluation, Evaluation = _api()

    window = EvaluationWindow(
        name="w1",
        start_round=1200,
        end_round=1201,
    )

    duplicate = RoundEvaluation(
        round_no=1200,
        history_end_round=1199,
        actual_numbers=(1, 2, 3, 4, 5, 6),
        model_name="combined",
        regime_id=None,
        strategy_name=None,
        top3=_topk(3),
        top5=_topk(5),
        top10=_topk(10),
    )

    with pytest.raises(ContractError):
        Evaluation(
            window=window,
            rounds=(duplicate, duplicate),
            top3=_topk(3),
            top5=_topk(5),
            top10=_topk(10),
            model_name="combined",
            regime_slices=(),
            strategy_slices=(),
        )
