"""M6-005 adaptive-weight RC1 regression test."""

from __future__ import annotations

from pathlib import Path
import math
import tempfile

from lrp.learning import (
    LearningRepository,
    LearningService,
    PredictionRecord,
    ResultRecord,
)


GENERATED_AT_KST = "2026-07-23T18:30:00+09:00"
RESULT_RECORDED_AT_KST = "2026-07-25T21:00:00+09:00"
REVIEWED_AT_KST = "2026-07-25T21:01:00+09:00"
AGGREGATED_AT_KST = "2026-07-25T21:02:00+09:00"


def make_prediction(
    *,
    prediction_id: str,
    round_no: int,
    model_name: str,
    numbers: tuple[int, ...],
    score: float,
) -> PredictionRecord:
    return PredictionRecord(
        prediction_id=prediction_id,
        round_no=round_no,
        set_id=prediction_id.rsplit(":", 1)[-1],
        numbers=numbers,
        score=score,
        model_name=model_name,
        seed=20260723,
        generated_at_kst=GENERATED_AT_KST,
        features={},
        parameters={
            "temperature": 0.85,
            "scenario": "adaptive-rc1",
        },
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        repository = LearningRepository(Path(temporary) / "learning.db")
        service = LearningService(repository)

        predictions = (
            make_prediction(
                prediction_id="1220:GPT-v3.3:S1",
                round_no=1220,
                model_name="GPT-v3.3",
                numbers=(1, 2, 3, 4, 5, 6),
                score=0.92,
            ),
            make_prediction(
                prediction_id="1220:Gemini-v7.1:S1",
                round_no=1220,
                model_name="Gemini-v7.1",
                numbers=(1, 2, 20, 21, 22, 23),
                score=0.82,
            ),
        )
        for prediction in predictions:
            assert repository.add_prediction(prediction)

        assert repository.add_result(
            ResultRecord(
                round_no=1220,
                numbers=(1, 2, 3, 4, 5, 6),
                bonus=45,
                recorded_at_kst=RESULT_RECORDED_AT_KST,
            )
        )

        review = service.run_incremental_review(
            reviewed_at_kst=REVIEWED_AT_KST,
        )
        assert review.created == 2

        aggregation = service.run_strategy_aggregation(
            aggregated_at_kst=AGGREGATED_AT_KST,
        )
        assert aggregation.created_events == 4

        first = service.get_adaptive_weights(
            strategy_type="model",
            history_limit=100,
        )
        second = service.get_adaptive_weights(
            strategy_type="model",
            history_limit=100,
        )

        assert first == second
        assert len(first) == 2
        assert first[0].strategy_name == "GPT-v3.3"
        assert first[0].normalized_weight > first[1].normalized_weight
        assert math.isclose(
            sum(item.normalized_weight for item in first),
            1.0,
            abs_tol=1e-12,
        )

        for item in first:
            assert 0.50 <= item.current_weight <= 1.50
            assert 0.0 < item.normalized_weight < 1.0
            assert item.revision == (4, 4)

        ranking_before = service.rank_strategies(
            strategy_type="model",
            history_limit=100,
        )
        ranking_after = service.rank_strategies(
            strategy_type="model",
            history_limit=100,
        )
        assert ranking_before == ranking_after

        assert service.adaptive_repository.latest(
            strategy_type="model",
            history_limit=100,
        ) == second

        print("PASS: M6-005 adaptive weight RC1")
        print("weights:", [item.as_dict() for item in first])


if __name__ == "__main__":
    main()
