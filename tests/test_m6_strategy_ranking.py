"""M6-004 strategy-ranking regression test."""

from __future__ import annotations

from pathlib import Path
import tempfile

from lrp.learning import (
    LearningRepository,
    LearningService,
    PredictionRecord,
    ResultRecord,
)


GENERATED_AT_KST = "2026-07-23T05:00:00+09:00"
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
    scenario: str,
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
        features={
            "gap_mix": 0.70,
            "pair_affinity": 0.60,
        },
        parameters={
            "temperature": 0.85,
            "scenario": scenario,
        },
    )


def add_round(
    *,
    repository: LearningRepository,
    round_no: int,
    winning_numbers: tuple[int, ...],
    bonus: int,
    predictions: tuple[PredictionRecord, ...],
) -> None:
    for prediction in predictions:
        assert repository.add_prediction(prediction) is True

    assert repository.add_result(
        ResultRecord(
            round_no=round_no,
            numbers=winning_numbers,
            bonus=bonus,
            recorded_at_kst=RESULT_RECORDED_AT_KST,
        )
    ) is True


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        database_path = Path(temporary) / "learning.db"
        repository = LearningRepository(database_path)
        service = LearningService(repository)

        add_round(
            repository=repository,
            round_no=1220,
            winning_numbers=(1, 2, 3, 4, 5, 6),
            bonus=7,
            predictions=(
                make_prediction(
                    prediction_id="1220:GPT-v3.3:S1",
                    round_no=1220,
                    model_name="GPT-v3.3",
                    numbers=(1, 2, 3, 4, 5, 6),
                    score=0.92,
                    scenario="gap",
                ),
                make_prediction(
                    prediction_id="1220:Gemini-v7.1:S1",
                    round_no=1220,
                    model_name="Gemini-v7.1",
                    numbers=(1, 2, 3, 20, 21, 22),
                    score=0.84,
                    scenario="pair",
                ),
            ),
        )

        add_round(
            repository=repository,
            round_no=1221,
            winning_numbers=(10, 11, 12, 13, 14, 15),
            bonus=16,
            predictions=(
                make_prediction(
                    prediction_id="1221:GPT-v3.3:S1",
                    round_no=1221,
                    model_name="GPT-v3.3",
                    numbers=(10, 11, 12, 13, 30, 31),
                    score=0.88,
                    scenario="gap",
                ),
                make_prediction(
                    prediction_id="1221:Gemini-v7.1:S1",
                    round_no=1221,
                    model_name="Gemini-v7.1",
                    numbers=(10, 11, 25, 26, 27, 28),
                    score=0.82,
                    scenario="pair",
                ),
            ),
        )

        review = service.run_incremental_review(
            reviewed_at_kst=REVIEWED_AT_KST,
        )
        assert review.scanned == 4
        assert review.created == 4
        assert review.skipped == 0

        aggregation = service.run_strategy_aggregation(
            aggregated_at_kst=AGGREGATED_AT_KST,
        )
        assert aggregation.scanned == 4
        assert aggregation.created_events == 8
        assert aggregation.skipped_events == 0

        model_rankings = service.rank_strategies(
            strategy_type="model",
            history_limit=100,
        )

        assert len(model_rankings) == 2
        assert model_rankings[0].rank_position == 1
        assert model_rankings[1].rank_position == 2
        assert model_rankings[0].strategy_name == "GPT-v3.3"
        assert model_rankings[1].strategy_name == "Gemini-v7.1"

        gpt = model_rankings[0]
        gemini = model_rankings[1]

        assert gpt.sample_count == 2
        assert gpt.average_match_count == 5.0
        assert gpt.prize_rate == 1.0
        assert gpt.rolling_matches[10] == 5.0
        assert gpt.rolling_prize_rates[10] == 1.0
        assert gpt.rank_score > gemini.rank_score
        assert 0.0 <= gpt.rank_score <= 1.0
        assert 0.0 <= gpt.confidence <= 1.0
        assert 0.0 <= gpt.stability <= 1.0
        assert gpt.trend in {"UP", "DOWN", "FLAT"}

        scenario_rankings = service.rank_strategies(
            strategy_type="scenario",
            history_limit=100,
        )

        assert len(scenario_rankings) == 2
        assert scenario_rankings[0].strategy_name == "gap"
        assert scenario_rankings[1].strategy_name == "pair"

        first_dataset = (
            service.ranking_repository.build_dataset(
                strategy_type="model",
                history_limit=100,
            )
        )
        second_dataset = (
            service.ranking_repository.build_dataset(
                strategy_type="model",
                history_limit=100,
            )
        )

        assert first_dataset is second_dataset
        assert first_dataset.strategy_count == 2
        assert first_dataset.history_point_count == 4
        assert first_dataset.revision == (8, 8)

        second_aggregation = service.run_strategy_aggregation(
            aggregated_at_kst=AGGREGATED_AT_KST,
        )

        assert second_aggregation.scanned == 0
        assert second_aggregation.created_events == 0
        assert second_aggregation.skipped_events == 0

        rankings_again = service.rank_strategies(
            strategy_type="model",
            history_limit=100,
        )

        assert rankings_again == model_rankings

        print("PASS: M6-004 strategy ranking")
        print(
            "model_rankings:",
            [item.as_dict() for item in model_rankings],
        )
        print(
            "scenario_rankings:",
            [item.as_dict() for item in scenario_rankings],
        )
        print(
            "dataset:",
            {
                "revision": first_dataset.revision,
                "strategy_count": first_dataset.strategy_count,
                "history_point_count": (
                    first_dataset.history_point_count
                ),
            },
        )


if __name__ == "__main__":
    main()
