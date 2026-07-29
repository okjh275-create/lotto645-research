"""M6-003 strategy-statistics aggregation tests."""

from __future__ import annotations

from pathlib import Path
import tempfile
import time

from lrp.learning import (
    LearningRepository,
    LearningService,
    PredictionRecord,
    ResultRecord,
)


def prediction(
    prediction_id: str,
    set_id: str,
    model_name: str,
    numbers: tuple[int, ...],
    score: float,
    scenario: str,
) -> PredictionRecord:
    return PredictionRecord(
        prediction_id=prediction_id,
        round_no=1220,
        set_id=set_id,
        numbers=numbers,
        score=score,
        model_name=model_name,
        seed=20260723,
        generated_at_kst=(
            "2026-07-23T04:00:00+09:00"
        ),
        features={
            "gap_mix": 0.7,
            "pair_affinity": 0.6,
        },
        parameters={
            "temperature": 0.82,
            "scenario": scenario,
        },
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        database = Path(temporary) / "learning.db"
        repository = LearningRepository(database)
        service = LearningService(repository)

        repository.add_prediction(
            prediction(
                "1220:GPT-v3.3:S1",
                "S1",
                "GPT-v3.3",
                (3, 8, 14, 22, 35, 41),
                0.91,
                "gap",
            )
        )
        repository.add_prediction(
            prediction(
                "1220:GPT-v3.3:S2",
                "S2",
                "GPT-v3.3",
                (1, 6, 14, 21, 33, 44),
                0.83,
                "pair",
            )
        )
        repository.add_prediction(
            prediction(
                "1220:Gemini-v7.1:S1",
                "S1",
                "Gemini-v7.1",
                (3, 8, 14, 21, 35, 42),
                0.86,
                "gap",
            )
        )

        repository.add_result(
            ResultRecord(
                round_no=1220,
                numbers=(3, 8, 14, 22, 35, 41),
                bonus=9,
                recorded_at_kst=(
                    "2026-07-25T21:00:00+09:00"
                ),
            )
        )

        review = service.run_incremental_review(
            reviewed_at_kst=(
                "2026-07-25T21:01:00+09:00"
            )
        )

        assert review.created == 3

        started = time.perf_counter()

        first = service.run_strategy_aggregation(
            aggregated_at_kst=(
                "2026-07-25T21:02:00+09:00"
            )
        )

        aggregation_elapsed = (
            time.perf_counter() - started
        )

        assert first.scanned == 3
        assert first.created_events == 6
        assert first.skipped_events == 0

        second = service.run_strategy_aggregation(
            aggregated_at_kst=(
                "2026-07-25T21:03:00+09:00"
            )
        )

        assert second.scanned == 0
        assert second.created_events == 0
        assert second.skipped_events == 0

        model_stats = (
            service.get_strategy_statistics(
                strategy_type="model"
            )
        )

        assert len(model_stats) == 2

        gpt = next(
            item
            for item in model_stats
            if item.strategy_name == "GPT-v3.3"
        )
        
        assert gpt.sample_count == 2
        assert gpt.total_matches == 7
        assert round(
            gpt.average_match_count,
            6,
        ) == 3.5
        assert gpt.hit3_count == 0
        assert gpt.hit4_count == 0
        assert gpt.hit5_count == 0
        assert gpt.hit6_count == 1
        assert gpt.prize_count == 1

        assert gemini.sample_count == 1
        assert gemini.total_matches == 5
        assert gemini.hit5_count == 1
        assert gemini.prize_count == 1

        scenario_stats = (
            service.get_strategy_statistics(
                strategy_type="scenario"
            )
        )

        assert len(scenario_stats) == 2

        counts = repository.counts()

        assert counts == {
            "predictions": 3,
            "results": 1,
            "reviews": 3,
            "strategy_stats": 4,
            "strategy_events": 6,
        }

        assert aggregation_elapsed < 1.0

        print(
            "PASS: M6-003 strategy statistics"
        )
        print("first_run:", first.as_dict())
        print("second_run:", second.as_dict())
        print(
            "model_stats:",
            [item.as_dict() for item in model_stats],
        )
        print(
            "scenario_stats:",
            [item.as_dict() for item in scenario_stats],
        )
        print(
            "aggregation_elapsed:",
            round(aggregation_elapsed, 6),
        )


if __name__ == "__main__":
    main()
