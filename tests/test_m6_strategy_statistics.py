"""M6-003 strategy-statistics aggregation regression test."""

from __future__ import annotations

from pathlib import Path
import tempfile
import time

from lrp.learning import (
    LearningRepository,
    LearningService,
    PredictionRecord,
    ResultRecord,
    StrategyStatistics,
)


ROUND_NO = 1220
GENERATED_AT_KST = "2026-07-23T04:00:00+09:00"
RESULT_RECORDED_AT_KST = "2026-07-25T21:00:00+09:00"
REVIEWED_AT_KST = "2026-07-25T21:01:00+09:00"
AGGREGATED_AT_KST = "2026-07-25T21:02:00+09:00"


def make_prediction(
    *,
    prediction_id: str,
    set_id: str,
    model_name: str,
    numbers: tuple[int, ...],
    score: float,
    scenario: str,
) -> PredictionRecord:
    """Create deterministic prediction data for this test."""

    return PredictionRecord(
        prediction_id=prediction_id,
        round_no=ROUND_NO,
        set_id=set_id,
        numbers=numbers,
        score=score,
        model_name=model_name,
        seed=20260723,
        generated_at_kst=GENERATED_AT_KST,
        features={
            "gap_mix": 0.7,
            "pair_affinity": 0.6,
        },
        parameters={
            "temperature": 0.82,
            "scenario": scenario,
        },
    )


def statistics_by_name(
    statistics: tuple[StrategyStatistics, ...],
) -> dict[str, StrategyStatistics]:
    """Index statistics by strategy name and reject duplicates."""

    indexed: dict[str, StrategyStatistics] = {}

    for item in statistics:
        if item.strategy_name in indexed:
            raise AssertionError(
                "duplicate strategy name in statistics: "
                f"{item.strategy_name}"
            )

        indexed[item.strategy_name] = item

    return indexed


def assert_repository_counts(
    repository: LearningRepository,
    *,
    predictions: int,
    results: int,
    reviews: int,
    strategy_stats: int,
    strategy_events: int,
) -> None:
    """Check expected counters without requiring exact API equality."""

    counts = repository.counts()

    assert counts["predictions"] == predictions
    assert counts["results"] == results
    assert counts["reviews"] == reviews
    assert counts["strategy_stats"] == strategy_stats
    assert counts["strategy_events"] == strategy_events


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        database_path = (
            Path(temporary_directory) / "learning.db"
        )

        repository = LearningRepository(database_path)
        service = LearningService(repository)

        # Exact 6-number match.
        assert repository.add_prediction(
            make_prediction(
                prediction_id="1220:GPT-v3.3:S1",
                set_id="S1",
                model_name="GPT-v3.3",
                numbers=(3, 8, 14, 22, 35, 41),
                score=0.91,
                scenario="gap",
            )
        )

        # One-number match: 14.
        assert repository.add_prediction(
            make_prediction(
                prediction_id="1220:GPT-v3.3:S2",
                set_id="S2",
                model_name="GPT-v3.3",
                numbers=(1, 6, 14, 21, 33, 44),
                score=0.83,
                scenario="pair",
            )
        )

        # Four-number match: 3, 8, 14, 35.
        assert repository.add_prediction(
            make_prediction(
                prediction_id="1220:Gemini-v7.1:S1",
                set_id="S1",
                model_name="Gemini-v7.1",
                numbers=(3, 8, 14, 21, 35, 42),
                score=0.86,
                scenario="gap",
            )
        )

        assert repository.add_result(
            ResultRecord(
                round_no=ROUND_NO,
                numbers=(3, 8, 14, 22, 35, 41),
                bonus=9,
                recorded_at_kst=RESULT_RECORDED_AT_KST,
            )
        )

        review_summary = service.run_incremental_review(
            reviewed_at_kst=REVIEWED_AT_KST
        )

        assert review_summary.scanned == 3
        assert review_summary.created == 3
        assert review_summary.skipped == 0

        started = time.perf_counter()

        first_run = service.run_strategy_aggregation(
            aggregated_at_kst=AGGREGATED_AT_KST
        )

        measured_elapsed_seconds = (
            time.perf_counter() - started
        )

        assert first_run.scanned == 3
        assert first_run.created_events == 6
        assert first_run.skipped_events == 0

        # Incremental checkpoint verification:
        # the same reviewed predictions must not be aggregated again.
        second_run = service.run_strategy_aggregation(
            aggregated_at_kst=(
                "2026-07-25T21:03:00+09:00"
            )
        )

        assert second_run.scanned == 0
        assert second_run.created_events == 0
        assert second_run.skipped_events == 0

        model_statistics = (
            service.get_strategy_statistics(
                strategy_type="model"
            )
        )

        assert len(model_statistics) == 2

        models = statistics_by_name(model_statistics)

        assert set(models) == {
            "GPT-v3.3",
            "Gemini-v7.1",
        }

        gpt = models["GPT-v3.3"]

        # GPT: 6 matches + 1 match = 7 total.
        assert gpt.sample_count == 2
        assert gpt.total_matches == 7
        assert round(
            gpt.average_match_count,
            6,
        ) == 3.5
        assert round(
            gpt.average_prediction_score,
            6,
        ) == 0.87
        assert gpt.hit3_count == 0
        assert gpt.hit4_count == 0
        assert gpt.hit5_count == 0
        assert gpt.hit6_count == 1
        assert gpt.prize_count == 1
        assert round(gpt.hit3_plus_rate, 6) == 0.5
        assert round(gpt.prize_rate, 6) == 0.5

        gemini = models["Gemini-v7.1"]

        # Gemini: 3, 8, 14, 35 = four matches.
        assert gemini.sample_count == 1
        assert gemini.total_matches == 4
        assert round(
            gemini.average_match_count,
            6,
        ) == 4.0
        assert round(
            gemini.average_prediction_score,
            6,
        ) == 0.86
        assert gemini.hit3_count == 0
        assert gemini.hit4_count == 1
        assert gemini.hit5_count == 0
        assert gemini.hit6_count == 0
        assert gemini.prize_count == 1
        assert round(
            gemini.hit3_plus_rate,
            6,
        ) == 1.0
        assert round(
            gemini.prize_rate,
            6,
        ) == 1.0

        scenario_statistics = (
            service.get_strategy_statistics(
                strategy_type="scenario"
            )
        )

        assert len(scenario_statistics) == 2

        scenarios = statistics_by_name(
            scenario_statistics
        )

        assert set(scenarios) == {
            "gap",
            "pair",
        }

        gap = scenarios["gap"]

        # Gap scenario: GPT exact hit 6 + Gemini hit 4.
        assert gap.sample_count == 2
        assert gap.total_matches == 10
        assert round(
            gap.average_match_count,
            6,
        ) == 5.0
        assert round(
            gap.average_prediction_score,
            6,
        ) == 0.885
        assert gap.hit3_count == 0
        assert gap.hit4_count == 1
        assert gap.hit5_count == 0
        assert gap.hit6_count == 1
        assert gap.prize_count == 2
        assert round(gap.hit3_plus_rate, 6) == 1.0
        assert round(gap.prize_rate, 6) == 1.0

        pair = scenarios["pair"]

        # Pair scenario: one match only.
        assert pair.sample_count == 1
        assert pair.total_matches == 1
        assert round(
            pair.average_match_count,
            6,
        ) == 1.0
        assert round(
            pair.average_prediction_score,
            6,
        ) == 0.83
        assert pair.hit3_count == 0
        assert pair.hit4_count == 0
        assert pair.hit5_count == 0
        assert pair.hit6_count == 0
        assert pair.prize_count == 0
        assert round(pair.hit3_plus_rate, 6) == 0.0
        assert round(pair.prize_rate, 6) == 0.0

        assert_repository_counts(
            repository,
            predictions=3,
            results=1,
            reviews=3,
            strategy_stats=4,
            strategy_events=6,
        )

        # Generous regression threshold for Windows/CI variance.
        assert measured_elapsed_seconds < 1.0

        print("PASS: M6-003 strategy statistics")
        print("review_run:", review_summary.as_dict())
        print("first_run:", first_run.as_dict())
        print("second_run:", second_run.as_dict())
        print(
            "model_stats:",
            [
                item.as_dict()
                for item in model_statistics
            ],
        )
        print(
            "scenario_stats:",
            [
                item.as_dict()
                for item in scenario_statistics
            ],
        )
        print(
            "aggregation_elapsed:",
            round(measured_elapsed_seconds, 6),
        )


if __name__ == "__main__":
    main()
