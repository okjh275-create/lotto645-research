"""Regression tests for Project E E-005A Performance Analyzer."""

from __future__ import annotations

from pathlib import Path
import tempfile

from lrp.learning import (
    LearningRepository,
    LearningService,
    PerformanceAnalyzer,
    PredictionRecord,
    RankingConfig,
    ResultRecord,
    StrategyPerformanceReport,
    StrategyPerformanceSummary,
)


GENERATED_AT_KST = "2026-07-27T20:00:00+09:00"
RECORDED_AT_KST = "2026-07-27T21:00:00+09:00"
REVIEWED_AT_KST = "2026-07-27T21:01:00+09:00"
AGGREGATED_AT_KST = "2026-07-27T21:02:00+09:00"
REPORT_AT_KST = "2026-07-27T21:03:00+09:00"


def make_prediction(
    *,
    round_no: int,
    model_name: str,
    set_id: str,
    numbers: tuple[int, ...],
    score: float,
    scenario: str,
) -> PredictionRecord:
    return PredictionRecord(
        prediction_id=f"{round_no}:{model_name}:{set_id}",
        round_no=round_no,
        set_id=set_id,
        numbers=numbers,
        score=score,
        model_name=model_name,
        seed=20260727,
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
    predictions: tuple[PredictionRecord, ...],
) -> None:
    for prediction in predictions:
        assert repository.add_prediction(prediction) is True

    bonus = next(
        number
        for number in range(1, 46)
        if number not in winning_numbers
    )

    assert repository.add_result(
        ResultRecord(
            round_no=round_no,
            numbers=winning_numbers,
            bonus=bonus,
            recorded_at_kst=RECORDED_AT_KST,
        )
    ) is True


def test_empty_report() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        repository = LearningRepository(
            Path(temporary) / "learning.db"
        )
        service = LearningService(repository)
        analyzer = PerformanceAnalyzer(
            service.ranking_repository
        )

        report = analyzer.analyze(
            strategy_type="model",
            history_limit=10,
            generated_at_kst=REPORT_AT_KST,
        )

        assert isinstance(
            report,
            StrategyPerformanceReport,
        )
        assert report.revision == (0, 0)
        assert report.strategy_count == 0
        assert report.total_samples == 0
        assert report.summaries == ()
        assert report.metadata["read_only"] is True


def test_performance_report() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        repository = LearningRepository(
            Path(temporary) / "learning.db"
        )
        service = LearningService(repository)

        rounds = (
            (
                1220,
                (1, 2, 3, 4, 5, 6),
                (1, 2, 20, 21, 22, 23),
                (1, 2, 3, 20, 21, 22),
            ),
            (
                1221,
                (10, 11, 12, 13, 14, 15),
                (10, 11, 12, 30, 31, 32),
                (10, 11, 30, 31, 32, 33),
            ),
            (
                1222,
                (20, 21, 22, 23, 24, 25),
                (20, 21, 22, 23, 30, 31),
                (20, 21, 32, 33, 34, 35),
            ),
            (
                1223,
                (30, 31, 32, 33, 34, 35),
                (30, 31, 32, 33, 34, 40),
                (30, 31, 42, 43, 44, 45),
            ),
        )

        for (
            round_no,
            winning,
            gpt_numbers,
            gemini_numbers,
        ) in rounds:
            add_round(
                repository=repository,
                round_no=round_no,
                winning_numbers=winning,
                predictions=(
                    make_prediction(
                        round_no=round_no,
                        model_name="GPT-v3.3",
                        set_id="S1",
                        numbers=gpt_numbers,
                        score=0.90,
                        scenario="gap",
                    ),
                    make_prediction(
                        round_no=round_no,
                        model_name="Gemini-v7.1",
                        set_id="S1",
                        numbers=gemini_numbers,
                        score=0.82,
                        scenario="pair",
                    ),
                ),
            )

        review = service.run_incremental_review(
            reviewed_at_kst=REVIEWED_AT_KST
        )
        assert review.scanned == 8
        assert review.created == 8

        aggregation = service.run_strategy_aggregation(
            aggregated_at_kst=AGGREGATED_AT_KST
        )
        assert aggregation.scanned == 8
        assert aggregation.created_events == 16

        before_counts = repository.counts()

        analyzer = PerformanceAnalyzer(
            service.ranking_repository,
            ranking_config=RankingConfig(
                windows=(2, 4),
                trend_short_window=2,
                trend_long_window=4,
                trend_threshold=0.20,
                confidence_scale=4.0,
            ),
        )

        report = analyzer.analyze(
            strategy_type="model",
            history_limit=3,
            generated_at_kst=REPORT_AT_KST,
        )

        after_counts = repository.counts()
        assert before_counts == after_counts

        assert isinstance(
            report,
            StrategyPerformanceReport,
        )
        assert report.revision == (16, 16)
        assert report.strategy_type == "model"
        assert report.history_limit == 3
        assert report.generated_at_kst == REPORT_AT_KST
        assert report.strategy_count == 2
        assert report.total_samples == 8
        assert report.total_history_points == 6

        gpt = report.get("model", "GPT-v3.3")
        gemini = report.get("model", "Gemini-v7.1")

        assert isinstance(
            gpt,
            StrategyPerformanceSummary,
        )
        assert isinstance(
            gemini,
            StrategyPerformanceSummary,
        )

        assert gpt.sample_count == 4
        assert gpt.history_count == 3
        assert [
            point.round_no
            for point in gpt.history
        ] == [1221, 1222, 1223]
        assert [
            point.match_count
            for point in gpt.history
        ] == [3, 4, 5]
        assert gpt.best_match_count == 5
        assert gpt.worst_match_count == 3
        assert gpt.average_match_count == 3.5
        assert gpt.average_prediction_score == 0.90
        assert gpt.hit3_plus_rate == 0.75
        assert gpt.prize_rate == 0.75
        assert gpt.trend == "UP"
        assert gpt.recent_gain > 0.0
        assert gpt.rank_position == 1

        assert gemini.sample_count == 4
        assert gemini.history_count == 3
        assert [
            point.round_no
            for point in gemini.history
        ] == [1221, 1222, 1223]
        assert [
            point.match_count
            for point in gemini.history
        ] == [2, 2, 2]
        assert gemini.best_match_count == 2
        assert gemini.worst_match_count == 2
        assert gemini.trend == "FLAT"
        assert gemini.rank_position == 2

        assert gpt.rank_score > gemini.rank_score
        assert 0.0 <= gpt.confidence <= 1.0
        assert 0.0 <= gpt.stability <= 1.0

        payload = report.as_dict(
            include_history=True
        )
        assert payload["strategy_count"] == 2
        assert payload["revision"] == [16, 16]
        assert len(
            payload["summaries"][0]["history"]
        ) == 3

        compact = report.as_dict(
            include_history=False
        )
        assert "history" not in compact["summaries"][0]

        second = analyzer.analyze(
            strategy_type="model",
            history_limit=3,
            generated_at_kst=REPORT_AT_KST,
        )
        assert second == report


def test_all_strategy_types() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        repository = LearningRepository(
            Path(temporary) / "learning.db"
        )
        service = LearningService(repository)

        add_round(
            repository=repository,
            round_no=1230,
            winning_numbers=(1, 2, 3, 4, 5, 6),
            predictions=(
                make_prediction(
                    round_no=1230,
                    model_name="GPT-v3.3",
                    set_id="S1",
                    numbers=(1, 2, 3, 4, 20, 21),
                    score=0.88,
                    scenario="gap",
                ),
            ),
        )

        service.run_incremental_review(
            reviewed_at_kst=REVIEWED_AT_KST
        )
        service.run_strategy_aggregation(
            aggregated_at_kst=AGGREGATED_AT_KST
        )

        analyzer = PerformanceAnalyzer(
            service.ranking_repository
        )
        report = analyzer.analyze(
            history_limit=10,
            generated_at_kst=REPORT_AT_KST,
        )

        assert report.strategy_type is None
        assert report.strategy_count == 2
        assert {
            item.key
            for item in report.summaries
        } == {
            ("model", "GPT-v3.3"),
            ("scenario", "gap"),
        }


def main() -> None:
    test_empty_report()
    test_performance_report()
    test_all_strategy_types()

    print(
        "PASS: Project E E-005A performance analyzer"
    )
    print("read_only_analysis: PASS")
    print("repository_revision: PASS")
    print("bounded_history: PASS")
    print("chronological_history: PASS")
    print("confidence_stability_trend: PASS")
    print("model_scenario_reports: PASS")
    print("deterministic_report: PASS")
    print("public_api_compatibility: PASS")


if __name__ == "__main__":
    main()
