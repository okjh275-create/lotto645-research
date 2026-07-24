"""M6-002 automatic review and incremental service tests."""

from __future__ import annotations

from pathlib import Path
import tempfile

from lrp.learning import (
    LearningRepository,
    LearningService,
    PredictionRecord,
    ResultRecord,
    determine_prize_rank,
    evaluate_prediction,
)


def make_prediction(
    prediction_id: str,
    set_id: str,
    numbers: tuple[int, ...],
) -> PredictionRecord:
    return PredictionRecord(
        prediction_id=prediction_id,
        round_no=1219,
        set_id=set_id,
        numbers=numbers,
        score=0.85,
        model_name="GPT-v3.3",
        seed=20260723,
        generated_at_kst=(
            "2026-07-23T03:00:00+09:00"
        ),
        features={
            "gap_mix": 0.71,
            "pair_affinity": 0.62,
        },
        parameters={
            "temperature": 0.82,
        },
    )


def main() -> None:
    assert determine_prize_rank(
        match_count=6,
        bonus_matched=False,
    ) == 1

    assert determine_prize_rank(
        match_count=5,
        bonus_matched=True,
    ) == 2

    assert determine_prize_rank(
        match_count=5,
        bonus_matched=False,
    ) == 3

    assert determine_prize_rank(
        match_count=4,
        bonus_matched=False,
    ) == 4

    assert determine_prize_rank(
        match_count=3,
        bonus_matched=False,
    ) == 5

    assert determine_prize_rank(
        match_count=2,
        bonus_matched=False,
    ) is None

    with tempfile.TemporaryDirectory() as temporary:
        database_path = (
            Path(temporary) / "learning.db"
        )

        repository = LearningRepository(
            database_path
        )
        service = LearningService(repository)

        prediction_1 = make_prediction(
            "1219:GPT-v3.3:S1",
            "S1",
            (3, 8, 14, 21, 34, 42),
        )
        prediction_2 = make_prediction(
            "1219:GPT-v3.3:S2",
            "S2",
            (3, 8, 14, 22, 35, 41),
        )

        result = ResultRecord(
            round_no=1219,
            numbers=(3, 8, 14, 22, 35, 41),
            bonus=9,
            recorded_at_kst=(
                "2026-07-25T21:00:00+09:00"
            ),
        )

        repository.add_prediction(prediction_1)
        repository.add_prediction(prediction_2)

        before_result = (
            service.run_incremental_review(
                reviewed_at_kst=(
                    "2026-07-25T21:01:00+09:00"
                )
            )
        )

        assert before_result.scanned == 0
        assert before_result.created == 0

        repository.add_result(result)

        direct_review = evaluate_prediction(
            prediction_1,
            result,
            reviewed_at_kst=(
                "2026-07-25T21:01:00+09:00"
            ),
        )

        assert direct_review.matched_numbers == (
            3,
            8,
            14,
        )
        assert direct_review.match_count == 3
        assert direct_review.prize_rank == 5
        assert direct_review.metrics["miss_count"] == 3

        first_run = service.run_incremental_review(
            reviewed_at_kst=(
                "2026-07-25T21:01:00+09:00"
            )
        )

        assert first_run.scanned == 2
        assert first_run.created == 2
        assert first_run.skipped == 0

        second_run = service.run_incremental_review(
            reviewed_at_kst=(
                "2026-07-25T21:02:00+09:00"
            )
        )

        assert second_run.scanned == 0
        assert second_run.created == 0
        assert second_run.skipped == 0

        counts = repository.counts()

        assert counts["predictions"] == 2
        assert counts["results"] == 1
        assert counts["reviews"] == 2

        assert repository.pending_prediction_ids() == ()

        print(
            "PASS: M6-002 automatic incremental review"
        )
        print(
            "first_run:",
            first_run.as_dict(),
        )
        print(
            "second_run:",
            second_run.as_dict(),
        )


if __name__ == "__main__":
    main()
