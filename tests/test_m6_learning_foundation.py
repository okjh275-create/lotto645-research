"""M6-001 learning foundation tests."""

from __future__ import annotations

from pathlib import Path
import tempfile

from lrp.learning import (
    LearningRepository,
    PredictionRecord,
    ResultRecord,
    ReviewRecord,
)


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        database_path = (
            Path(temporary) / "learning.db"
        )

        repository = LearningRepository(
            database_path
        )

        prediction = PredictionRecord(
            prediction_id="1219:GPT-v3.3:S1",
            round_no=1219,
            set_id="S1",
            numbers=(3, 8, 14, 21, 34, 42),
            score=0.91,
            model_name="GPT-v3.3",
            seed=20260721,
            generated_at_kst=(
                "2026-07-21T14:32:00+09:00"
            ),
            features={
                "gap_mix": 0.73,
                "pair_affinity": 0.61,
            },
            parameters={
                "temperature": 0.82,
            },
        )

        result = ResultRecord(
            round_no=1219,
            numbers=(3, 8, 14, 22, 35, 41),
            bonus=9,
            recorded_at_kst=(
                "2026-07-25T21:00:00+09:00"
            ),
        )

        review = ReviewRecord(
            prediction_id=prediction.prediction_id,
            round_no=1219,
            matched_numbers=(3, 8, 14),
            match_count=3,
            bonus_matched=False,
            prize_rank=5,
            reviewed_at_kst=(
                "2026-07-25T21:01:00+09:00"
            ),
            metrics={
                "precision": 0.5,
            },
        )

        assert repository.add_prediction(
            prediction
        ) is True

        assert repository.add_prediction(
            prediction
        ) is False

        assert repository.pending_prediction_ids() == (
            prediction.prediction_id,
        )

        assert repository.add_result(result) is True
        assert repository.add_result(result) is False

        assert repository.add_review(review) is True
        assert repository.add_review(review) is False

        assert repository.pending_prediction_ids() == ()

        counts = repository.counts()

        assert counts["predictions"] == 1
        assert counts["results"] == 1
        assert counts["reviews"] == 1

        print(
            "PASS: M6-001 learning foundation"
        )


if __name__ == "__main__":
    main()
