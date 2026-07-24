"""Prediction evaluation against official Lotto 6/45 results."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from .models import (
    PredictionRecord,
    ResultRecord,
    ReviewRecord,
)


_KST = ZoneInfo("Asia/Seoul")


def determine_prize_rank(
    *,
    match_count: int,
    bonus_matched: bool,
) -> int | None:
    """Return the Korean Lotto 6/45 prize rank."""

    if match_count == 6:
        return 1

    if match_count == 5 and bonus_matched:
        return 2

    if match_count == 5:
        return 3

    if match_count == 4:
        return 4

    if match_count == 3:
        return 5

    return None


def evaluate_prediction(
    prediction: PredictionRecord,
    result: ResultRecord,
    *,
    reviewed_at_kst: str | None = None,
) -> ReviewRecord:
    """Evaluate one prediction without mutating either record."""

    if prediction.round_no != result.round_no:
        raise ValueError(
            "prediction and result round numbers must match"
        )

    prediction_numbers = set(prediction.numbers)
    winning_numbers = set(result.numbers)

    matched_numbers = tuple(
        sorted(prediction_numbers & winning_numbers)
    )
    match_count = len(matched_numbers)
    bonus_matched = result.bonus in prediction_numbers

    timestamp = reviewed_at_kst
    if timestamp is None:
        timestamp = datetime.now(_KST).isoformat(
            timespec="seconds"
        )

    metrics = {
        "hit_rate": round(match_count / 6.0, 6),
        "precision": round(match_count / 6.0, 6),
        "recall": round(match_count / 6.0, 6),
        "miss_count": 6 - match_count,
        "prediction_score": float(prediction.score),
        "model_name": prediction.model_name,
        "set_id": prediction.set_id,
    }

    return ReviewRecord(
        prediction_id=prediction.prediction_id,
        round_no=prediction.round_no,
        matched_numbers=matched_numbers,
        match_count=match_count,
        bonus_matched=bonus_matched,
        prize_rank=determine_prize_rank(
            match_count=match_count,
            bonus_matched=bonus_matched,
        ),
        reviewed_at_kst=timestamp,
        metrics=metrics,
    )
