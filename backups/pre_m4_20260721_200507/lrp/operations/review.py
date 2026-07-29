"""Prediction review and hit-analysis services."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo
import json

from lrp.contracts import ContractError

_KST = ZoneInfo("Asia/Seoul")


def _numbers(values: Iterable[object], *, field: str) -> tuple[int, ...]:
    try:
        result = tuple(sorted(int(value) for value in values))
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{field} must contain integers") from exc
    if len(result) != 6 or len(set(result)) != 6:
        raise ContractError(f"{field} must contain six unique numbers")
    if any(number < 1 or number > 45 for number in result):
        raise ContractError(f"{field} contains an invalid lotto number")
    return result


def _load_prediction(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ContractError("prediction JSON must be an object")
    return payload


def prize_rank(main_hits: int, bonus_hit: bool) -> str:
    if main_hits == 6:
        return "1"
    if main_hits == 5 and bonus_hit:
        return "2"
    if main_hits == 5:
        return "3"
    if main_hits == 4:
        return "4"
    if main_hits == 3:
        return "5"
    return "none"


def review_prediction(
    prediction: str | Path | Mapping[str, Any],
    *,
    winning_numbers: Sequence[int],
    bonus: int | None = None,
) -> dict[str, Any]:
    payload = _load_prediction(prediction)
    winning = _numbers(winning_numbers, field="winning_numbers")
    if bonus is not None:
        bonus = int(bonus)
        if bonus < 1 or bonus > 45 or bonus in winning:
            raise ContractError("bonus must be 1..45 and outside winning numbers")

    rows: list[dict[str, Any]] = []
    for item in payload.get("sets", []):
        numbers = _numbers(item.get("numbers", ()), field="set numbers")
        matched = tuple(number for number in numbers if number in winning)
        bonus_hit = bonus is not None and bonus in numbers
        rows.append(
            {
                "id": str(item.get("id", "")),
                "numbers": list(numbers),
                "matched_numbers": list(matched),
                "main_hits": len(matched),
                "bonus_hit": bonus_hit,
                "rank": prize_rank(len(matched), bonus_hit),
                "score": float(item.get("score", 0.0)),
            }
        )

    distribution = {str(hit): 0 for hit in range(7)}
    for row in rows:
        distribution[str(row["main_hits"])] += 1

    best_hits = max((row["main_hits"] for row in rows), default=0)
    best_ids = [row["id"] for row in rows if row["main_hits"] == best_hits]
    practical = set(str(value) for value in payload.get("top5_practical", []))
    practical_rows = [row for row in rows if row["id"] in practical]

    return {
        "schema_version": "1.0",
        "artifact_type": "lotto645_review",
        "round": int(payload["round"]),
        "seed": int(payload["seed"]),
        "reviewed_at_kst": datetime.now(_KST).isoformat(timespec="seconds"),
        "winning_numbers": list(winning),
        "bonus": bonus,
        "summary": {
            "set_count": len(rows),
            "best_main_hits": best_hits,
            "best_set_ids": best_ids,
            "hit_distribution": distribution,
            "practical_best_hits": max(
                (row["main_hits"] for row in practical_rows), default=0
            ),
            "winning_rank_counts": {
                rank: sum(row["rank"] == rank for row in rows)
                for rank in ("1", "2", "3", "4", "5")
            },
        },
        "sets": rows,
        "prediction_metadata": dict(payload.get("metadata", {})),
    }
