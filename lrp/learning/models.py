"""Immutable learning-domain records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


def _normalize_numbers(
    values: Sequence[int],
    *,
    field_name: str,
    expected_count: int = 6,
) -> tuple[int, ...]:
    try:
        numbers = tuple(sorted(int(value) for value in values))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field_name} must contain integers"
        ) from exc

    if len(numbers) != expected_count:
        raise ValueError(
            f"{field_name} must contain exactly "
            f"{expected_count} numbers"
        )

    if len(set(numbers)) != expected_count:
        raise ValueError(
            f"{field_name} must not contain duplicates"
        )

    if any(number < 1 or number > 45 for number in numbers):
        raise ValueError(
            f"{field_name} numbers must be between 1 and 45"
        )

    return numbers


def _validate_round(round_no: int) -> int:
    if isinstance(round_no, bool) or not isinstance(round_no, int):
        raise ValueError("round_no must be an integer")

    if round_no <= 0:
        raise ValueError("round_no must be positive")

    return round_no


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    """One generated combination and its generation context."""

    prediction_id: str
    round_no: int
    set_id: str
    numbers: tuple[int, ...]
    score: float
    model_name: str
    seed: int
    generated_at_kst: str
    features: Mapping[str, Any] = field(
        default_factory=dict
    )
    parameters: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.prediction_id.strip():
            raise ValueError(
                "prediction_id must not be empty"
            )

        _validate_round(self.round_no)

        if not self.set_id.strip():
            raise ValueError("set_id must not be empty")

        object.__setattr__(
            self,
            "numbers",
            _normalize_numbers(
                self.numbers,
                field_name="numbers",
            ),
        )

        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError(
                "score must be between 0 and 1"
            )

        if not self.model_name.strip():
            raise ValueError(
                "model_name must not be empty"
            )


@dataclass(frozen=True, slots=True)
class ResultRecord:
    """Official winning result for one round."""

    round_no: int
    numbers: tuple[int, ...]
    bonus: int
    recorded_at_kst: str

    def __post_init__(self) -> None:
        _validate_round(self.round_no)

        normalized = _normalize_numbers(
            self.numbers,
            field_name="numbers",
        )
        object.__setattr__(
            self,
            "numbers",
            normalized,
        )

        if (
            isinstance(self.bonus, bool)
            or not isinstance(self.bonus, int)
            or self.bonus < 1
            or self.bonus > 45
        ):
            raise ValueError(
                "bonus must be an integer between 1 and 45"
            )

        if self.bonus in normalized:
            raise ValueError(
                "bonus must not duplicate a winning number"
            )


@dataclass(frozen=True, slots=True)
class ReviewRecord:
    """Evaluation of one prediction against one result."""

    prediction_id: str
    round_no: int
    matched_numbers: tuple[int, ...]
    match_count: int
    bonus_matched: bool
    prize_rank: int | None
    reviewed_at_kst: str
    metrics: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.prediction_id.strip():
            raise ValueError(
                "prediction_id must not be empty"
            )

        _validate_round(self.round_no)

        normalized = tuple(
            sorted(int(value) for value in self.matched_numbers)
        )

        if len(set(normalized)) != len(normalized):
            raise ValueError(
                "matched_numbers must not contain duplicates"
            )

        if any(number < 1 or number > 45 for number in normalized):
            raise ValueError(
                "matched_numbers must be between 1 and 45"
            )

        object.__setattr__(
            self,
            "matched_numbers",
            normalized,
        )

        if self.match_count != len(normalized):
            raise ValueError(
                "match_count must equal matched_numbers length"
            )

        if self.match_count < 0 or self.match_count > 6:
            raise ValueError(
                "match_count must be between 0 and 6"
            )

        if self.prize_rank is not None and (
            self.prize_rank < 1 or self.prize_rank > 5
        ):
            raise ValueError(
                "prize_rank must be 1 to 5 or None"
            )
