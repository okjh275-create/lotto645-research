"""Draw-history input and derived prediction context."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from lrp.contracts import ContractError


_NUMBER_COLUMNS = (
    ("n1", "n2", "n3", "n4", "n5", "n6"),
    (
        "number1",
        "number2",
        "number3",
        "number4",
        "number5",
        "number6",
    ),
    ("num1", "num2", "num3", "num4", "num5", "num6"),
)


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise ContractError(f"{field} must be an integer")

    try:
        result = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ContractError(
            f"{field} must be an integer"
        ) from exc

    return result


def _normalize_numbers(
    values: Iterable[object],
    *,
    field: str,
) -> tuple[int, int, int, int, int, int]:
    converted = tuple(
        sorted(_integer(value, field=field) for value in values)
    )

    if len(converted) != 6:
        raise ContractError(
            f"{field} must contain exactly six numbers"
        )

    if len(set(converted)) != 6:
        raise ContractError(
            f"{field} contains duplicate numbers"
        )

    invalid = tuple(
        number for number in converted if not 1 <= number <= 45
    )
    if invalid:
        raise ContractError(
            f"{field} contains invalid numbers: {invalid}"
        )

    return converted  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class HistoryRow:
    round_no: int
    numbers: tuple[int, int, int, int, int, int]
    bonus: int | None = None

    def __post_init__(self) -> None:
        if self.round_no <= 0:
            raise ContractError(
                "round number must be positive"
            )

        if self.bonus is not None:
            if not 1 <= self.bonus <= 45:
                raise ContractError(
                    "bonus number must be between 1 and 45"
                )
            if self.bonus in self.numbers:
                raise ContractError(
                    "bonus number must not duplicate main numbers"
                )


def _mapping_value(
    row: Mapping[str, Any],
    *names: str,
) -> Any:
    normalized = {
        str(key).strip().lower(): value
        for key, value in row.items()
    }

    for name in names:
        key = name.lower()
        if key in normalized:
            return normalized[key]

    return None


def _numbers_from_mapping(
    row: Mapping[str, Any],
) -> tuple[int, int, int, int, int, int]:
    direct = _mapping_value(
        row,
        "numbers",
        "nums",
        "winning_numbers",
    )

    if direct is not None:
        if isinstance(direct, str):
            cleaned = (
                direct.replace(",", " ")
                .replace("|", " ")
                .replace("-", " ")
            )
            values: Sequence[object] = cleaned.split()
        elif isinstance(direct, Sequence):
            values = direct
        else:
            raise ContractError(
                "numbers must be a sequence or delimited string"
            )

        return _normalize_numbers(values, field="numbers")

    normalized_keys = {
        str(key).strip().lower()
        for key in row
    }

    for columns in _NUMBER_COLUMNS:
        if all(column in normalized_keys for column in columns):
            normalized_row = {
                str(key).strip().lower(): value
                for key, value in row.items()
            }
            return _normalize_numbers(
                (normalized_row[column] for column in columns),
                field="numbers",
            )

    raise ContractError(
        "draw row does not contain six number columns"
    )


def _history_row_from_mapping(
    row: Mapping[str, Any],
) -> HistoryRow:
    raw_round = _mapping_value(
        row,
        "round",
        "round_no",
        "draw",
        "draw_no",
    )
    if raw_round is None:
        raise ContractError(
            "draw row is missing round"
        )

    raw_bonus = _mapping_value(
        row,
        "bonus",
        "bonus_number",
    )

    bonus = None
    if raw_bonus not in (None, ""):
        bonus = _integer(raw_bonus, field="bonus")

    return HistoryRow(
        round_no=_integer(raw_round, field="round"),
        numbers=_numbers_from_mapping(row),
        bonus=bonus,
    )


def _validate_history(
    rows: Iterable[HistoryRow],
) -> tuple[HistoryRow, ...]:
    ordered = tuple(
        sorted(rows, key=lambda item: item.round_no)
    )

    if not ordered:
        raise ContractError(
            "history must contain at least one draw"
        )

    rounds = [row.round_no for row in ordered]
    if len(rounds) != len(set(rounds)):
        raise ContractError(
            "history contains duplicate rounds"
        )

    return ordered


def load_history_csv(path: str | Path) -> tuple[HistoryRow, ...]:
    source = Path(path)

    if not source.is_file():
        raise FileNotFoundError(source)

    with source.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        reader = csv.DictReader(stream)

        if reader.fieldnames is None:
            raise ContractError(
                "CSV file has no header"
            )

        rows = tuple(
            _history_row_from_mapping(row)
            for row in reader
        )

    return _validate_history(rows)


def load_history_json(path: str | Path) -> tuple[HistoryRow, ...]:
    source = Path(path)

    if not source.is_file():
        raise FileNotFoundError(source)

    with source.open("r", encoding="utf-8-sig") as stream:
        payload = json.load(stream)

    if isinstance(payload, Mapping):
        payload = payload.get(
            "draws",
            payload.get("history"),
        )

    if not isinstance(payload, list):
        raise ContractError(
            "JSON history must be a list or contain a draws list"
        )

    rows = tuple(
        _history_row_from_mapping(row)
        for row in payload
        if isinstance(row, Mapping)
    )

    if len(rows) != len(payload):
        raise ContractError(
            "every JSON draw must be an object"
        )

    return _validate_history(rows)


def load_history(path: str | Path) -> tuple[HistoryRow, ...]:
    source = Path(path)
    suffix = source.suffix.lower()

    if suffix == ".csv":
        return load_history_csv(source)

    if suffix == ".json":
        return load_history_json(source)

    raise ContractError(
        "history file must use .csv or .json"
    )


def history_until_round(
    rows: Iterable[HistoryRow],
    *,
    target_round: int,
) -> tuple[HistoryRow, ...]:
    if target_round <= 1:
        raise ContractError(
            "target_round must be greater than one"
        )

    selected = tuple(
        row for row in rows
        if row.round_no < target_round
    )

    if not selected:
        raise ContractError(
            "history has no draws before target round"
        )

    return selected


def previous_numbers(
    rows: Sequence[HistoryRow],
) -> frozenset[int]:
    if not rows:
        raise ContractError("history is empty")

    latest = max(rows, key=lambda row: row.round_no)
    return frozenset(latest.numbers)


def long_gap_numbers(
    rows: Sequence[HistoryRow],
    *,
    recent_draw_count: int = 5,
) -> frozenset[int]:
    if recent_draw_count <= 0:
        raise ContractError(
            "recent_draw_count must be positive"
        )

    recent = tuple(
        sorted(
            rows,
            key=lambda row: row.round_no,
            reverse=True,
        )[:recent_draw_count]
    )

    appeared = {
        number
        for row in recent
        for number in row.numbers
    }

    result = frozenset(
        number for number in range(1, 46)
        if number not in appeared
    )

    if not result:
        raise ContractError(
            "no long-gap numbers were derived"
        )

    return result


def to_statistics_draws(
    rows: Iterable[HistoryRow],
    *,
    draw_type: type,
) -> tuple[object, ...]:
    return tuple(
        draw_type(
            round=row.round_no,
            numbers=row.numbers,
            bonus=row.bonus,
        )
        for row in rows
    )
