from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Literal

from lrp.contracts import ContractError


HistoryFormat = Literal["csv", "json"]


def export_history(
    *,
    database: str | Path,
    output: str | Path,
    file_format: HistoryFormat,
) -> Path:
    database_path = Path(database)
    output_path = Path(output)

    if not database_path.is_file():
        raise FileNotFoundError(database_path)

    if file_format not in ("csv", "json"):
        raise ValueError(
            "file_format must be csv or json"
        )

    rows = _load_draw_rows(database_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if file_format == "json":
        _write_json(output_path, rows)
    else:
        _write_csv(output_path, rows)

    return output_path


def _load_draw_rows(
    database_path: Path,
) -> tuple[dict[str, int], ...]:
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row

    try:
        table = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name = 'draw_history'
            """
        ).fetchone()

        if table is None:
            raise ContractError(
                "draw_history table was not found"
            )

        raw_rows = connection.execute(
            """
            SELECT
                round,
                n1,
                n2,
                n3,
                n4,
                n5,
                n6,
                bonus
            FROM draw_history
            ORDER BY round
            """
        ).fetchall()
    finally:
        connection.close()

    if not raw_rows:
        raise ContractError(
            "draw_history contains no rows"
        )

    rows = tuple(
        {
            "round": int(row["round"]),
            "n1": int(row["n1"]),
            "n2": int(row["n2"]),
            "n3": int(row["n3"]),
            "n4": int(row["n4"]),
            "n5": int(row["n5"]),
            "n6": int(row["n6"]),
            "bonus": int(row["bonus"]),
        }
        for row in raw_rows
    )

    _validate_rows(rows)
    return rows


def _validate_rows(
    rows: tuple[dict[str, int], ...],
) -> None:
    rounds = [
        row["round"]
        for row in rows
    ]

    if len(rounds) != len(set(rounds)):
        raise ContractError(
            "draw_history contains duplicate rounds"
        )

    for row in rows:
        numbers = [
            row[f"n{index}"]
            for index in range(1, 7)
        ]
        bonus = row["bonus"]

        if any(
            number < 1 or number > 45
            for number in numbers
        ):
            raise ContractError(
                "draw numbers must be between 1 and 45"
            )

        if len(set(numbers)) != 6:
            raise ContractError(
                "draw numbers must be unique"
            )

        if bonus < 1 or bonus > 45:
            raise ContractError(
                "bonus must be between 1 and 45"
            )

        if bonus in numbers:
            raise ContractError(
                "bonus must differ from draw numbers"
            )


def _write_json(
    output_path: Path,
    rows: tuple[dict[str, int], ...],
) -> None:
    payload = {
        "draws": list(rows),
    }

    output_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_csv(
    output_path: Path,
    rows: tuple[dict[str, int], ...],
) -> None:
    fieldnames = [
        "round",
        "n1",
        "n2",
        "n3",
        "n4",
        "n5",
        "n6",
        "bonus",
    ]

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)
