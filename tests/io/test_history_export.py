from __future__ import annotations

import sqlite3
from pathlib import Path

from lrp.io import load_history
from lrp.io.history_export import (
    export_history,
)


def make_database(path: Path) -> None:
    connection = sqlite3.connect(path)

    try:
        connection.execute(
            """
            CREATE TABLE draw_history (
                round INTEGER PRIMARY KEY,
                n1 INTEGER NOT NULL,
                n2 INTEGER NOT NULL,
                n3 INTEGER NOT NULL,
                n4 INTEGER NOT NULL,
                n5 INTEGER NOT NULL,
                n6 INTEGER NOT NULL,
                bonus INTEGER NOT NULL
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO draw_history
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    601,
                    2,
                    16,
                    19,
                    31,
                    34,
                    35,
                    37,
                ),
                (
                    602,
                    13,
                    14,
                    22,
                    27,
                    30,
                    38,
                    2,
                ),
            ],
        )
        connection.commit()
    finally:
        connection.close()


def test_json_export_round_trips(
    tmp_path: Path,
) -> None:
    database = tmp_path / "lotto.db"
    output = tmp_path / "history.json"

    make_database(database)

    result = export_history(
        database=database,
        output=output,
        file_format="json",
    )

    history = load_history(result)

    assert result == output
    assert len(history) == 2
    assert history[0].round_no == 601
    assert history[1].round_no == 602


def test_csv_export_round_trips(
    tmp_path: Path,
) -> None:
    database = tmp_path / "lotto.db"
    output = tmp_path / "history.csv"

    make_database(database)

    export_history(
        database=database,
        output=output,
        file_format="csv",
    )

    history = load_history(output)

    assert len(history) == 2
    assert history[0].numbers == (
        2,
        16,
        19,
        31,
        34,
        35,
    )
    assert history[1].bonus == 2
