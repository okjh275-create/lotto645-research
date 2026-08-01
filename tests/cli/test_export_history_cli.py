from __future__ import annotations

import sqlite3
from pathlib import Path

from lrp.cli import main
from lrp.io import load_history


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
        connection.execute(
            """
            INSERT INTO draw_history
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
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
        )
        connection.commit()
    finally:
        connection.close()


def test_export_history_command(
    tmp_path: Path,
    capsys,
) -> None:
    database = tmp_path / "lotto.db"
    output = tmp_path / "history.json"

    make_database(database)

    exit_code = main(
        [
            "export-history",
            "--db",
            str(database),
            "--format",
            "json",
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"status": "PASS"' in captured.out
    assert output.is_file()

    history = load_history(output)

    assert len(history) == 1
    assert history[0].round_no == 601


def test_export_history_missing_database(
    tmp_path: Path,
    capsys,
) -> None:
    output = tmp_path / "history.json"

    exit_code = main(
        [
            "export-history",
            "--db",
            str(tmp_path / "missing.db"),
            "--format",
            "json",
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 1
    assert '"status": "ERROR"' in captured.err
    assert "FileNotFoundError" in captured.err
