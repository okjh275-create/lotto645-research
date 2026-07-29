"""Append-only SQLite repository for learning records."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import json
import sqlite3
from typing import Any, Iterator

from .models import (
    PredictionRecord,
    ResultRecord,
    ReviewRecord,
)


_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS learning_predictions (
    prediction_id TEXT PRIMARY KEY,
    round_no INTEGER NOT NULL,
    set_id TEXT NOT NULL,
    numbers_json TEXT NOT NULL,
    score REAL NOT NULL,
    model_name TEXT NOT NULL,
    seed INTEGER NOT NULL,
    generated_at_kst TEXT NOT NULL,
    features_json TEXT NOT NULL,
    parameters_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS
idx_learning_predictions_round
ON learning_predictions(round_no);

CREATE TABLE IF NOT EXISTS learning_results (
    round_no INTEGER PRIMARY KEY,
    numbers_json TEXT NOT NULL,
    bonus INTEGER NOT NULL,
    recorded_at_kst TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS learning_reviews (
    prediction_id TEXT PRIMARY KEY,
    round_no INTEGER NOT NULL,
    matched_numbers_json TEXT NOT NULL,
    match_count INTEGER NOT NULL,
    bonus_matched INTEGER NOT NULL,
    prize_rank INTEGER,
    reviewed_at_kst TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    FOREIGN KEY(prediction_id)
        REFERENCES learning_predictions(prediction_id)
);

CREATE INDEX IF NOT EXISTS
idx_learning_reviews_round
ON learning_reviews(round_no);
"""


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


class LearningRepository:
    """Idempotent append-only learning data store."""

    def __init__(
        self,
        database_path: str | Path = "data/learning.db",
    ) -> None:
        self.database_path = Path(database_path)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        connection = sqlite3.connect(
            self.database_path,
            timeout=10.0,
        )
        connection.row_factory = sqlite3.Row
        connection.execute(
            "PRAGMA foreign_keys = ON"
        )

        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(_SCHEMA)

    def add_prediction(
        self,
        record: PredictionRecord,
    ) -> bool:
        self.initialize()

        values = (
            record.prediction_id,
            record.round_no,
            record.set_id,
            _json(record.numbers),
            float(record.score),
            record.model_name,
            int(record.seed),
            record.generated_at_kst,
            _json(dict(record.features)),
            _json(dict(record.parameters)),
        )

        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO learning_predictions (
                    prediction_id,
                    round_no,
                    set_id,
                    numbers_json,
                    score,
                    model_name,
                    seed,
                    generated_at_kst,
                    features_json,
                    parameters_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )

            if cursor.rowcount == 1:
                return True

            existing = connection.execute(
                """
                SELECT
                    prediction_id,
                    round_no,
                    set_id,
                    numbers_json,
                    score,
                    model_name,
                    seed,
                    generated_at_kst,
                    features_json,
                    parameters_json
                FROM learning_predictions
                WHERE prediction_id = ?
                """,
                (record.prediction_id,),
            ).fetchone()

            if existing is None or tuple(existing) != values:
                raise ValueError(
                    "prediction_id already exists "
                    "with different content"
                )

            return False

    def add_result(
        self,
        record: ResultRecord,
    ) -> bool:
        self.initialize()

        values = (
            record.round_no,
            _json(record.numbers),
            record.bonus,
            record.recorded_at_kst,
        )

        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO learning_results (
                    round_no,
                    numbers_json,
                    bonus,
                    recorded_at_kst
                ) VALUES (?, ?, ?, ?)
                """,
                values,
            )

            if cursor.rowcount == 1:
                return True

            existing = connection.execute(
                """
                SELECT
                    round_no,
                    numbers_json,
                    bonus,
                    recorded_at_kst
                FROM learning_results
                WHERE round_no = ?
                """,
                (record.round_no,),
            ).fetchone()

            if existing is None or tuple(existing) != values:
                raise ValueError(
                    "round result already exists "
                    "with different content"
                )

            return False

    def add_review(
        self,
        record: ReviewRecord,
    ) -> bool:
        self.initialize()

        values = (
            record.prediction_id,
            record.round_no,
            _json(record.matched_numbers),
            record.match_count,
            int(record.bonus_matched),
            record.prize_rank,
            record.reviewed_at_kst,
            _json(dict(record.metrics)),
        )

        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO learning_reviews (
                    prediction_id,
                    round_no,
                    matched_numbers_json,
                    match_count,
                    bonus_matched,
                    prize_rank,
                    reviewed_at_kst,
                    metrics_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )

            if cursor.rowcount == 1:
                return True

            existing = connection.execute(
                """
                SELECT
                    prediction_id,
                    round_no,
                    matched_numbers_json,
                    match_count,
                    bonus_matched,
                    prize_rank,
                    reviewed_at_kst,
                    metrics_json
                FROM learning_reviews
                WHERE prediction_id = ?
                """,
                (record.prediction_id,),
            ).fetchone()

            if existing is None or tuple(existing) != values:
                raise ValueError(
                    "prediction review already exists "
                    "with different content"
                )

            return False

    def counts(self) -> dict[str, int]:
        self.initialize()

        with self._connect() as connection:
            return {
                "predictions": int(
                    connection.execute(
                        "SELECT COUNT(*) "
                        "FROM learning_predictions"
                    ).fetchone()[0]
                ),
                "results": int(
                    connection.execute(
                        "SELECT COUNT(*) "
                        "FROM learning_results"
                    ).fetchone()[0]
                ),
                "reviews": int(
                    connection.execute(
                        "SELECT COUNT(*) "
                        "FROM learning_reviews"
                    ).fetchone()[0]
                ),
            }

    def pending_prediction_ids(
        self,
        *,
        round_no: int | None = None,
    ) -> tuple[str, ...]:
        self.initialize()

        query = """
            SELECT p.prediction_id
            FROM learning_predictions AS p
            LEFT JOIN learning_reviews AS r
                ON r.prediction_id = p.prediction_id
            WHERE r.prediction_id IS NULL
        """
        parameters: tuple[Any, ...] = ()

        if round_no is not None:
            query += " AND p.round_no = ?"
            parameters = (round_no,)

        query += " ORDER BY p.round_no, p.prediction_id"

        with self._connect() as connection:
            rows = connection.execute(
                query,
                parameters,
            ).fetchall()

        return tuple(
            str(row["prediction_id"])
            for row in rows
        )
