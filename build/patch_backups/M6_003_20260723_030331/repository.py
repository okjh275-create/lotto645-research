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


def _decode_json(value: str) -> Any:
    return json.loads(value)


class LearningRepository:
    """Idempotent append-only learning data store."""

    def __init__(
        self,
        database_path: str | Path = "data/learning.db",
    ) -> None:
        self.database_path = Path(database_path)
        self._initialized = False

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
        if self._initialized:
            return

        with self._connect() as connection:
            connection.executescript(_SCHEMA)

        self._initialized = True

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

    def get_prediction(
        self,
        prediction_id: str,
    ) -> PredictionRecord | None:
        self.initialize()

        with self._connect() as connection:
            row = connection.execute(
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
                (prediction_id,),
            ).fetchone()

        if row is None:
            return None

        return PredictionRecord(
            prediction_id=str(row["prediction_id"]),
            round_no=int(row["round_no"]),
            set_id=str(row["set_id"]),
            numbers=tuple(
                int(value)
                for value in _decode_json(
                    row["numbers_json"]
                )
            ),
            score=float(row["score"]),
            model_name=str(row["model_name"]),
            seed=int(row["seed"]),
            generated_at_kst=str(
                row["generated_at_kst"]
            ),
            features=dict(
                _decode_json(row["features_json"])
            ),
            parameters=dict(
                _decode_json(row["parameters_json"])
            ),
        )

    def get_result(
        self,
        round_no: int,
    ) -> ResultRecord | None:
        self.initialize()

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    round_no,
                    numbers_json,
                    bonus,
                    recorded_at_kst
                FROM learning_results
                WHERE round_no = ?
                """,
                (round_no,),
            ).fetchone()

        if row is None:
            return None

        return ResultRecord(
            round_no=int(row["round_no"]),
            numbers=tuple(
                int(value)
                for value in _decode_json(
                    row["numbers_json"]
                )
            ),
            bonus=int(row["bonus"]),
            recorded_at_kst=str(
                row["recorded_at_kst"]
            ),
        )

    def pending_predictions_with_results(
        self,
        *,
        round_no: int | None = None,
        limit: int | None = None,
    ) -> tuple[PredictionRecord, ...]:
        """Return only predictions ready for incremental review."""

        self.initialize()

        query = """
            SELECT
                p.prediction_id,
                p.round_no,
                p.set_id,
                p.numbers_json,
                p.score,
                p.model_name,
                p.seed,
                p.generated_at_kst,
                p.features_json,
                p.parameters_json
            FROM learning_predictions AS p
            INNER JOIN learning_results AS official
                ON official.round_no = p.round_no
            LEFT JOIN learning_reviews AS review
                ON review.prediction_id = p.prediction_id
            WHERE review.prediction_id IS NULL
        """
        parameters: list[Any] = []

        if round_no is not None:
            query += " AND p.round_no = ?"
            parameters.append(round_no)

        query += """
            ORDER BY
                p.round_no ASC,
                p.prediction_id ASC
        """

        if limit is not None:
            if limit <= 0:
                raise ValueError("limit must be positive")

            query += " LIMIT ?"
            parameters.append(limit)

        with self._connect() as connection:
            rows = connection.execute(
                query,
                tuple(parameters),
            ).fetchall()

        return tuple(
            PredictionRecord(
                prediction_id=str(row["prediction_id"]),
                round_no=int(row["round_no"]),
                set_id=str(row["set_id"]),
                numbers=tuple(
                    int(value)
                    for value in _decode_json(
                        row["numbers_json"]
                    )
                ),
                score=float(row["score"]),
                model_name=str(row["model_name"]),
                seed=int(row["seed"]),
                generated_at_kst=str(
                    row["generated_at_kst"]
                ),
                features=dict(
                    _decode_json(row["features_json"])
                ),
                parameters=dict(
                    _decode_json(row["parameters_json"])
                ),
            )
            for row in rows
        )

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
