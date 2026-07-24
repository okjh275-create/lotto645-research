"""Read-optimized data source for the M6-004 ranking engine.

This module deliberately avoids changing the append-only learning schema.
It composes the existing public LearningRepository statistics API with
bounded, read-only SQLite queries over strategy events.

The cache is revision-aware: a new strategy event changes the repository
revision and invalidates cached ranking datasets automatically.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import sqlite3
from typing import Iterator, Mapping, Sequence

from .ranking import StrategyPerformancePoint
from .repository import LearningRepository
from .strategy_stats import StrategyStatistics


StrategyKey = tuple[str, str]


@dataclass(frozen=True, slots=True)
class RankingDataset:
    """Immutable input bundle consumed by StrategyRankingEngine."""

    revision: tuple[int, int]
    statistics: tuple[StrategyStatistics, ...]
    histories: Mapping[
        StrategyKey,
        tuple[StrategyPerformancePoint, ...],
    ]

    @property
    def strategy_count(self) -> int:
        return len(self.statistics)

    @property
    def history_point_count(self) -> int:
        return sum(
            len(points)
            for points in self.histories.values()
        )


class RankingRepository:
    """Build and cache bounded ranking datasets from learning records."""

    def __init__(
        self,
        repository: LearningRepository,
    ) -> None:
        self.repository = repository
        self._cache_key: tuple[
            tuple[int, int],
            str | None,
            int,
        ] | None = None
        self._cache_value: RankingDataset | None = None

    def repository_revision(self) -> tuple[int, int]:
        """Return a cheap monotonic fingerprint for strategy events.

        The tuple is ``(event_count, maximum_rowid)``.  Because strategy
        events are append-only, either value changes whenever new ranking
        evidence is aggregated.
        """

        self.repository.initialize()

        with self._read_connection() as connection:
            row = connection.execute(
                """
                SELECT
                    COUNT(*) AS event_count,
                    COALESCE(MAX(rowid), 0) AS maximum_rowid
                FROM learning_strategy_events
                """
            ).fetchone()

        return (
            int(row["event_count"]),
            int(row["maximum_rowid"]),
        )

    def get_strategy_history(
        self,
        *,
        strategy_type: str,
        strategy_name: str,
        limit: int = 100,
    ) -> tuple[StrategyPerformancePoint, ...]:
        """Return the newest bounded history in chronological order."""

        normalized_type = self._required_text(
            strategy_type,
            field_name="strategy_type",
        )
        normalized_name = self._required_text(
            strategy_name,
            field_name="strategy_name",
        )

        if limit <= 0:
            raise ValueError("limit must be positive")

        self.repository.initialize()

        with self._read_connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    event.prediction_id,
                    prediction.round_no,
                    review.match_count,
                    prediction.score AS prediction_score,
                    review.prize_rank
                FROM learning_strategy_events AS event
                INNER JOIN learning_predictions AS prediction
                    ON prediction.prediction_id =
                       event.prediction_id
                INNER JOIN learning_reviews AS review
                    ON review.prediction_id =
                       event.prediction_id
                WHERE event.strategy_type = ?
                  AND event.strategy_name = ?
                ORDER BY
                    prediction.round_no DESC,
                    event.prediction_id DESC
                LIMIT ?
                """,
                (
                    normalized_type,
                    normalized_name,
                    int(limit),
                ),
            ).fetchall()

        # SQL reads newest first for an indexed bounded query.  The engine
        # receives chronological order to make rolling/trend semantics clear.
        return tuple(
            StrategyPerformancePoint(
                prediction_id=str(row["prediction_id"]),
                round_no=int(row["round_no"]),
                match_count=int(row["match_count"]),
                prediction_score=float(
                    row["prediction_score"]
                ),
                prize_rank=(
                    None
                    if row["prize_rank"] is None
                    else int(row["prize_rank"])
                ),
            )
            for row in reversed(rows)
        )

    def build_dataset(
        self,
        *,
        strategy_type: str | None = None,
        history_limit: int = 100,
    ) -> RankingDataset:
        """Build or reuse a complete ranking-engine input dataset."""

        if history_limit <= 0:
            raise ValueError("history_limit must be positive")

        normalized_type = (
            None
            if strategy_type is None
            else self._required_text(
                strategy_type,
                field_name="strategy_type",
            )
        )

        revision = self.repository_revision()
        cache_key = (
            revision,
            normalized_type,
            int(history_limit),
        )

        if (
            self._cache_key == cache_key
            and self._cache_value is not None
        ):
            return self._cache_value

        statistics = self.repository.get_strategy_statistics(
            strategy_type=normalized_type
        )

        histories: dict[
            StrategyKey,
            tuple[StrategyPerformancePoint, ...],
        ] = {}

        for item in statistics:
            key = (
                item.strategy_type,
                item.strategy_name,
            )
            histories[key] = self.get_strategy_history(
                strategy_type=item.strategy_type,
                strategy_name=item.strategy_name,
                limit=history_limit,
            )

        dataset = RankingDataset(
            revision=revision,
            statistics=statistics,
            histories=histories,
        )

        self._cache_key = cache_key
        self._cache_value = dataset
        return dataset

    def invalidate_cache(self) -> None:
        """Explicitly clear the local read cache."""

        self._cache_key = None
        self._cache_value = None

    @contextmanager
    def _read_connection(
        self,
    ) -> Iterator[sqlite3.Connection]:
        """Open and always close a short-lived read connection.

        ``sqlite3.Connection`` used directly as a context manager commits
        or rolls back a transaction, but it does not close the connection.
        Explicit cleanup is required so Windows can remove temporary
        database files immediately after a regression test.
        """

        connection = sqlite3.connect(
            self.repository.database_path,
            timeout=10.0,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")

        try:
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _required_text(
        value: str,
        *,
        field_name: str,
    ) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError(
                f"{field_name} must not be empty"
            )
        return normalized
