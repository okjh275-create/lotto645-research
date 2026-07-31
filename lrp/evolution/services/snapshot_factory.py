from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any

from lrp.evolution.contracts.learning_cycle import (
    LearningCycleResult,
)
from lrp.evolution.contracts.snapshot_schema import (
    LearningCycleSnapshot,
)


class SnapshotFactory:
    """Create persisted snapshots from learning-cycle results."""

    def __init__(
        self,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if clock is not None and not callable(clock):
            raise TypeError(
                "clock must be callable"
            )

        self._clock = (
            clock
            if clock is not None
            else self._utc_now
        )

    def create(
        self,
        result: LearningCycleResult,
        *,
        snapshot_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> LearningCycleSnapshot:
        if not isinstance(
            result,
            LearningCycleResult,
        ):
            raise TypeError(
                "result must be a LearningCycleResult"
            )

        normalized_snapshot_id = (
            self._normalize_snapshot_id(
                snapshot_id
            )
        )
        normalized_metadata = (
            self._normalize_metadata(metadata)
        )
        created_at_utc = (
            self._normalize_created_at(
                self._clock()
            )
        )

        return LearningCycleSnapshot(
            snapshot_id=normalized_snapshot_id,
            result=result,
            created_at_utc=created_at_utc,
            metadata=normalized_metadata,
        )

    @staticmethod
    def _normalize_snapshot_id(
        snapshot_id: str,
    ) -> str:
        if not isinstance(snapshot_id, str):
            raise TypeError(
                "snapshot_id must be a string"
            )

        normalized = snapshot_id.strip()

        if not normalized:
            raise ValueError(
                "snapshot_id must not be empty"
            )

        return normalized

    @staticmethod
    def _normalize_metadata(
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if metadata is None:
            return {}

        if not isinstance(metadata, Mapping):
            raise TypeError(
                "metadata must be a mapping"
            )

        return dict(metadata)

    @staticmethod
    def _normalize_created_at(
        created_at: datetime,
    ) -> datetime:
        if not isinstance(created_at, datetime):
            raise TypeError(
                "clock must return a datetime"
            )

        if created_at.tzinfo is None:
            raise ValueError(
                "clock must return a timezone-aware "
                "datetime"
            )

        return created_at.astimezone(
            timezone.utc
        )

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)
