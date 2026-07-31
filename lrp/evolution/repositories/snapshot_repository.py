from __future__ import annotations

from abc import ABC, abstractmethod

from lrp.evolution.contracts.snapshot_schema import (
    LearningCycleSnapshot,
)


class SnapshotRepository(ABC):
    """Persistence contract for learning-cycle snapshots."""

    @abstractmethod
    def save(
        self,
        snapshot: LearningCycleSnapshot,
        *,
        overwrite: bool = False,
    ) -> None:
        """Persist one snapshot."""

    @abstractmethod
    def load(
        self,
        snapshot_id: str,
    ) -> LearningCycleSnapshot:
        """Load one snapshot by identifier."""

    @abstractmethod
    def exists(
        self,
        snapshot_id: str,
    ) -> bool:
        """Return whether the snapshot exists."""

    @abstractmethod
    def list_ids(self) -> tuple[str, ...]:
        """Return persisted snapshot identifiers."""

    @abstractmethod
    def delete(
        self,
        snapshot_id: str,
    ) -> bool:
        """Delete a snapshot and report whether it existed."""
