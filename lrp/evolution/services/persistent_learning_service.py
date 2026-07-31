from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from lrp.evolution.contracts.learning_cycle import (
    LearningCycleResult,
)
from lrp.evolution.contracts.snapshot_schema import (
    LearningCycleSnapshot,
)
from lrp.evolution.repositories.snapshot_repository import (
    SnapshotRepository,
)
from lrp.evolution.services.snapshot_factory import (
    SnapshotFactory,
)


class PersistentLearningService:
    """Persist and restore completed learning-cycle results."""

    def __init__(
        self,
        repository: SnapshotRepository,
        snapshot_factory: SnapshotFactory | None = None,
    ) -> None:
        if not isinstance(
            repository,
            SnapshotRepository,
        ):
            raise TypeError(
                "repository must be a "
                "SnapshotRepository"
            )

        if (
            snapshot_factory is not None
            and not isinstance(
                snapshot_factory,
                SnapshotFactory,
            )
        ):
            raise TypeError(
                "snapshot_factory must be a "
                "SnapshotFactory"
            )

        self._repository = repository
        self._snapshot_factory = (
            snapshot_factory
            if snapshot_factory is not None
            else SnapshotFactory()
        )

    @property
    def repository(self) -> SnapshotRepository:
        return self._repository

    @property
    def snapshot_factory(self) -> SnapshotFactory:
        return self._snapshot_factory

    def persist(
        self,
        result: LearningCycleResult,
        *,
        snapshot_id: str,
        metadata: Mapping[str, Any] | None = None,
        overwrite: bool = False,
    ) -> LearningCycleSnapshot:
        if not isinstance(overwrite, bool):
            raise TypeError(
                "overwrite must be a boolean"
            )

        snapshot = self._snapshot_factory.create(
            result,
            snapshot_id=snapshot_id,
            metadata=metadata,
        )

        self._repository.save(
            snapshot,
            overwrite=overwrite,
        )

        return snapshot

    def load(
        self,
        snapshot_id: str,
    ) -> LearningCycleSnapshot:
        return self._repository.load(
            snapshot_id
        )

    def exists(
        self,
        snapshot_id: str,
    ) -> bool:
        return self._repository.exists(
            snapshot_id
        )

    def list_ids(self) -> tuple[str, ...]:
        return self._repository.list_ids()

    def delete(
        self,
        snapshot_id: str,
    ) -> bool:
        return self._repository.delete(
            snapshot_id
        )
