from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from lrp.evolution.contracts.learning_context import (
    LearningContext,
)
from lrp.evolution.contracts.persistent_learning import (
    PersistentLearningRunResult,
)
from lrp.evolution.contracts.reinforcement import (
    RewardFeedback,
)
from lrp.evolution.services.learning_cycle import (
    LearningCycle,
)
from lrp.evolution.services.persistent_learning_service import (
    PersistentLearningService,
)


class PersistentLearningRunner:
    """Run a learning cycle and persist its result."""

    def __init__(
        self,
        persistence_service: PersistentLearningService,
        learning_cycle: LearningCycle | None = None,
    ) -> None:
        if not isinstance(
            persistence_service,
            PersistentLearningService,
        ):
            raise TypeError(
                "persistence_service must be a "
                "PersistentLearningService"
            )

        if (
            learning_cycle is not None
            and not isinstance(
                learning_cycle,
                LearningCycle,
            )
        ):
            raise TypeError(
                "learning_cycle must be a "
                "LearningCycle"
            )

        self._persistence_service = (
            persistence_service
        )
        self._learning_cycle = (
            learning_cycle
            if learning_cycle is not None
            else LearningCycle()
        )

    @property
    def persistence_service(
        self,
    ) -> PersistentLearningService:
        return self._persistence_service

    @property
    def learning_cycle(self) -> LearningCycle:
        return self._learning_cycle

    def run(
        self,
        *,
        context: LearningContext,
        feedbacks: Iterable[RewardFeedback],
        snapshot_id: str,
        metadata: Mapping[str, Any] | None = None,
        overwrite: bool = False,
    ) -> PersistentLearningRunResult:
        if not isinstance(overwrite, bool):
            raise TypeError(
                "overwrite must be a boolean"
            )

        learning_result = (
            self._learning_cycle.run(
                context=context,
                feedbacks=feedbacks,
            )
        )

        snapshot = (
            self._persistence_service.persist(
                learning_result,
                snapshot_id=snapshot_id,
                metadata=metadata,
                overwrite=overwrite,
            )
        )

        return PersistentLearningRunResult(
            learning_result=learning_result,
            snapshot=snapshot,
        )
