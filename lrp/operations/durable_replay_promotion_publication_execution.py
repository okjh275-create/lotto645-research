from __future__ import annotations

from pathlib import Path

from lrp.operations.durable_replay_promotion_publication_request import (
    DurableReplayPromotionPublicationRequest,
)
from lrp.production.champion_registry_publisher import (
    ProductionChampionPublicationResult,
    ProductionChampionRegistryPublisher,
)


class DurableReplayPromotionPublicationExecutionService:
    def __init__(
        self,
        publisher: ProductionChampionRegistryPublisher | None = None,
    ) -> None:
        self._publisher = (
            publisher
            if publisher is not None
            else ProductionChampionRegistryPublisher()
        )

    def execute(
        self,
        request: DurableReplayPromotionPublicationRequest,
    ) -> ProductionChampionPublicationResult:
        if not isinstance(
            request,
            DurableReplayPromotionPublicationRequest,
        ):
            raise TypeError(
                "request must be DurableReplayPromotionPublicationRequest"
            )

        if request.action != "prepare_publish":
            raise ValueError("request.action must be prepare_publish")

        for name, value in (
            ("source_decision", request.source_decision),
            ("registry_root", request.registry_root),
        ):
            if not isinstance(value, (str, Path)):
                raise TypeError(f"{name} must be str or Path")

            if isinstance(value, str):
                if not value.strip():
                    raise ValueError(f"{name} must not be empty")
            elif str(value) in ("", "."):
                raise ValueError(f"{name} must not be empty")

        return self._publisher.publish(
            source_decision=request.source_decision,
            registry_root=request.registry_root,
        )