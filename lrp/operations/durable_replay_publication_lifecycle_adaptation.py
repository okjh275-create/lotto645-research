from __future__ import annotations

from lrp.operations.durable_replay_promotion_publication_execution import (
    DurableReplayPromotionPublicationExecutionService,
)
from lrp.operations.durable_replay_promotion_publication_request import (
    DurableReplayPromotionPublicationRequest,
)
from lrp.production.champion_registry_publisher import (
    ProductionChampionPublicationResult,
)
from lrp.production.production_lifecycle import (
    ProductionLifecycleStageResult,
)


class DurableReplayPublicationLifecycleAdaptationService:
    def __init__(
        self,
        execution_service: DurableReplayPromotionPublicationExecutionService
        | None = None,
    ) -> None:
        self._execution_service = (
            execution_service
            if execution_service is not None
            else DurableReplayPromotionPublicationExecutionService()
        )

    def adapt(
        self,
        request: DurableReplayPromotionPublicationRequest,
    ) -> ProductionLifecycleStageResult:
        if not isinstance(
            request,
            DurableReplayPromotionPublicationRequest,
        ):
            raise TypeError(
                "request must be DurableReplayPromotionPublicationRequest"
            )

        if request.action != "prepare_publish":
            raise ValueError("request.action must be prepare_publish")

        publication: ProductionChampionPublicationResult = (
            self._execution_service.execute(request)
        )

        detail = {
            "source_path": publication.source_path,
            "source_sha256": publication.source_sha256,
            "published_path": publication.published_path,
            "published_at_kst": publication.published_at_kst,
            "selected_model": publication.selected_model,
        }

        return ProductionLifecycleStageResult(
            name="publication",
            status="PASS",
            detail=detail,
        )