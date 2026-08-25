from __future__ import annotations

from lrp.operations.durable_replay_promotion_publication_request import (
    DurableReplayPromotionPublicationRequest,
)
from lrp.operations.durable_replay_publication_lifecycle_adaptation import (
    DurableReplayPublicationLifecycleAdaptationService,
)
from lrp.production.production_lifecycle import (
    ProductionLifecycleStageResult,
)


class DurableReplayPublicationLifecycleEntrypoint:
    def __init__(
        self,
        adaptation_service: DurableReplayPublicationLifecycleAdaptationService,
    ) -> None:
        self._adaptation_service = adaptation_service

    def run(
        self,
        request: DurableReplayPromotionPublicationRequest,
    ) -> ProductionLifecycleStageResult:
        if not isinstance(request, DurableReplayPromotionPublicationRequest):
            raise TypeError(
                "request must be DurableReplayPromotionPublicationRequest"
            )

        if request.action != "prepare_publish":
            raise ValueError("request action must be prepare_publish")

        result = self._adaptation_service.adapt(request)

        if not isinstance(result, ProductionLifecycleStageResult):
            raise TypeError(
                "adaptation result must be ProductionLifecycleStageResult"
            )

        return result