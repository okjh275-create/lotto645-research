from __future__ import annotations

from pathlib import Path

from lrp.operations.durable_replay_promotion_publication_request import (
    DurableReplayPromotionPublicationRequest,
    DurableReplayPromotionPublicationRequestService,
)
from lrp.operations.durable_replay_result_artifact_promotion_action_plan_source_adapter import (
    DurableReplayResultArtifactPromotionActionPlanSourceAdapter,
)


class DurableReplayResultArtifactPromotionPublicationRequestSourceAdapter:
    def __init__(
        self,
        source_adapter: DurableReplayResultArtifactPromotionActionPlanSourceAdapter | None = None,
        publication_request_service: DurableReplayPromotionPublicationRequestService | None = None,
    ) -> None:
        self._source_adapter = (
            source_adapter
            if source_adapter is not None
            else DurableReplayResultArtifactPromotionActionPlanSourceAdapter()
        )
        self._publication_request_service = (
            publication_request_service
            if publication_request_service is not None
            else DurableReplayPromotionPublicationRequestService()
        )

    def adapt(
        self,
        artifact_root: str | Path,
        end_round: int,
        *,
        source_decision: str | Path,
        registry_root: str | Path,
    ) -> DurableReplayPromotionPublicationRequest:
        action_plan = self._source_adapter.adapt(
            artifact_root,
            end_round,
        )
        return self._publication_request_service.build(
            action_plan=action_plan,
            source_decision=source_decision,
            registry_root=registry_root,
        )
