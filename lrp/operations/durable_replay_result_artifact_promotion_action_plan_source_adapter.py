from __future__ import annotations

from pathlib import Path

from lrp.operations.durable_replay_result_artifact_promotion_eligibility_source_adapter import (
    DurableReplayResultArtifactPromotionEligibilitySourceAdapter,
)
from lrp.operations.durable_replay_result_promotion_action_plan import (
    DurableReplayResultPromotionActionPlan,
    DurableReplayResultPromotionActionPlanService,
)


class DurableReplayResultArtifactPromotionActionPlanSourceAdapter:
    def __init__(
        self,
        source_adapter: DurableReplayResultArtifactPromotionEligibilitySourceAdapter | None = None,
        action_plan_service: DurableReplayResultPromotionActionPlanService | None = None,
    ) -> None:
        self._source_adapter = (
            source_adapter
            if source_adapter is not None
            else DurableReplayResultArtifactPromotionEligibilitySourceAdapter()
        )
        self._action_plan_service = (
            action_plan_service
            if action_plan_service is not None
            else DurableReplayResultPromotionActionPlanService()
        )

    def adapt(
        self,
        artifact_root: str | Path,
        end_round: int,
    ) -> DurableReplayResultPromotionActionPlan:
        eligibility = self._source_adapter.adapt(artifact_root, end_round)
        return self._action_plan_service.plan(eligibility)
