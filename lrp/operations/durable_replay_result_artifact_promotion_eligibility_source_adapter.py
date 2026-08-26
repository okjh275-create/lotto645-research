from __future__ import annotations

from pathlib import Path

from lrp.operations.durable_replay_result_artifact_comparison_source_adapter import (
    DurableReplayResultArtifactComparisonSourceAdapter,
)
from lrp.operations.durable_replay_result_promotion_eligibility import (
    DurableReplayResultPromotionEligibility,
    DurableReplayResultPromotionEligibilityService,
)


class DurableReplayResultArtifactPromotionEligibilitySourceAdapter:
    def __init__(
        self,
        source_adapter: DurableReplayResultArtifactComparisonSourceAdapter | None = None,
        eligibility_service: DurableReplayResultPromotionEligibilityService | None = None,
    ) -> None:
        self._source_adapter = (
            source_adapter
            if source_adapter is not None
            else DurableReplayResultArtifactComparisonSourceAdapter()
        )
        self._eligibility_service = (
            eligibility_service
            if eligibility_service is not None
            else DurableReplayResultPromotionEligibilityService()
        )

    def adapt(
        self,
        artifact_root: str | Path,
        end_round: int,
    ) -> DurableReplayResultPromotionEligibility:
        assessment = self._source_adapter.adapt(artifact_root, end_round)
        return self._eligibility_service.evaluate(assessment)
