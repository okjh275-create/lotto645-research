from __future__ import annotations

from pathlib import Path

from lrp.operations.durable_replay_result_artifact_consumer import (
    DurableReplayResultArtifactConsumerRequest,
)
from lrp.operations.durable_replay_result_artifact_inspection import (
    DurableReplayResultArtifactInspection,
    DurableReplayResultArtifactInspectionService,
)


class DurableReplayResultArtifactSourceAdapter:
    def __init__(
        self,
        inspection_service: DurableReplayResultArtifactInspectionService | None = None,
    ) -> None:
        self._inspection_service = (
            inspection_service
            if inspection_service is not None
            else DurableReplayResultArtifactInspectionService()
        )

    def adapt(
        self,
        artifact_root: str | Path,
        end_round: int,
    ) -> DurableReplayResultArtifactInspection:
        request = DurableReplayResultArtifactConsumerRequest(
            artifact_root=artifact_root,
            end_round=end_round,
        )
        return self._inspection_service.inspect(request=request)
