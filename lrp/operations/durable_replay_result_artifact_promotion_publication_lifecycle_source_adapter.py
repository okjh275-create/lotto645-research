from __future__ import annotations

from pathlib import Path

from lrp.operations.durable_replay_publication_lifecycle_entrypoint import (
    DurableReplayPublicationLifecycleEntrypoint,
)
from lrp.operations.durable_replay_result_artifact_promotion_publication_request_source_adapter import (
    DurableReplayResultArtifactPromotionPublicationRequestSourceAdapter,
)
from lrp.production.production_lifecycle import ProductionLifecycleStageResult


class DurableReplayResultArtifactPromotionPublicationLifecycleSourceAdapter:
    def __init__(
        self,
        source_adapter: DurableReplayResultArtifactPromotionPublicationRequestSourceAdapter,
        lifecycle_entrypoint: DurableReplayPublicationLifecycleEntrypoint,
    ) -> None:
        self._source_adapter = source_adapter
        self._lifecycle_entrypoint = lifecycle_entrypoint

    def run(
        self,
        artifact_root: str | Path,
        end_round: int,
        *,
        source_decision: str | Path,
        registry_root: str | Path,
    ) -> ProductionLifecycleStageResult:
        request = self._source_adapter.adapt(
            artifact_root,
            end_round,
            source_decision=source_decision,
            registry_root=registry_root,
        )
        return self._lifecycle_entrypoint.run(request)