from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lrp.contracts.exceptions import ContractError
from lrp.evaluation.topk_replay_evaluation import (
    TopKReplayEvaluationResult,
)
from lrp.operations.durable_replay_artifact_discovery import (
    DurableReplayArtifactDiscoveryRequest,
    DurableReplayArtifactDiscoveryService,
    DurableReplayArtifactSelector,
)
from lrp.operations.durable_replay_execution import (
    DurableReplayExecutionRequest,
    DurableReplayExecutionService,
)


@dataclass(frozen=True)
class DurableReplayCompositionRequest:
    artifact_root: str | Path
    history_path: str | Path
    window_name: str
    start_round: int
    end_round: int
    candidate_selectors: tuple[
        DurableReplayArtifactSelector,
        ...,
    ]
    baseline_selectors: tuple[
        DurableReplayArtifactSelector,
        ...,
    ]


class DurableReplayCompositionService:
    def execute(
        self,
        *,
        request: DurableReplayCompositionRequest,
    ) -> TopKReplayEvaluationResult:
        if not isinstance(
            request,
            DurableReplayCompositionRequest,
        ):
            raise ContractError(
                "request must be "
                "DurableReplayCompositionRequest"
            )

        discovery_request = (
            DurableReplayArtifactDiscoveryRequest(
                artifact_root=request.artifact_root,
                candidate_selectors=(
                    request.candidate_selectors
                ),
                baseline_selectors=(
                    request.baseline_selectors
                ),
            )
        )

        (
            candidate_sources,
            baseline_sources,
        ) = (
            DurableReplayArtifactDiscoveryService()
            .discover(
                request=discovery_request
            )
        )

        execution_request = (
            DurableReplayExecutionRequest(
                history_path=request.history_path,
                window_name=request.window_name,
                start_round=request.start_round,
                end_round=request.end_round,
                candidate_sources=candidate_sources,
                baseline_sources=baseline_sources,
            )
        )

        return (
            DurableReplayExecutionService()
            .execute(
                request=execution_request
            )
        )
