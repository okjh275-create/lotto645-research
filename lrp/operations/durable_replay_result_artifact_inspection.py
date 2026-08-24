"""Read-only inspection of persisted durable replay evaluation results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from lrp.operations.durable_replay_result_artifact_consumer import (
    DurableReplayResultArtifactConsumer,
    DurableReplayResultArtifactConsumerRequest,
)


@dataclass(frozen=True)
class DurableReplayResultArtifactInspection:
    status: str
    round_count: int
    candidate_model_name: str
    baseline_model_name: str
    evaluation: Mapping[str, object]


class DurableReplayResultArtifactInspectionService:
    def inspect(
        self,
        request: DurableReplayResultArtifactConsumerRequest,
    ) -> DurableReplayResultArtifactInspection:
        payload = DurableReplayResultArtifactConsumer().consume(
            request=request
        )

        status = payload["status"]
        round_count = payload["round_count"]
        candidate_model_name = payload["candidate_model_name"]
        baseline_model_name = payload["baseline_model_name"]
        evaluation = payload["evaluation"]

        if not isinstance(status, str):
            raise TypeError("status must be str")
        if not isinstance(round_count, int):
            raise TypeError("round_count must be int")
        if not isinstance(candidate_model_name, str):
            raise TypeError("candidate_model_name must be str")
        if not isinstance(baseline_model_name, str):
            raise TypeError("baseline_model_name must be str")
        if not isinstance(evaluation, Mapping):
            raise TypeError("evaluation must be a mapping")

        return DurableReplayResultArtifactInspection(
            status=status,
            round_count=round_count,
            candidate_model_name=candidate_model_name,
            baseline_model_name=baseline_model_name,
            evaluation=MappingProxyType(dict(evaluation)),
        )
