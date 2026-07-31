from __future__ import annotations

from dataclasses import replace

from lrp.evolution.contracts.execution import (
    EvolutionRunResult,
)
from lrp.evolution.contracts.pipeline import (
    EvolutionPipelineRequest,
)
from lrp.evolution.policies import (
    AdaptiveWeightPolicy,
)
from lrp.evolution.services.evolution_pipeline import (
    EvolutionPipeline,
)
from lrp.evolution.storage import (
    SnapshotNotFoundError,
    SnapshotRepository,
)


class EvolutionCoordinator:
    """Coordinate pipeline, policy, and snapshot persistence."""

    def __init__(
        self,
        *,
        pipeline: EvolutionPipeline,
        policy: AdaptiveWeightPolicy,
        repository: SnapshotRepository,
    ) -> None:
        if not isinstance(pipeline, EvolutionPipeline):
            raise TypeError(
                "pipeline must implement "
                "EvolutionPipeline"
            )

        if not isinstance(policy, AdaptiveWeightPolicy):
            raise TypeError(
                "policy must be an "
                "AdaptiveWeightPolicy"
            )

        if not isinstance(repository, SnapshotRepository):
            raise TypeError(
                "repository must be a "
                "SnapshotRepository"
            )

        self._pipeline = pipeline
        self._policy = policy
        self._repository = repository

    @property
    def pipeline(self) -> EvolutionPipeline:
        return self._pipeline

    @property
    def policy(self) -> AdaptiveWeightPolicy:
        return self._policy

    @property
    def repository(self) -> SnapshotRepository:
        return self._repository

    def execute(
        self,
        request: EvolutionPipelineRequest,
    ) -> EvolutionRunResult:
        if not isinstance(
            request,
            EvolutionPipelineRequest,
        ):
            raise TypeError(
                "request must be an "
                "EvolutionPipelineRequest"
            )

        previous = self._load_previous_profile()

        effective_request = replace(
            request,
            previous_profile=previous,
        )

        candidate = self.pipeline.calculate(
            effective_request
        )

        if candidate.revision != request.revision:
            raise ValueError(
                "pipeline candidate revision must "
                "match request revision"
            )

        decision = self.policy.evaluate(
            candidate,
            previous=previous,
        )

        if not decision.applied:
            return EvolutionRunResult(
                decision=decision,
                previous_profile=previous,
                snapshot=None,
            )

        snapshot = self.repository.save(
            decision.profile,
            saved_at=request.generated_at,
        )

        return EvolutionRunResult(
            decision=decision,
            previous_profile=previous,
            snapshot=snapshot,
        )

    def _load_previous_profile(self):
        try:
            snapshot = self.repository.load_latest()
        except SnapshotNotFoundError:
            return None

        return snapshot.profile