from __future__ import annotations

from pathlib import Path

from lrp.operations.durable_replay_result_artifact_source_adapter import (
    DurableReplayResultArtifactSourceAdapter,
)
from lrp.operations.durable_replay_result_comparison_assessment import (
    DurableReplayResultComparisonAssessment,
    DurableReplayResultComparisonAssessmentService,
)
from lrp.operations.durable_replay_result_comparison_summary import (
    DurableReplayResultComparisonSummaryService,
)


class DurableReplayResultArtifactComparisonSourceAdapter:
    def __init__(
        self,
        source_adapter: DurableReplayResultArtifactSourceAdapter | None = None,
        summary_service: DurableReplayResultComparisonSummaryService | None = None,
        assessment_service: DurableReplayResultComparisonAssessmentService | None = None,
    ) -> None:
        self._source_adapter = (
            DurableReplayResultArtifactSourceAdapter()
            if source_adapter is None
            else source_adapter
        )
        self._summary_service = (
            DurableReplayResultComparisonSummaryService()
            if summary_service is None
            else summary_service
        )
        self._assessment_service = (
            DurableReplayResultComparisonAssessmentService()
            if assessment_service is None
            else assessment_service
        )

    def adapt(
        self,
        artifact_root: str | Path,
        end_round: int,
    ) -> DurableReplayResultComparisonAssessment:
        inspection = self._source_adapter.adapt(artifact_root, end_round)
        summary = self._summary_service.summarize(inspection)
        return self._assessment_service.assess(summary)
