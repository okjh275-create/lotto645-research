"""Deterministic read-only summary of durable replay comparison results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from lrp.operations.durable_replay_result_artifact_inspection import (
    DurableReplayResultArtifactInspection,
)


@dataclass(frozen=True)
class DurableReplayResultComparisonSummary:
    status: str
    round_count: int
    candidate_model_name: str
    baseline_model_name: str
    top3_baseline_delta_mean_best_hits: float
    top5_baseline_delta_mean_best_hits: float
    top10_baseline_delta_mean_best_hits: float
    top3_baseline_delta_3plus_rate: float
    top5_baseline_delta_3plus_rate: float
    top10_baseline_delta_3plus_rate: float
    top3_baseline_delta_4plus_rate: float
    top5_baseline_delta_4plus_rate: float
    top10_baseline_delta_4plus_rate: float
    window: Mapping[str, object]


class DurableReplayResultComparisonSummaryService:
    def summarize(
        self,
        inspection: DurableReplayResultArtifactInspection,
    ) -> DurableReplayResultComparisonSummary:
        evaluation = inspection.evaluation
        top3 = evaluation["top3"]
        top5 = evaluation["top5"]
        top10 = evaluation["top10"]
        window = evaluation["window"]

        if not isinstance(top3, Mapping):
            raise TypeError("top3 must be a mapping")
        if not isinstance(top5, Mapping):
            raise TypeError("top5 must be a mapping")
        if not isinstance(top10, Mapping):
            raise TypeError("top10 must be a mapping")
        if not isinstance(window, Mapping):
            raise TypeError("window must be a mapping")

        return DurableReplayResultComparisonSummary(
            status=inspection.status,
            round_count=inspection.round_count,
            candidate_model_name=inspection.candidate_model_name,
            baseline_model_name=inspection.baseline_model_name,
            top3_baseline_delta_mean_best_hits=float(
                top3["baseline_delta_mean_best_hits"]
            ),
            top5_baseline_delta_mean_best_hits=float(
                top5["baseline_delta_mean_best_hits"]
            ),
            top10_baseline_delta_mean_best_hits=float(
                top10["baseline_delta_mean_best_hits"]
            ),
            top3_baseline_delta_3plus_rate=float(
                top3["baseline_delta_3plus_rate"]
            ),
            top5_baseline_delta_3plus_rate=float(
                top5["baseline_delta_3plus_rate"]
            ),
            top10_baseline_delta_3plus_rate=float(
                top10["baseline_delta_3plus_rate"]
            ),
            top3_baseline_delta_4plus_rate=float(
                top3["baseline_delta_4plus_rate"]
            ),
            top5_baseline_delta_4plus_rate=float(
                top5["baseline_delta_4plus_rate"]
            ),
            top10_baseline_delta_4plus_rate=float(
                top10["baseline_delta_4plus_rate"]
            ),
            window=MappingProxyType(dict(window)),
        )
