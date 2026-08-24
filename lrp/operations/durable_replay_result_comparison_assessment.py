"""Deterministic sign-only assessment of replay comparison summaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Literal

from lrp.operations.durable_replay_result_comparison_summary import (
    DurableReplayResultComparisonSummary,
)

AssessmentLabel = Literal[
    "candidate_advantage",
    "neutral",
    "baseline_advantage",
]


@dataclass(frozen=True)
class DurableReplayResultComparisonAssessment:
    status: str
    round_count: int
    candidate_model_name: str
    baseline_model_name: str
    top3_baseline_delta_mean_best_hits_assessment: AssessmentLabel
    top5_baseline_delta_mean_best_hits_assessment: AssessmentLabel
    top10_baseline_delta_mean_best_hits_assessment: AssessmentLabel
    top3_baseline_delta_3plus_rate_assessment: AssessmentLabel
    top5_baseline_delta_3plus_rate_assessment: AssessmentLabel
    top10_baseline_delta_3plus_rate_assessment: AssessmentLabel
    top3_baseline_delta_4plus_rate_assessment: AssessmentLabel
    top5_baseline_delta_4plus_rate_assessment: AssessmentLabel
    top10_baseline_delta_4plus_rate_assessment: AssessmentLabel
    candidate_advantage_count: int
    neutral_count: int
    baseline_advantage_count: int
    window: Mapping[str, object]


def _classify(value: float) -> AssessmentLabel:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("comparison delta must be a real numeric value")
    if math.isnan(float(value)):
        raise ValueError("comparison delta must not be NaN")
    if value > 0:
        return "candidate_advantage"
    if value < 0:
        return "baseline_advantage"
    return "neutral"


class DurableReplayResultComparisonAssessmentService:
    def assess(
        self,
        summary: DurableReplayResultComparisonSummary,
    ) -> DurableReplayResultComparisonAssessment:
        if not isinstance(summary.window, Mapping):
            raise TypeError("summary window must be a mapping")

        labels = (
            _classify(summary.top3_baseline_delta_mean_best_hits),
            _classify(summary.top5_baseline_delta_mean_best_hits),
            _classify(summary.top10_baseline_delta_mean_best_hits),
            _classify(summary.top3_baseline_delta_3plus_rate),
            _classify(summary.top5_baseline_delta_3plus_rate),
            _classify(summary.top10_baseline_delta_3plus_rate),
            _classify(summary.top3_baseline_delta_4plus_rate),
            _classify(summary.top5_baseline_delta_4plus_rate),
            _classify(summary.top10_baseline_delta_4plus_rate),
        )

        return DurableReplayResultComparisonAssessment(
            status=summary.status,
            round_count=summary.round_count,
            candidate_model_name=summary.candidate_model_name,
            baseline_model_name=summary.baseline_model_name,
            top3_baseline_delta_mean_best_hits_assessment=labels[0],
            top5_baseline_delta_mean_best_hits_assessment=labels[1],
            top10_baseline_delta_mean_best_hits_assessment=labels[2],
            top3_baseline_delta_3plus_rate_assessment=labels[3],
            top5_baseline_delta_3plus_rate_assessment=labels[4],
            top10_baseline_delta_3plus_rate_assessment=labels[5],
            top3_baseline_delta_4plus_rate_assessment=labels[6],
            top5_baseline_delta_4plus_rate_assessment=labels[7],
            top10_baseline_delta_4plus_rate_assessment=labels[8],
            candidate_advantage_count=labels.count("candidate_advantage"),
            neutral_count=labels.count("neutral"),
            baseline_advantage_count=labels.count("baseline_advantage"),
            window=MappingProxyType(dict(summary.window)),
        )
