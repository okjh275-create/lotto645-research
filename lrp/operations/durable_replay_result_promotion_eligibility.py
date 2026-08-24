from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from lrp.operations.durable_replay_result_comparison_assessment import (
    DurableReplayResultComparisonAssessment,
)


EligibilityRecommendation = Literal[
    "eligible",
    "ineligible",
    "insufficient_evidence",
]


@dataclass(frozen=True)
class DurableReplayResultPromotionEligibility:
    status: str
    round_count: int
    candidate_model_name: str
    baseline_model_name: str
    recommendation: EligibilityRecommendation
    candidate_advantage_count: int
    neutral_count: int
    baseline_advantage_count: int
    window: Mapping[str, object]


class DurableReplayResultPromotionEligibilityService:
    def evaluate(
        self,
        assessment: DurableReplayResultComparisonAssessment,
    ) -> DurableReplayResultPromotionEligibility:
        candidate = self._validated_count(
            "candidate_advantage_count",
            assessment.candidate_advantage_count,
        )
        neutral = self._validated_count(
            "neutral_count",
            assessment.neutral_count,
        )
        baseline = self._validated_count(
            "baseline_advantage_count",
            assessment.baseline_advantage_count,
        )

        if candidate + neutral + baseline != 9:
            raise ValueError("assessment aggregate counts must sum to 9")

        if not isinstance(assessment.window, Mapping):
            raise TypeError("assessment window must be a mapping")

        if candidate > baseline and candidate >= 2:
            recommendation: EligibilityRecommendation = "eligible"
        elif baseline > candidate:
            recommendation = "ineligible"
        else:
            recommendation = "insufficient_evidence"

        return DurableReplayResultPromotionEligibility(
            status=assessment.status,
            round_count=assessment.round_count,
            candidate_model_name=assessment.candidate_model_name,
            baseline_model_name=assessment.baseline_model_name,
            recommendation=recommendation,
            candidate_advantage_count=candidate,
            neutral_count=neutral,
            baseline_advantage_count=baseline,
            window=MappingProxyType(dict(assessment.window)),
        )

    @staticmethod
    def _validated_count(name: str, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an int and bool is not accepted")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
        return value