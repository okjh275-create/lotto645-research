from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from lrp.operations.durable_replay_result_promotion_eligibility import (
    DurableReplayResultPromotionEligibility,
)

PromotionAction = Literal[
    "prepare_publish",
    "hold",
    "block",
]


@dataclass(frozen=True)
class DurableReplayResultPromotionActionPlan:
    status: str
    round_count: int
    candidate_model_name: str
    baseline_model_name: str
    recommendation: str
    action: PromotionAction
    window: Mapping[str, object]


class DurableReplayResultPromotionActionPlanService:
    def plan(
        self,
        eligibility: DurableReplayResultPromotionEligibility,
    ) -> DurableReplayResultPromotionActionPlan:
        recommendation = eligibility.recommendation

        if recommendation == "eligible":
            action: PromotionAction = "prepare_publish"
        elif recommendation == "insufficient_evidence":
            action = "hold"
        elif recommendation == "ineligible":
            action = "block"
        else:
            raise ValueError("unknown promotion eligibility recommendation")

        if not isinstance(eligibility.window, Mapping):
            raise TypeError("promotion eligibility window must be a mapping")

        return DurableReplayResultPromotionActionPlan(
            status=eligibility.status,
            round_count=eligibility.round_count,
            candidate_model_name=eligibility.candidate_model_name,
            baseline_model_name=eligibility.baseline_model_name,
            recommendation=recommendation,
            action=action,
            window=MappingProxyType(dict(eligibility.window)),
        )