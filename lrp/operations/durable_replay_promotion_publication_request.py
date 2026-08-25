from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from lrp.operations.durable_replay_result_promotion_action_plan import (
    DurableReplayResultPromotionActionPlan,
)


@dataclass(frozen=True)
class DurableReplayPromotionPublicationRequest:
    status: str
    round_count: int
    candidate_model_name: str
    baseline_model_name: str
    recommendation: str
    action: str
    window: Mapping[str, object]
    source_decision: str | Path
    registry_root: str | Path


def _require_publication_identity(value: object, field_name: str) -> str | Path:
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{field_name} must be str or Path")
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"{field_name} must not be empty")
    else:
        # Path("") normalizes to "."; reject it as the empty-path sentinel.
        if str(value) in ("", "."):
            raise ValueError(f"{field_name} must not be empty")
    return value


class DurableReplayPromotionPublicationRequestService:
    def build(
        self,
        *,
        action_plan: DurableReplayResultPromotionActionPlan,
        source_decision: str | Path,
        registry_root: str | Path,
    ) -> DurableReplayPromotionPublicationRequest:
        if action_plan.action != "prepare_publish":
            raise ValueError("action_plan.action must be prepare_publish")
        if not isinstance(action_plan.window, Mapping):
            raise TypeError("action_plan.window must be a Mapping")

        source_decision = _require_publication_identity(
            source_decision, "source_decision"
        )
        registry_root = _require_publication_identity(
            registry_root, "registry_root"
        )

        return DurableReplayPromotionPublicationRequest(
            status=action_plan.status,
            round_count=action_plan.round_count,
            candidate_model_name=action_plan.candidate_model_name,
            baseline_model_name=action_plan.baseline_model_name,
            recommendation=action_plan.recommendation,
            action=action_plan.action,
            window=MappingProxyType(dict(action_plan.window)),
            source_decision=source_decision,
            registry_root=registry_root,
        )