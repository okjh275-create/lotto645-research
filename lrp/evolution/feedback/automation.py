"""Orchestrate adaptive feedback, recommendation, safety, and planning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from lrp.evolution.contracts import (
    AdaptiveWeightProfile,
)
from lrp.evolution.feedback.analyzer import (
    AdaptiveFeedbackAnalyzer,
)
from lrp.evolution.feedback.contracts import (
    AdaptiveFeedback,
    AdaptiveRecommendation,
)
from lrp.evolution.feedback.profile_update import (
    AdaptiveProfileUpdatePlan,
    AdaptiveProfileUpdatePlanner,
)
from lrp.evolution.feedback.recommendation import (
    AdaptiveRecommendationEngine,
)
from lrp.evolution.feedback.safety import (
    AdaptiveSafetyGuard,
    AdaptiveSafetyResult,
)


@dataclass(frozen=True, slots=True)
class AdaptiveAutomationResult:
    """Complete adaptive-automation planning result."""

    feedback: tuple[
        AdaptiveFeedback,
        ...,
    ]
    recommendation: AdaptiveRecommendation
    safety_result: AdaptiveSafetyResult
    update_plan: AdaptiveProfileUpdatePlan

    def as_dict(self) -> dict[str, Any]:
        return {
            "feedback": [
                item.as_dict()
                for item in self.feedback
            ],
            "recommendation": (
                self.recommendation.as_dict()
            ),
            "safety_result": (
                self.safety_result.as_dict()
            ),
            "update_plan": (
                self.update_plan.as_dict()
            ),
        }


class AdaptiveAutomationService:
    """Create a safe next-profile plan from validation evidence."""

    def __init__(
        self,
        *,
        analyzer: (
            AdaptiveFeedbackAnalyzer | None
        ) = None,
        recommendation_engine: (
            AdaptiveRecommendationEngine | None
        ) = None,
        safety_guard: (
            AdaptiveSafetyGuard | None
        ) = None,
        planner: (
            AdaptiveProfileUpdatePlanner | None
        ) = None,
    ) -> None:
        if (
            analyzer is not None
            and not isinstance(
                analyzer,
                AdaptiveFeedbackAnalyzer,
            )
        ):
            raise TypeError(
                "analyzer must be an "
                "AdaptiveFeedbackAnalyzer or None"
            )

        if (
            recommendation_engine is not None
            and not isinstance(
                recommendation_engine,
                AdaptiveRecommendationEngine,
            )
        ):
            raise TypeError(
                "recommendation_engine must be an "
                "AdaptiveRecommendationEngine or None"
            )

        if (
            safety_guard is not None
            and not isinstance(
                safety_guard,
                AdaptiveSafetyGuard,
            )
        ):
            raise TypeError(
                "safety_guard must be an "
                "AdaptiveSafetyGuard or None"
            )

        if (
            planner is not None
            and not isinstance(
                planner,
                AdaptiveProfileUpdatePlanner,
            )
        ):
            raise TypeError(
                "planner must be an "
                "AdaptiveProfileUpdatePlanner or None"
            )

        self._analyzer = (
            analyzer
            if analyzer is not None
            else AdaptiveFeedbackAnalyzer()
        )
        self._recommendation_engine = (
            recommendation_engine
            if recommendation_engine is not None
            else AdaptiveRecommendationEngine()
        )
        self._safety_guard = (
            safety_guard
            if safety_guard is not None
            else AdaptiveSafetyGuard()
        )
        self._planner = (
            planner
            if planner is not None
            else AdaptiveProfileUpdatePlanner()
        )

    @property
    def analyzer(
        self,
    ) -> AdaptiveFeedbackAnalyzer:
        return self._analyzer

    @property
    def recommendation_engine(
        self,
    ) -> AdaptiveRecommendationEngine:
        return self._recommendation_engine

    @property
    def safety_guard(
        self,
    ) -> AdaptiveSafetyGuard:
        return self._safety_guard

    @property
    def planner(
        self,
    ) -> AdaptiveProfileUpdatePlanner:
        return self._planner

    def run(
        self,
        *,
        report: Mapping[str, Any],
        policy_name: str,
        recommendation_id: str,
        current_profile: AdaptiveWeightProfile,
        created_at_utc: datetime | None = None,
        target_confidence: float | None = None,
        target_sample_size: int | None = None,
    ) -> AdaptiveAutomationResult:
        if not isinstance(
            current_profile,
            AdaptiveWeightProfile,
        ):
            raise TypeError(
                "current_profile must be an "
                "AdaptiveWeightProfile"
            )

        feedback = self.analyzer.analyze(
            report,
            policy_name=policy_name,
        )

        current_weights = (
            current_profile
            .to_probability_weights()
        )

        recommendation = (
            self.recommendation_engine.recommend(
                recommendation_id=(
                    recommendation_id
                ),
                feedback=feedback,
                current_weights=(
                    self._profile_weights(
                        current_weights
                    )
                ),
                created_at_utc=(
                    created_at_utc
                ),
            )
        )

        safety_result = (
            self.safety_guard.validate(
                recommendation=(
                    recommendation
                ),
                current_weights=(
                    self._profile_weights(
                        current_weights
                    )
                ),
            )
        )

        update_plan = self.planner.plan(
            current_profile=current_profile,
            safety_result=safety_result,
            confidence=target_confidence,
            sample_size=target_sample_size,
            generated_at=created_at_utc,
        )

        return AdaptiveAutomationResult(
            feedback=feedback,
            recommendation=recommendation,
            safety_result=safety_result,
            update_plan=update_plan,
        )

    @staticmethod
    def _profile_weights(
        weights: Mapping[str, float],
    ) -> dict[str, float]:
        return {
            "hot_weight": float(
                weights["hot"]
            ),
            "cold_weight": float(
                weights["cold"]
            ),
            "gap_weight": float(
                weights["gap"]
            ),
            "trend_weight": float(
                weights["trend"]
            ),
            "transition_weight": float(
                weights["transition"]
            ),
            "learning_weight": float(
                weights["learning"]
            ),
            "adaptive_weight": float(
                weights["adaptive"]
            ),
        }
