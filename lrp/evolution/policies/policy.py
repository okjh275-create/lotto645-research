from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import ClassVar, Mapping

from lrp.evolution.contracts import AdaptiveWeightProfile
from lrp.evolution.policies.calibration import (
    AdaptivePolicyConfig,
)


@dataclass(frozen=True, slots=True)
class AdaptivePolicyDecision:
    """Result of evaluating an adaptive weight profile."""

    applied: bool
    profile: AdaptiveWeightProfile
    reasons: tuple[str, ...] = ()
    clamped_components: tuple[str, ...] = ()

    @property
    def rejected(self) -> bool:
        return not self.applied

    @property
    def was_clamped(self) -> bool:
        return bool(self.clamped_components)


class AdaptiveWeightPolicy:
    """Apply confidence, sample, revision, and delta safeguards."""

    COMPONENTS: ClassVar[tuple[str, ...]] = (
        "hot",
        "cold",
        "gap",
        "trend",
        "transition",
        "learning",
        "adaptive",
    )

    def __init__(
        self,
        config: AdaptivePolicyConfig | None = None,
    ) -> None:
        self._config = config or AdaptivePolicyConfig()

    @property
    def config(self) -> AdaptivePolicyConfig:
        return self._config

    def evaluate(
        self,
        candidate: AdaptiveWeightProfile,
        *,
        previous: AdaptiveWeightProfile | None = None,
    ) -> AdaptivePolicyDecision:
        """Evaluate and safely apply a candidate profile."""

        if not isinstance(
            candidate,
            AdaptiveWeightProfile,
        ):
            raise TypeError(
                "candidate must be an AdaptiveWeightProfile"
            )

        if previous is not None and not isinstance(
            previous,
            AdaptiveWeightProfile,
        ):
            raise TypeError(
                "previous must be an AdaptiveWeightProfile"
            )

        rejection_reasons = self._collect_rejection_reasons(
            candidate=candidate,
            previous=previous,
        )

        if rejection_reasons and not self.config.fail_open:
            fallback = previous or AdaptiveWeightProfile.default(
                revision=1,
                generated_at=candidate.generated_at,
            )

            return AdaptivePolicyDecision(
                applied=False,
                profile=fallback,
                reasons=tuple(rejection_reasons),
            )

        accepted_profile = candidate
        clamped_components: tuple[str, ...] = ()

        if previous is not None:
            (
                accepted_profile,
                clamped_components,
            ) = self._clamp_against_previous(
                candidate=candidate,
                previous=previous,
            )

        reasons = list(rejection_reasons)

        if rejection_reasons and self.config.fail_open:
            reasons.append("fail_open_applied")

        if clamped_components:
            reasons.append("component_delta_clamped")

        return AdaptivePolicyDecision(
            applied=True,
            profile=accepted_profile,
            reasons=tuple(reasons),
            clamped_components=clamped_components,
        )

    def _collect_rejection_reasons(
        self,
        *,
        candidate: AdaptiveWeightProfile,
        previous: AdaptiveWeightProfile | None,
    ) -> list[str]:
        reasons: list[str] = []

        if candidate.confidence < self.config.min_confidence:
            reasons.append("confidence_below_threshold")

        if candidate.sample_size < self.config.min_sample_size:
            reasons.append("sample_size_below_threshold")

        if (
            previous is not None
            and candidate.revision <= previous.revision
        ):
            reasons.append("revision_not_newer")

        return reasons

    def _clamp_against_previous(
        self,
        *,
        candidate: AdaptiveWeightProfile,
        previous: AdaptiveWeightProfile,
    ) -> tuple[
        AdaptiveWeightProfile,
        tuple[str, ...],
    ]:
        candidate_weights = (
            candidate.to_probability_weights()
        )
        previous_weights = (
            previous.to_probability_weights()
        )

        clamped: dict[str, float] = {}
        changed: list[str] = []

        for component in self.COMPONENTS:
            candidate_value = candidate_weights[component]
            previous_value = previous_weights[component]

            lower = max(
                0.0,
                previous_value
                - self.config.max_component_delta,
            )
            upper = min(
                1.0,
                previous_value
                + self.config.max_component_delta,
            )

            safe_value = min(
                max(candidate_value, lower),
                upper,
            )

            clamped[component] = safe_value

            if not isclose(
                safe_value,
                candidate_value,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                changed.append(component)

        if not changed:
            return candidate, ()

        normalized = self._normalize(clamped)

        profile = AdaptiveWeightProfile(
            hot_weight=normalized["hot"],
            cold_weight=normalized["cold"],
            gap_weight=normalized["gap"],
            trend_weight=normalized["trend"],
            transition_weight=normalized["transition"],
            learning_weight=normalized["learning"],
            adaptive_weight=normalized["adaptive"],
            confidence=candidate.confidence,
            sample_size=candidate.sample_size,
            revision=candidate.revision,
            generated_at=candidate.generated_at,
        )

        return profile, tuple(changed)

    def _normalize(
        self,
        weights: Mapping[str, float],
    ) -> dict[str, float]:
        total = sum(
            float(weights[component])
            for component in self.COMPONENTS
        )

        if total <= 0.0:
            raise ValueError(
                "clamped weight total must be greater than 0"
            )

        return {
            component: float(weights[component]) / total
            for component in self.COMPONENTS
        }