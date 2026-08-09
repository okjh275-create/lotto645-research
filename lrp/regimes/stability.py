"""Stability policy for deterministic regime decisions."""

from __future__ import annotations

from dataclasses import dataclass
import math

from lrp.contracts import ContractError

from .contracts import (
    RegimeDecision,
    RegimeFeatureSnapshot,
)
from .detector import RegimeDetector


@dataclass(frozen=True, slots=True)
class RegimeStabilityConfig:
    """Controls regime transition hysteresis."""

    hysteresis_margin: float = 0.05
    minimum_retained_score: float = 0.35
    secondary_max_gap: float = 0.30

    def __post_init__(self) -> None:
        for name in (
            "hysteresis_margin",
            "minimum_retained_score",
            "secondary_max_gap",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ContractError(
                    f"{name} must be between 0 and 1"
                )


class RegimeStabilityPolicy:
    """Apply hysteresis and secondary-score consistency."""

    def __init__(
        self,
        *,
        detector: RegimeDetector | None = None,
        config: RegimeStabilityConfig | None = None,
    ) -> None:
        self.detector = detector or RegimeDetector()
        self.config = config or RegimeStabilityConfig()

    def decide(
        self,
        features: RegimeFeatureSnapshot,
        *,
        previous: RegimeDecision | None = None,
    ) -> RegimeDecision:
        current = self.detector.detect(features)

        if previous is None:
            return self._normalize_secondary(current)

        if not isinstance(previous, RegimeDecision):
            raise ContractError(
                "previous must be a RegimeDecision or None"
            )

        stabilized = self._apply_hysteresis(
            current,
            previous,
        )

        return self._normalize_secondary(stabilized)

    def _apply_hysteresis(
        self,
        current: RegimeDecision,
        previous: RegimeDecision,
    ) -> RegimeDecision:
        previous_name = previous.primary

        if previous_name == current.primary:
            return current

        previous_score = current.scores.get(
            previous_name
        )

        if previous_score is None:
            return current

        if (
            previous_score
            < self.config.minimum_retained_score
        ):
            return current

        score_gap = (
            current.confidence
            - previous_score
        )

        if score_gap > self.config.hysteresis_margin:
            return current

        secondary = current.primary
        secondary_confidence = current.confidence

        if secondary == previous_name:
            secondary = None
            secondary_confidence = None

        return RegimeDecision(
            primary=previous_name,
            confidence=previous_score,
            secondary=secondary,
            secondary_confidence=(
                secondary_confidence
            ),
            scores=current.scores,
            features=current.features,
        )

    def _normalize_secondary(
        self,
        decision: RegimeDecision,
    ) -> RegimeDecision:
        secondary = decision.secondary

        if secondary is None:
            return decision

        secondary_score = decision.scores[secondary]
        gap = decision.confidence - secondary_score

        if (
            secondary_score > decision.confidence
            or gap > self.config.secondary_max_gap
        ):
            return RegimeDecision(
                primary=decision.primary,
                confidence=decision.confidence,
                secondary=None,
                secondary_confidence=None,
                scores=decision.scores,
                features=decision.features,
            )

        if (
            decision.secondary_confidence
            != secondary_score
        ):
            return RegimeDecision(
                primary=decision.primary,
                confidence=decision.confidence,
                secondary=secondary,
                secondary_confidence=secondary_score,
                scores=decision.scores,
                features=decision.features,
            )

        return decision
