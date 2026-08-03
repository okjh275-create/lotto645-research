from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite
from typing import ClassVar, Mapping

from lrp.evolution.contracts import AdaptiveWeightProfile


class AdaptiveWeightCalculator:
    """Calculate normalized adaptive probability-fusion weights."""

    COMPONENTS: ClassVar[tuple[str, ...]] = (
        "hot",
        "cold",
        "gap",
        "trend",
        "transition",
        "learning",
        "adaptive",
    )

    COMPONENT_TO_FIELD: ClassVar[dict[str, str]] = {
        "hot": "hot_weight",
        "cold": "cold_weight",
        "gap": "gap_weight",
        "trend": "trend_weight",
        "transition": "transition_weight",
        "learning": "learning_weight",
        "adaptive": "adaptive_weight",
    }

    def __init__(
        self,
        *,
        adjustment_scale: float = 0.25,
        minimum_weight: float = 0.03,
    ) -> None:
        self._validate_configuration(
            adjustment_scale=adjustment_scale,
            minimum_weight=minimum_weight,
        )

        self._adjustment_scale = float(adjustment_scale)
        self._minimum_weight = float(minimum_weight)

    @property
    def adjustment_scale(self) -> float:
        return self._adjustment_scale

    @property
    def minimum_weight(self) -> float:
        return self._minimum_weight

    def calculate(
        self,
        signals: Mapping[str, float] | None = None,
        *,
        confidence: float,
        sample_size: int,
        revision: int,
        generated_at: datetime | None = None,
        baseline: AdaptiveWeightProfile | None = None,
    ) -> AdaptiveWeightProfile:
        """Create an adaptive profile from normalized learning signals.

        Each signal must be in the range [-1.0, 1.0].

        Positive values increase a component's relative influence.
        Negative values decrease a component's relative influence.
        Missing signals are interpreted as 0.0.
        """

        timestamp = generated_at or datetime.now(timezone.utc)

        base_profile = baseline or AdaptiveWeightProfile.default(
            generated_at=timestamp,
        )

        validated_signals = self._validate_signals(
            signals or {},
        )

        self._validate_confidence(confidence)

        baseline_weights = (
            base_profile.to_probability_weights()
        )

        adjusted_weights = self._apply_signals(
            baseline_weights=baseline_weights,
            signals=validated_signals,
        )

        normalized_adjusted = self._normalize(
            adjusted_weights,
        )

        blended_weights = self._blend_with_baseline(
            baseline_weights=baseline_weights,
            adjusted_weights=normalized_adjusted,
            confidence=float(confidence),
        )

        final_weights = self._normalize(blended_weights)

        return AdaptiveWeightProfile(
            hot_weight=final_weights["hot"],
            cold_weight=final_weights["cold"],
            gap_weight=final_weights["gap"],
            trend_weight=final_weights["trend"],
            transition_weight=final_weights["transition"],
            learning_weight=final_weights["learning"],
            adaptive_weight=final_weights["adaptive"],
            confidence=float(confidence),
            sample_size=sample_size,
            revision=revision,
            generated_at=timestamp,
        )

    def _apply_signals(
        self,
        *,
        baseline_weights: Mapping[str, float],
        signals: Mapping[str, float],
    ) -> dict[str, float]:
        adjusted: dict[str, float] = {}

        for component in self.COMPONENTS:
            baseline_value = float(
                baseline_weights[component]
            )
            signal = float(signals.get(component, 0.0))

            multiplier = (
                1.0
                + self.adjustment_scale * signal
            )

            raw_weight = baseline_value * multiplier

            adjusted[component] = max(
                raw_weight,
                self.minimum_weight,
            )

        return adjusted

    def _blend_with_baseline(
        self,
        *,
        baseline_weights: Mapping[str, float],
        adjusted_weights: Mapping[str, float],
        confidence: float,
    ) -> dict[str, float]:
        baseline_ratio = 1.0 - confidence

        return {
            component: (
                float(baseline_weights[component])
                * baseline_ratio
                + float(adjusted_weights[component])
                * confidence
            )
            for component in self.COMPONENTS
        }

    def _normalize(
        self,
        weights: Mapping[str, float],
    ) -> dict[str, float]:
        total = sum(
            float(weights[component])
            for component in self.COMPONENTS
        )

        if not isfinite(total) or total <= 0.0:
            raise ValueError(
                "weight total must be finite and greater than 0"
            )

        return {
            component: float(weights[component]) / total
            for component in self.COMPONENTS
        }

    def _validate_signals(
        self,
        signals: Mapping[str, float],
    ) -> dict[str, float]:
        if not isinstance(signals, Mapping):
            raise TypeError("signals must be a mapping")

        unknown = sorted(
            set(signals) - set(self.COMPONENTS)
        )

        if unknown:
            names = ", ".join(unknown)
            raise ValueError(
                f"unknown adaptive signal components: {names}"
            )

        validated: dict[str, float] = {}

        for component, raw_value in signals.items():
            if isinstance(raw_value, bool):
                raise TypeError(
                    f"signal '{component}' must be numeric"
                )

            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"signal '{component}' must be numeric"
                ) from exc

            if not isfinite(value):
                raise ValueError(
                    f"signal '{component}' must be finite"
                )

            if not -1.0 <= value <= 1.0:
                raise ValueError(
                    f"signal '{component}' must be "
                    "between -1.0 and 1.0"
                )

            validated[component] = value

        return validated

    @staticmethod
    def _validate_confidence(
        confidence: float,
    ) -> None:
        if isinstance(confidence, bool):
            raise TypeError("confidence must be numeric")

        try:
            value = float(confidence)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "confidence must be numeric"
            ) from exc

        if not isfinite(value):
            raise ValueError("confidence must be finite")

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "confidence must be between 0.0 and 1.0"
            )

    @classmethod
    def _validate_configuration(
        cls,
        *,
        adjustment_scale: float,
        minimum_weight: float,
    ) -> None:
        for name, raw_value in (
            ("adjustment_scale", adjustment_scale),
            ("minimum_weight", minimum_weight),
        ):
            if isinstance(raw_value, bool):
                raise TypeError(f"{name} must be numeric")

            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"{name} must be numeric"
                ) from exc

            if not isfinite(value):
                raise ValueError(
                    f"{name} must be finite"
                )

        if float(adjustment_scale) < 0.0:
            raise ValueError(
                "adjustment_scale must be greater than "
                "or equal to 0"
            )

        if not 0.0 <= float(minimum_weight) < 1.0:
            raise ValueError(
                "minimum_weight must be between "
                "0.0 inclusive and 1.0 exclusive"
            )

        if float(minimum_weight) * len(
            cls.COMPONENTS
        ) >= 1.0:
            raise ValueError(
                "minimum_weight is too large for "
                "the number of components"
            )