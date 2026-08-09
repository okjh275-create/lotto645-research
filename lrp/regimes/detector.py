"""Deterministic baseline regime detector."""

from __future__ import annotations

from dataclasses import dataclass
import math

from lrp.contracts import ContractError

from .contracts import (
    RegimeDecision,
    RegimeFeatureSnapshot,
)


def _unit(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _proximity(
    value: float,
    target: float,
    width: float,
) -> float:
    if width <= 0.0:
        raise ContractError("width must be positive")

    return _unit(
        1.0 - abs(value - target) / width
    )


@dataclass(frozen=True, slots=True)
class RegimeDetectorConfig:
    """Thresholds for the deterministic baseline detector."""

    neutral_margin: float = 0.08
    mixed_margin: float = 0.10
    secondary_min_score: float = 0.35

    def __post_init__(self) -> None:
        for name in (
            "neutral_margin",
            "mixed_margin",
            "secondary_min_score",
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


class RegimeDetector:
    """Classify a global regime feature snapshot."""

    def __init__(
        self,
        config: RegimeDetectorConfig | None = None,
    ) -> None:
        self.config = config or RegimeDetectorConfig()

    def detect(
        self,
        features: RegimeFeatureSnapshot,
    ) -> RegimeDecision:
        if not isinstance(
            features,
            RegimeFeatureSnapshot,
        ):
            raise ContractError(
                "features must be a RegimeFeatureSnapshot"
            )

        scores = self._scores(features)

        ordered = sorted(
            scores.items(),
            key=lambda item: (-item[1], item[0]),
        )

        primary, primary_score = ordered[0]
        second_name, second_score = ordered[1]

        gap = primary_score - second_score

        if gap <= self.config.mixed_margin:
            primary = "mixed"
            primary_score = max(
                scores["mixed"],
                1.0 - gap / max(
                    self.config.mixed_margin,
                    1e-12,
                ),
            )
            scores["mixed"] = _unit(primary_score)
            ordered = sorted(
                scores.items(),
                key=lambda item: (-item[1], item[0]),
            )
            primary, primary_score = ordered[0]
            second_name, second_score = ordered[1]

        if (
            primary != "mixed"
            and abs(
                features.low_band_ratio
                - features.high_band_ratio
            ) <= self.config.neutral_margin
            and features.average_gap_reversion < 0.55
            and features.pair_variance < 0.20
        ):
            neutral_score = max(
                scores["neutral"],
                0.65,
            )
            scores["neutral"] = _unit(neutral_score)
            ordered = sorted(
                scores.items(),
                key=lambda item: (-item[1], item[0]),
            )
            primary, primary_score = ordered[0]
            second_name, second_score = ordered[1]

        secondary: str | None = None
        secondary_confidence: float | None = None

        if (
            second_name != primary
            and second_score >= self.config.secondary_min_score
        ):
            secondary = second_name
            secondary_confidence = second_score

        return RegimeDecision(
            primary=primary,
            confidence=_unit(primary_score),
            secondary=secondary,
            secondary_confidence=(
                _unit(secondary_confidence)
                if secondary_confidence is not None
                else None
            ),
            scores=scores,
            features=features,
        )

    def _scores(
        self,
        f: RegimeFeatureSnapshot,
    ) -> dict[str, float]:
        gap_recovery = _unit(
            0.45 * f.average_gap_reversion
            + 0.20 * f.frequency_dispersion
            + 0.20 * f.recency_variance
            + 0.15 * (1.0 - f.average_recency)
        )

        cluster_rotation = _unit(
            0.40 * f.pair_density
            + 0.30 * f.pair_variance
            + 0.20 * f.frequency_dispersion
            + 0.10 * f.average_frequency
        )

        high_band_expansion = _unit(
            0.65 * f.high_band_ratio
            + 0.20 * f.average_frequency
            + 0.15 * f.pair_density
        )

        low_band_expansion = _unit(
            0.65 * f.low_band_ratio
            + 0.20 * f.average_frequency
            + 0.15 * f.pair_density
        )

        balance = 1.0 - abs(
            f.low_band_ratio
            - f.high_band_ratio
        )

        neutral = _unit(
            0.40 * balance
            + 0.20 * _proximity(
                f.average_recency,
                0.5,
                0.5,
            )
            + 0.20 * _proximity(
                f.average_gap_reversion,
                0.5,
                0.5,
            )
            + 0.20 * (1.0 - f.pair_variance)
        )

        raw = [
            gap_recovery,
            cluster_rotation,
            high_band_expansion,
            low_band_expansion,
        ]

        spread = max(raw) - min(raw)
        mixed = _unit(1.0 - spread)

        return {
            "neutral": neutral,
            "mixed": mixed,
            "gap_recovery": gap_recovery,
            "cluster_rotation": cluster_rotation,
            "high_band_expansion": high_band_expansion,
            "low_band_expansion": low_band_expansion,
        }
