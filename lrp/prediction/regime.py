"""Number-level regime detection for Project F."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Any, Iterable, Mapping, Protocol
from zoneinfo import ZoneInfo

from lrp.contracts import ContractError

from .models import (
    NumberRegime,
    RegimeProfile,
    ordered_regimes,
)


_KST = ZoneInfo("Asia/Seoul")


class NumberFeatureLike(Protocol):
    """Minimum feature contract required by RegimeDetector."""

    number: int
    freq_all: int
    freq10: int
    freq20: int
    freq50: int
    gap: int


def _finite(
    value: object,
    *,
    field_name: str,
) -> float:
    """Validate a finite numeric value."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise ContractError(
            f"{field_name} must be numeric"
        )

    result = float(value)

    if not math.isfinite(result):
        raise ContractError(
            f"{field_name} must be finite"
        )

    return result


def _non_negative(
    value: object,
    *,
    field_name: str,
) -> float:
    """Validate a finite non-negative numeric value."""

    result = _finite(
        value,
        field_name=field_name,
    )

    if result < 0.0:
        raise ContractError(
            f"{field_name} must be non-negative"
        )

    return result


def _unit(value: float) -> float:
    """Clamp a numeric value to the inclusive 0..1 range."""

    return max(
        0.0,
        min(1.0, float(value)),
    )


def _ratio(
    numerator: float,
    denominator: float,
) -> float:
    """Return a safe ratio."""

    if denominator <= 0.0:
        return 0.0

    return numerator / denominator


@dataclass(frozen=True, slots=True)
class RegimeDetectorConfig:
    """Configuration for deterministic regime scoring."""

    short_window: int = 10
    mid_window: int = 20
    long_window: int = 50
    expected_draw_rate: float = 6.0 / 45.0
    gap_saturation: int = 15
    confidence_saturation: int = 50

    def __post_init__(self) -> None:
        for field_name in (
            "short_window",
            "mid_window",
            "long_window",
            "gap_saturation",
            "confidence_saturation",
        ):
            value = getattr(self, field_name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ContractError(
                    f"{field_name} must be a positive integer"
                )

        if not (
            self.short_window
            < self.mid_window
            < self.long_window
        ):
            raise ContractError(
                "windows must satisfy short < mid < long"
            )

        expected = _finite(
            self.expected_draw_rate,
            field_name="expected_draw_rate",
        )

        if not 0.0 < expected < 1.0:
            raise ContractError(
                "expected_draw_rate must be between 0 and 1"
            )

    def as_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly configuration mapping."""

        return {
            "short_window": self.short_window,
            "mid_window": self.mid_window,
            "long_window": self.long_window,
            "expected_draw_rate": (
                self.expected_draw_rate
            ),
            "gap_saturation": self.gap_saturation,
            "confidence_saturation": (
                self.confidence_saturation
            ),
        }


@dataclass(frozen=True, slots=True)
class _ValidatedFeature:
    """Internal validated feature representation."""

    number: int
    freq_all: float
    freq10: float
    freq20: float
    freq50: float
    gap: float


class RegimeDetector:
    """Convert frequency and gap features into regime scores."""

    def __init__(
        self,
        config: RegimeDetectorConfig | None = None,
    ) -> None:
        if config is None:
            config = RegimeDetectorConfig()

        if not isinstance(
            config,
            RegimeDetectorConfig,
        ):
            raise ContractError(
                "config must be a RegimeDetectorConfig"
            )

        self._config = config

    @property
    def config(self) -> RegimeDetectorConfig:
        """Return the active detector configuration."""

        return self._config

    def _validate_feature(
        self,
        feature: object,
    ) -> _ValidatedFeature:
        required = (
            "number",
            "freq_all",
            "freq10",
            "freq20",
            "freq50",
            "gap",
        )

        missing = tuple(
            name
            for name in required
            if not hasattr(feature, name)
        )

        if missing:
            raise ContractError(
                "feature is missing required fields: "
                + ", ".join(missing)
            )

        number = getattr(feature, "number")

        if (
            isinstance(number, bool)
            or not isinstance(number, int)
            or not 1 <= number <= 45
        ):
            raise ContractError(
                "feature number must be an integer "
                "between 1 and 45"
            )

        values = {
            field_name: _non_negative(
                getattr(feature, field_name),
                field_name=field_name,
            )
            for field_name in (
                "freq_all",
                "freq10",
                "freq20",
                "freq50",
                "gap",
            )
        }

        if values["freq10"] > self._config.short_window:
            raise ContractError(
                "freq10 exceeds short window"
            )

        if values["freq20"] > self._config.mid_window:
            raise ContractError(
                "freq20 exceeds mid window"
            )

        if values["freq50"] > self._config.long_window:
            raise ContractError(
                "freq50 exceeds long window"
            )

        return _ValidatedFeature(
            number=number,
            freq_all=values["freq_all"],
            freq10=values["freq10"],
            freq20=values["freq20"],
            freq50=values["freq50"],
            gap=values["gap"],
        )

    def _score(
        self,
        feature: _ValidatedFeature,
    ) -> NumberRegime:
        config = self._config

        short_rate = _ratio(
            feature.freq10,
            config.short_window,
        )
        mid_rate = _ratio(
            feature.freq20,
            config.mid_window,
        )
        long_rate = _ratio(
            feature.freq50,
            config.long_window,
        )

        expected = config.expected_draw_rate

        short_relative = _ratio(
            short_rate,
            expected,
        )
        mid_relative = _ratio(
            mid_rate,
            expected,
        )
        long_relative = _ratio(
            long_rate,
            expected,
        )

        hot_score = _unit(
            (
                0.50 * short_relative
                + 0.30 * mid_relative
                + 0.20 * long_relative
            )
            / 2.0
        )

        cold_score = _unit(
            1.0
            - (
                0.50 * min(1.0, short_relative)
                + 0.30 * min(1.0, mid_relative)
                + 0.20 * min(1.0, long_relative)
            )
        )

        gap_score = _unit(
            feature.gap
            / config.gap_saturation
        )

        trend_raw = (
            0.65 * (short_rate - mid_rate)
            + 0.35 * (mid_rate - long_rate)
        )

        trend_scale = max(
            expected,
            1.0 / config.long_window,
        )

        trend_score = _unit(
            0.5
            + trend_raw
            / (2.0 * trend_scale)
        )

        frequency_shift = _unit(
            abs(short_rate - mid_rate)
            / max(expected, 0.000001)
        )

        gap_pressure = _unit(
            feature.gap
            / config.gap_saturation
        )

        transition_score = _unit(
            0.60 * frequency_shift
            + 0.40 * gap_pressure
        )

        observed_sample = min(
            feature.freq_all,
            config.confidence_saturation,
        )

        sample_confidence = _unit(
            observed_sample
            / config.confidence_saturation
        )

        window_consistency = _unit(
            1.0
            - (
                abs(short_rate - mid_rate)
                + abs(mid_rate - long_rate)
            )
            / 2.0
        )

        confidence = _unit(
            0.55 * sample_confidence
            + 0.45 * window_consistency
        )

        return NumberRegime(
            number=feature.number,
            hot_score=hot_score,
            cold_score=cold_score,
            gap_score=gap_score,
            trend_score=trend_score,
            transition_score=transition_score,
            confidence=confidence,
            metadata={
                "freq_all": feature.freq_all,
                "freq10": feature.freq10,
                "freq20": feature.freq20,
                "freq50": feature.freq50,
                "gap": feature.gap,
                "short_rate": short_rate,
                "mid_rate": mid_rate,
                "long_rate": long_rate,
            },
        )

    def detect(
        self,
        features: Iterable[NumberFeatureLike],
        *,
        round_no: int | None = None,
        generated_at_kst: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RegimeProfile:
        """Build a deterministic 45-number regime profile."""

        if round_no is not None and (
            isinstance(round_no, bool)
            or not isinstance(round_no, int)
            or round_no <= 0
        ):
            raise ContractError(
                "round_no must be a positive integer or None"
            )

        if generated_at_kst is None:
            generated_at_kst = datetime.now(
                _KST
            ).isoformat(timespec="seconds")
        elif (
            not isinstance(generated_at_kst, str)
            or not generated_at_kst.strip()
        ):
            raise ContractError(
                "generated_at_kst must be a "
                "non-empty string or None"
            )

        if metadata is None:
            metadata = {}

        if not isinstance(metadata, Mapping):
            raise ContractError(
                "metadata must be a mapping"
            )

        validated = tuple(
            self._validate_feature(feature)
            for feature in features
        )

        if len(validated) != 45:
            raise ContractError(
                "features must contain exactly 45 records"
            )

        numbers = tuple(
            item.number
            for item in validated
        )

        if len(set(numbers)) != 45:
            raise ContractError(
                "features must contain unique numbers"
            )

        if set(numbers) != set(range(1, 46)):
            raise ContractError(
                "features must cover numbers 1 through 45"
            )

        regimes = ordered_regimes(
            self._score(feature)
            for feature in validated
        )

        profile_metadata = {
            "engine": "F-001",
            "detector": type(self).__name__,
            "config": self._config.as_dict(),
        }
        profile_metadata.update(
            dict(metadata)
        )

        return RegimeProfile(
            round_no=round_no,
            generated_at_kst=generated_at_kst,
            regimes=regimes,
            metadata=profile_metadata,
        )
