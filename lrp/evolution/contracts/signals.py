from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import ClassVar, Mapping, Protocol, runtime_checkable


SIGNAL_COMPONENTS: tuple[str, ...] = (
    "hot",
    "cold",
    "gap",
    "trend",
    "transition",
    "learning",
    "adaptive",
)


@dataclass(frozen=True, slots=True)
class SignalFrame:
    """Signals emitted by one evolution signal source."""

    source: str
    signals: Mapping[str, float]
    reliability: float = 1.0

    COMPONENTS: ClassVar[tuple[str, ...]] = (
        SIGNAL_COMPONENTS
    )

    def __post_init__(self) -> None:
        normalized_source = self._normalize_source(
            self.source
        )
        normalized_signals = self._normalize_signals(
            self.signals
        )
        normalized_reliability = (
            self._normalize_reliability(
                self.reliability
            )
        )

        object.__setattr__(
            self,
            "source",
            normalized_source,
        )
        object.__setattr__(
            self,
            "signals",
            MappingProxyType(normalized_signals),
        )
        object.__setattr__(
            self,
            "reliability",
            normalized_reliability,
        )

    def signal_for(
        self,
        component: str,
    ) -> float | None:
        """Return one signal or None when not supplied."""

        if component not in self.COMPONENTS:
            raise ValueError(
                f"unknown signal component: {component}"
            )

        return self.signals.get(component)

    @staticmethod
    def _normalize_source(
        source: str,
    ) -> str:
        if not isinstance(source, str):
            raise TypeError(
                "source must be a string"
            )

        normalized = source.strip()

        if not normalized:
            raise ValueError(
                "source must not be empty"
            )

        return normalized

    @classmethod
    def _normalize_signals(
        cls,
        signals: Mapping[str, float],
    ) -> dict[str, float]:
        if not isinstance(signals, Mapping):
            raise TypeError(
                "signals must be a mapping"
            )

        if not signals:
            raise ValueError(
                "signals must contain at least one item"
            )

        normalized: dict[str, float] = {}

        for raw_name, raw_value in signals.items():
            if not isinstance(raw_name, str):
                raise TypeError(
                    "signal names must be strings"
                )

            name = raw_name.strip()

            if not name:
                raise ValueError(
                    "signal names must not be empty"
                )

            if name not in cls.COMPONENTS:
                raise ValueError(
                    f"unknown signal component: {name}"
                )

            if name in normalized:
                raise ValueError(
                    f"duplicate signal component: {name}"
                )

            if isinstance(raw_value, bool):
                raise TypeError(
                    f"signal '{name}' must be numeric"
                )

            if not isinstance(
                raw_value,
                (int, float),
            ):
                raise TypeError(
                    f"signal '{name}' must be numeric"
                )

            value = float(raw_value)

            if not isfinite(value):
                raise ValueError(
                    f"signal '{name}' must be finite"
                )

            if not -1.0 <= value <= 1.0:
                raise ValueError(
                    f"signal '{name}' must be "
                    "between -1.0 and 1.0"
                )

            normalized[name] = value

        return normalized

    @staticmethod
    def _normalize_reliability(
        reliability: float,
    ) -> float:
        if isinstance(reliability, bool):
            raise TypeError(
                "reliability must be numeric"
            )

        if not isinstance(
            reliability,
            (int, float),
        ):
            raise TypeError(
                "reliability must be numeric"
            )

        value = float(reliability)

        if not isfinite(value):
            raise ValueError(
                "reliability must be finite"
            )

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "reliability must be between "
                "0.0 and 1.0"
            )

        return value


@dataclass(frozen=True, slots=True)
class SignalAggregationResult:
    """Final seven-component signal aggregation result."""

    signals: Mapping[str, float]
    total_weight: float
    source_count: int

    COMPONENTS: ClassVar[tuple[str, ...]] = (
        SIGNAL_COMPONENTS
    )

    def __post_init__(self) -> None:
        normalized_signals = self._normalize_signals(
            self.signals
        )
        total_weight = self._normalize_total_weight(
            self.total_weight
        )
        self._validate_source_count(
            self.source_count
        )

        object.__setattr__(
            self,
            "signals",
            MappingProxyType(normalized_signals),
        )
        object.__setattr__(
            self,
            "total_weight",
            total_weight,
        )

    @classmethod
    def _normalize_signals(
        cls,
        signals: Mapping[str, float],
    ) -> dict[str, float]:
        if not isinstance(signals, Mapping):
            raise TypeError(
                "signals must be a mapping"
            )

        provided = set(signals)
        required = set(cls.COMPONENTS)

        missing = sorted(required - provided)
        unknown = sorted(provided - required)

        if missing:
            raise ValueError(
                "missing aggregated signals: "
                + ", ".join(missing)
            )

        if unknown:
            raise ValueError(
                "unknown aggregated signals: "
                + ", ".join(unknown)
            )

        normalized: dict[str, float] = {}

        for name in cls.COMPONENTS:
            raw_value = signals[name]

            if isinstance(raw_value, bool):
                raise TypeError(
                    f"signal '{name}' must be numeric"
                )

            if not isinstance(
                raw_value,
                (int, float),
            ):
                raise TypeError(
                    f"signal '{name}' must be numeric"
                )

            value = float(raw_value)

            if not isfinite(value):
                raise ValueError(
                    f"signal '{name}' must be finite"
                )

            if not -1.0 <= value <= 1.0:
                raise ValueError(
                    f"signal '{name}' must be "
                    "between -1.0 and 1.0"
                )

            normalized[name] = value

        return normalized

    @staticmethod
    def _normalize_total_weight(
        total_weight: float,
    ) -> float:
        if isinstance(total_weight, bool):
            raise TypeError(
                "total_weight must be numeric"
            )

        if not isinstance(
            total_weight,
            (int, float),
        ):
            raise TypeError(
                "total_weight must be numeric"
            )

        value = float(total_weight)

        if not isfinite(value):
            raise ValueError(
                "total_weight must be finite"
            )

        if value <= 0.0:
            raise ValueError(
                "total_weight must be greater than 0"
            )

        return value

    @staticmethod
    def _validate_source_count(
        source_count: int,
    ) -> None:
        if isinstance(source_count, bool):
            raise TypeError(
                "source_count must be an integer"
            )

        if not isinstance(source_count, int):
            raise TypeError(
                "source_count must be an integer"
            )

        if source_count < 1:
            raise ValueError(
                "source_count must be greater than "
                "or equal to 1"
            )


@runtime_checkable
class SignalSource(Protocol):
    """Contract implemented by evolution signal producers."""

    def produce(self) -> SignalFrame:
        """Produce one validated signal frame."""
        ...
