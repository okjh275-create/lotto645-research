from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite

from lrp.evolution.contracts.signals import (
    SIGNAL_COMPONENTS,
    SignalAggregationResult,
    SignalFrame,
)


class WeightedSignalAggregator:
    """Combine multiple signal frames using source weights."""

    COMPONENTS = SIGNAL_COMPONENTS

    def aggregate(
        self,
        frames: Sequence[SignalFrame],
        *,
        source_weights: Mapping[str, float] | None = None,
    ) -> SignalAggregationResult:
        """Return one seven-component weighted signal result.

        Missing component values do not vote on that component.
        Components supplied by no source are returned as 0.0.
        """

        validated_frames = self._validate_frames(
            frames
        )
        validated_weights = self._validate_source_weights(
            source_weights or {},
            frames=validated_frames,
        )

        weighted_totals = {
            component: 0.0
            for component in self.COMPONENTS
        }
        component_weights = {
            component: 0.0
            for component in self.COMPONENTS
        }

        total_weight = 0.0

        for frame in validated_frames:
            configured_weight = validated_weights.get(
                frame.source,
                1.0,
            )
            effective_weight = (
                configured_weight
                * frame.reliability
            )

            total_weight += effective_weight

            if effective_weight == 0.0:
                continue

            for component, signal in frame.signals.items():
                weighted_totals[component] += (
                    signal * effective_weight
                )
                component_weights[component] += (
                    effective_weight
                )

        if total_weight <= 0.0:
            raise ValueError(
                "effective source weight total must be "
                "greater than 0"
            )

        aggregated = {
            component: self._component_average(
                weighted_total=weighted_totals[component],
                component_weight=component_weights[
                    component
                ],
            )
            for component in self.COMPONENTS
        }

        return SignalAggregationResult(
            signals=aggregated,
            total_weight=total_weight,
            source_count=len(validated_frames),
        )

    @staticmethod
    def _component_average(
        *,
        weighted_total: float,
        component_weight: float,
    ) -> float:
        if component_weight == 0.0:
            return 0.0

        value = weighted_total / component_weight

        return min(
            1.0,
            max(-1.0, value),
        )

    @classmethod
    def _validate_frames(
        cls,
        frames: Sequence[SignalFrame],
    ) -> tuple[SignalFrame, ...]:
        if isinstance(
            frames,
            (str, bytes),
        ) or not isinstance(frames, Sequence):
            raise TypeError(
                "frames must be a sequence"
            )

        if not frames:
            raise ValueError(
                "frames must contain at least one item"
            )

        validated: list[SignalFrame] = []
        seen_sources: set[str] = set()

        for frame in frames:
            if not isinstance(frame, SignalFrame):
                raise TypeError(
                    "each frame must be a SignalFrame"
                )

            if frame.source in seen_sources:
                raise ValueError(
                    f"duplicate signal source: "
                    f"{frame.source}"
                )

            seen_sources.add(frame.source)
            validated.append(frame)

        return tuple(validated)

    @staticmethod
    def _validate_source_weights(
        source_weights: Mapping[str, float],
        *,
        frames: Sequence[SignalFrame],
    ) -> dict[str, float]:
        if not isinstance(source_weights, Mapping):
            raise TypeError(
                "source_weights must be a mapping"
            )

        known_sources = {
            frame.source
            for frame in frames
        }

        unknown_sources = sorted(
            set(source_weights) - known_sources
        )

        if unknown_sources:
            raise ValueError(
                "unknown source weights: "
                + ", ".join(unknown_sources)
            )

        validated: dict[str, float] = {}

        for source, raw_weight in source_weights.items():
            if not isinstance(source, str):
                raise TypeError(
                    "source weight names must be strings"
                )

            if isinstance(raw_weight, bool):
                raise TypeError(
                    f"weight for '{source}' must be numeric"
                )

            if not isinstance(
                raw_weight,
                (int, float),
            ):
                raise TypeError(
                    f"weight for '{source}' must be numeric"
                )

            weight = float(raw_weight)

            if not isfinite(weight):
                raise ValueError(
                    f"weight for '{source}' must be finite"
                )

            if weight < 0.0:
                raise ValueError(
                    f"weight for '{source}' must be "
                    "greater than or equal to 0"
                )

            validated[source] = weight

        return validated


class SignalAggregator:
    """Combine signal frames with equal configured weights."""

    def __init__(
        self,
        weighted_aggregator: (
            WeightedSignalAggregator | None
        ) = None,
    ) -> None:
        if (
            weighted_aggregator is not None
            and not isinstance(
                weighted_aggregator,
                WeightedSignalAggregator,
            )
        ):
            raise TypeError(
                "weighted_aggregator must be a "
                "WeightedSignalAggregator or None"
            )

        self._weighted_aggregator = (
            weighted_aggregator
            if weighted_aggregator is not None
            else WeightedSignalAggregator()
        )

    @property
    def weighted_aggregator(
        self,
    ) -> WeightedSignalAggregator:
        return self._weighted_aggregator

    def aggregate(
        self,
        frames: Sequence[SignalFrame],
    ) -> SignalAggregationResult:
        """Aggregate using frame reliability only."""

        return self.weighted_aggregator.aggregate(
            frames
        )
