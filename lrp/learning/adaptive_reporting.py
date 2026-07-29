"""Adaptive-weight reporting and validation for Project E E-005B."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import math
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

from .adaptive_models import AdaptiveRevision, AdaptiveWeight


_KST = ZoneInfo("Asia/Seoul")
_VALID_DIRECTIONS = {"RAISED", "LOWERED", "UNCHANGED"}


def _required_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _positive_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _finite_number(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    return normalized


@dataclass(frozen=True, slots=True)
class AdaptiveWeightChange:
    """Validated report row for one adaptive strategy weight."""

    weight: AdaptiveWeight
    delta: float
    direction: str

    def __post_init__(self) -> None:
        if not isinstance(self.weight, AdaptiveWeight):
            raise ValueError("weight must be an AdaptiveWeight")

        delta = _finite_number(self.delta, field_name="delta")
        direction = _required_text(
            self.direction,
            field_name="direction",
        ).upper()

        if direction not in _VALID_DIRECTIONS:
            raise ValueError(
                "direction must be RAISED, LOWERED, or UNCHANGED"
            )

        tolerance = 1e-12
        if delta > tolerance and direction != "RAISED":
            raise ValueError("positive delta requires RAISED direction")
        if delta < -tolerance and direction != "LOWERED":
            raise ValueError("negative delta requires LOWERED direction")
        if abs(delta) <= tolerance and direction != "UNCHANGED":
            raise ValueError("zero delta requires UNCHANGED direction")

        expected = (
            self.weight.current_weight
            - self.weight.previous_weight
        )
        if not math.isclose(delta, expected, abs_tol=1e-12):
            raise ValueError(
                "delta must equal current_weight - previous_weight"
            )

        object.__setattr__(self, "delta", delta)
        object.__setattr__(self, "direction", direction)

    @property
    def strategy_key(self) -> tuple[str, str]:
        return self.weight.strategy_key

    def as_dict(self) -> dict[str, object]:
        payload = self.weight.as_dict()
        payload.update(
            {
                "delta": round(self.delta, 6),
                "direction": self.direction,
            }
        )
        return payload


@dataclass(frozen=True, slots=True)
class AdaptiveWeightReport:
    """Immutable E-005B adaptive-weight report."""

    revision: AdaptiveRevision
    strategy_type: str | None
    history_limit: int
    generated_at_kst: str
    changes: tuple[AdaptiveWeightChange, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        revision = tuple(self.revision)
        if (
            len(revision) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in revision
            )
        ):
            raise ValueError(
                "revision must contain two non-negative integers"
            )

        strategy_type = self.strategy_type
        if strategy_type is not None:
            strategy_type = _required_text(
                strategy_type,
                field_name="strategy_type",
            ).lower()

        history_limit = _positive_integer(
            self.history_limit,
            field_name="history_limit",
        )
        generated_at_kst = _required_text(
            self.generated_at_kst,
            field_name="generated_at_kst",
        )

        changes = tuple(self.changes)
        if not all(
            isinstance(item, AdaptiveWeightChange)
            for item in changes
        ):
            raise ValueError("changes contains invalid items")

        keys = [item.strategy_key for item in changes]
        if len(keys) != len(set(keys)):
            raise ValueError("changes contains duplicate strategies")

        expected_order = tuple(
            sorted(
                changes,
                key=lambda item: (
                    -item.weight.normalized_weight,
                    item.weight.rank_position,
                    item.weight.strategy_type,
                    item.weight.strategy_name,
                ),
            )
        )
        if changes != expected_order:
            raise ValueError(
                "changes must be ordered by normalized weight"
            )

        if strategy_type is not None and any(
            item.weight.strategy_type != strategy_type
            for item in changes
        ):
            raise ValueError(
                "change strategy_type does not match report filter"
            )

        if changes and any(
            item.weight.revision != revision
            for item in changes
        ):
            raise ValueError(
                "all adaptive weights must match report revision"
            )

        normalized_total = sum(
            item.weight.normalized_weight
            for item in changes
        )
        if changes and not math.isclose(
            normalized_total,
            1.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "normalized weights must sum to 1.0"
            )

        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")

        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "strategy_type", strategy_type)
        object.__setattr__(self, "history_limit", history_limit)
        object.__setattr__(
            self,
            "generated_at_kst",
            generated_at_kst,
        )
        object.__setattr__(self, "changes", changes)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def strategy_count(self) -> int:
        return len(self.changes)

    @property
    def raised_count(self) -> int:
        return sum(
            item.direction == "RAISED"
            for item in self.changes
        )

    @property
    def lowered_count(self) -> int:
        return sum(
            item.direction == "LOWERED"
            for item in self.changes
        )

    @property
    def unchanged_count(self) -> int:
        return sum(
            item.direction == "UNCHANGED"
            for item in self.changes
        )

    @property
    def normalized_total(self) -> float:
        return sum(
            item.weight.normalized_weight
            for item in self.changes
        )

    def get(
        self,
        strategy_type: str,
        strategy_name: str,
    ) -> AdaptiveWeightChange | None:
        key = (
            _required_text(
                strategy_type,
                field_name="strategy_type",
            ).lower(),
            _required_text(
                strategy_name,
                field_name="strategy_name",
            ),
        )
        return next(
            (
                item
                for item in self.changes
                if item.strategy_key == key
            ),
            None,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "revision": list(self.revision),
            "strategy_type": self.strategy_type,
            "history_limit": self.history_limit,
            "generated_at_kst": self.generated_at_kst,
            "strategy_count": self.strategy_count,
            "raised_count": self.raised_count,
            "lowered_count": self.lowered_count,
            "unchanged_count": self.unchanged_count,
            "normalized_total": round(
                self.normalized_total,
                6,
            ),
            "changes": [
                item.as_dict()
                for item in self.changes
            ],
            "metadata": dict(self.metadata),
        }


class AdaptiveWeightReporter:
    """Validate and explain already-calculated adaptive weights."""

    def build(
        self,
        *,
        weights: Sequence[AdaptiveWeight],
        strategy_type: str | None = None,
        history_limit: int = 100,
        generated_at_kst: str | None = None,
        revision: AdaptiveRevision | None = None,
    ) -> AdaptiveWeightReport:
        history_limit = _positive_integer(
            history_limit,
            field_name="history_limit",
        )
        normalized_type = (
            None
            if strategy_type is None
            else _required_text(
                strategy_type,
                field_name="strategy_type",
            ).lower()
        )

        ordered_weights = tuple(weights)
        if not all(
            isinstance(item, AdaptiveWeight)
            for item in ordered_weights
        ):
            raise ValueError("weights contains invalid items")

        if revision is None:
            if ordered_weights:
                revision = ordered_weights[0].revision
            else:
                revision = (0, 0)

        timestamp = generated_at_kst
        if timestamp is None:
            timestamp = datetime.now(_KST).isoformat(
                timespec="seconds"
            )
        else:
            timestamp = _required_text(
                timestamp,
                field_name="generated_at_kst",
            )

        changes = tuple(
            AdaptiveWeightChange(
                weight=item,
                delta=(
                    item.current_weight
                    - item.previous_weight
                ),
                direction=self._direction(
                    item.current_weight
                    - item.previous_weight
                ),
            )
            for item in ordered_weights
        )

        return AdaptiveWeightReport(
            revision=revision,
            strategy_type=normalized_type,
            history_limit=history_limit,
            generated_at_kst=timestamp,
            changes=changes,
            metadata={
                "source": "lrp.learning",
                "reporter": "E-005B",
                "calculation_owner": "AdaptiveWeightEngine",
                "storage": "memory_only",
                "validation": {
                    "revision_consistency": True,
                    "unique_strategy_keys": True,
                    "normalized_total": (
                        round(
                            sum(
                                item.normalized_weight
                                for item in ordered_weights
                            ),
                            12,
                        )
                        if ordered_weights
                        else 0.0
                    ),
                },
            },
        )

    @staticmethod
    def _direction(delta: float) -> str:
        if delta > 1e-12:
            return "RAISED"
        if delta < -1e-12:
            return "LOWERED"
        return "UNCHANGED"
