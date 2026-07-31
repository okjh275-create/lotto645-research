from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from types import MappingProxyType
from typing import Mapping, TypeAlias


ContextValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | None
)


@dataclass(frozen=True, slots=True)
class LearningContext:
    """Immutable state exchanged across one learning cycle."""

    cycle_id: str
    round_no: int
    version: int = 1
    signals: Mapping[str, float] | None = None
    rewards: Mapping[str, float] | None = None
    selected_policy: str | None = None
    selected_arm: str | None = None
    weights: Mapping[str, float] | None = None
    metadata: Mapping[str, ContextValue] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cycle_id",
            self._normalize_required_text(
                self.cycle_id,
                field_name="cycle_id",
            ),
        )
        object.__setattr__(
            self,
            "round_no",
            self._normalize_positive_integer(
                self.round_no,
                field_name="round_no",
            ),
        )
        object.__setattr__(
            self,
            "version",
            self._normalize_positive_integer(
                self.version,
                field_name="version",
            ),
        )
        object.__setattr__(
            self,
            "signals",
            self._normalize_numeric_mapping(
                self.signals,
                field_name="signals",
            ),
        )
        object.__setattr__(
            self,
            "rewards",
            self._normalize_numeric_mapping(
                self.rewards,
                field_name="rewards",
            ),
        )
        object.__setattr__(
            self,
            "selected_policy",
            self._normalize_optional_text(
                self.selected_policy,
                field_name="selected_policy",
            ),
        )
        object.__setattr__(
            self,
            "selected_arm",
            self._normalize_optional_text(
                self.selected_arm,
                field_name="selected_arm",
            ),
        )
        object.__setattr__(
            self,
            "weights",
            self._normalize_numeric_mapping(
                self.weights,
                field_name="weights",
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            self._normalize_metadata(
                self.metadata,
            ),
        )

    def with_signals(
        self,
        signals: Mapping[str, float],
    ) -> LearningContext:
        """Return a new context containing signal values."""

        return replace(
            self,
            signals=signals,
        )

    def with_rewards(
        self,
        rewards: Mapping[str, float],
    ) -> LearningContext:
        """Return a new context containing reward values."""

        return replace(
            self,
            rewards=rewards,
        )

    def with_selection(
        self,
        *,
        policy: str,
        arm: str,
    ) -> LearningContext:
        """Return a new context containing policy selection."""

        return replace(
            self,
            selected_policy=policy,
            selected_arm=arm,
        )

    def with_weights(
        self,
        weights: Mapping[str, float],
    ) -> LearningContext:
        """Return a new context containing adaptive weights."""

        return replace(
            self,
            weights=weights,
        )

    def with_metadata(
        self,
        metadata: Mapping[str, ContextValue],
    ) -> LearningContext:
        """Return a new context containing metadata."""

        return replace(
            self,
            metadata=metadata,
        )

    def advance_version(
        self,
    ) -> LearningContext:
        """Return a new context with an incremented version."""

        return replace(
            self,
            version=self.version + 1,
        )

    def snapshot(
        self,
    ) -> dict[str, object]:
        """Return a detached serializable representation."""

        return {
            "cycle_id": self.cycle_id,
            "round_no": self.round_no,
            "version": self.version,
            "signals": dict(self.signals),
            "rewards": dict(self.rewards),
            "selected_policy": self.selected_policy,
            "selected_arm": self.selected_arm,
            "weights": dict(self.weights),
            "metadata": dict(self.metadata),
        }

    @staticmethod
    def _normalize_required_text(
        value: str,
        *,
        field_name: str,
    ) -> str:
        if not isinstance(value, str):
            raise TypeError(
                f"{field_name} must be a string"
            )

        normalized = value.strip()

        if not normalized:
            raise ValueError(
                f"{field_name} must not be empty"
            )

        return normalized

    @classmethod
    def _normalize_optional_text(
        cls,
        value: str | None,
        *,
        field_name: str,
    ) -> str | None:
        if value is None:
            return None

        return cls._normalize_required_text(
            value,
            field_name=field_name,
        )

    @staticmethod
    def _normalize_positive_integer(
        value: int,
        *,
        field_name: str,
    ) -> int:
        if isinstance(value, bool):
            raise TypeError(
                f"{field_name} must be an integer"
            )

        if not isinstance(value, int):
            raise TypeError(
                f"{field_name} must be an integer"
            )

        if value < 1:
            raise ValueError(
                f"{field_name} must be greater than "
                "or equal to 1"
            )

        return value

    @classmethod
    def _normalize_numeric_mapping(
        cls,
        values: Mapping[str, float] | None,
        *,
        field_name: str,
    ) -> Mapping[str, float]:
        if values is None:
            return MappingProxyType({})

        if not isinstance(values, Mapping):
            raise TypeError(
                f"{field_name} must be a mapping"
            )

        normalized: dict[str, float] = {}

        for key, value in values.items():
            normalized_key = cls._normalize_required_text(
                key,
                field_name=f"{field_name} key",
            )
            normalized_value = cls._normalize_finite_number(
                value,
                field_name=(
                    f"{field_name}[{normalized_key}]"
                ),
            )

            if normalized_key in normalized:
                raise ValueError(
                    f"duplicate normalized key in "
                    f"{field_name}: {normalized_key}"
                )

            normalized[normalized_key] = normalized_value

        return MappingProxyType(normalized)

    @classmethod
    def _normalize_metadata(
        cls,
        metadata: Mapping[str, ContextValue] | None,
    ) -> Mapping[str, ContextValue]:
        if metadata is None:
            return MappingProxyType({})

        if not isinstance(metadata, Mapping):
            raise TypeError(
                "metadata must be a mapping"
            )

        normalized: dict[str, ContextValue] = {}

        for key, value in metadata.items():
            normalized_key = cls._normalize_required_text(
                key,
                field_name="metadata key",
            )

            if not cls._is_context_value(value):
                raise TypeError(
                    "metadata values must be scalar "
                    "context values"
                )

            if (
                isinstance(value, float)
                and not isfinite(value)
            ):
                raise ValueError(
                    f"metadata[{normalized_key}] "
                    "must be finite"
                )

            if normalized_key in normalized:
                raise ValueError(
                    "duplicate normalized key in "
                    f"metadata: {normalized_key}"
                )

            normalized[normalized_key] = value

        return MappingProxyType(normalized)

    @staticmethod
    def _normalize_finite_number(
        value: float,
        *,
        field_name: str,
    ) -> float:
        if isinstance(value, bool):
            raise TypeError(
                f"{field_name} must be numeric"
            )

        if not isinstance(value, (int, float)):
            raise TypeError(
                f"{field_name} must be numeric"
            )

        normalized = float(value)

        if not isfinite(normalized):
            raise ValueError(
                f"{field_name} must be finite"
            )

        return normalized

    @staticmethod
    def _is_context_value(
        value: object,
    ) -> bool:
        return (
            value is None
            or isinstance(
                value,
                (str, int, float, bool),
            )
        )
