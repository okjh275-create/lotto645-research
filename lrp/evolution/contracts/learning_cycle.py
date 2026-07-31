from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from lrp.evolution.contracts.learning_context import (
    LearningContext,
)


@dataclass(frozen=True, slots=True)
class LearningCycleStep:
    """Immutable execution trace for one learning-cycle step."""

    index: int
    name: str
    version_before: int
    version_after: int
    reward_key: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "index",
            self._normalize_positive_integer(
                self.index,
                field_name="index",
            ),
        )
        object.__setattr__(
            self,
            "name",
            self._normalize_required_text(
                self.name,
                field_name="name",
            ),
        )
        object.__setattr__(
            self,
            "version_before",
            self._normalize_positive_integer(
                self.version_before,
                field_name="version_before",
            ),
        )
        object.__setattr__(
            self,
            "version_after",
            self._normalize_positive_integer(
                self.version_after,
                field_name="version_after",
            ),
        )
        object.__setattr__(
            self,
            "reward_key",
            self._normalize_required_text(
                self.reward_key,
                field_name="reward_key",
            ),
        )

        if self.version_after <= self.version_before:
            raise ValueError(
                "version_after must be greater than "
                "version_before"
            )

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


@dataclass(frozen=True, slots=True)
class LearningCycleResult:
    """Immutable result returned after one learning cycle."""

    initial_context: LearningContext
    final_context: LearningContext
    steps: tuple[LearningCycleStep, ...]
    metadata: Mapping[str, str | int | float | bool | None] | None = None

    def __post_init__(self) -> None:
        if not isinstance(
            self.initial_context,
            LearningContext,
        ):
            raise TypeError(
                "initial_context must be a LearningContext"
            )

        if not isinstance(
            self.final_context,
            LearningContext,
        ):
            raise TypeError(
                "final_context must be a LearningContext"
            )

        if not isinstance(self.steps, tuple):
            raise TypeError(
                "steps must be a tuple"
            )

        for step in self.steps:
            if not isinstance(
                step,
                LearningCycleStep,
            ):
                raise TypeError(
                    "steps must contain only "
                    "LearningCycleStep values"
                )

        if (
            self.initial_context.cycle_id
            != self.final_context.cycle_id
        ):
            raise ValueError(
                "cycle_id must not change during "
                "a learning cycle"
            )

        if (
            self.initial_context.round_no
            != self.final_context.round_no
        ):
            raise ValueError(
                "round_no must not change during "
                "a learning cycle"
            )

        if self.steps:
            if (
                self.steps[0].version_before
                != self.initial_context.version
            ):
                raise ValueError(
                    "first step version_before must match "
                    "initial context version"
                )

            if (
                self.steps[-1].version_after
                != self.final_context.version
            ):
                raise ValueError(
                    "last step version_after must match "
                    "final context version"
                )

            for previous, current in zip(
                self.steps,
                self.steps[1:],
            ):
                if (
                    previous.version_after
                    != current.version_before
                ):
                    raise ValueError(
                        "learning-cycle step versions "
                        "must be contiguous"
                    )

        elif (
            self.initial_context.version
            != self.final_context.version
        ):
            raise ValueError(
                "an empty cycle must not change version"
            )

        object.__setattr__(
            self,
            "metadata",
            self._normalize_metadata(
                self.metadata,
            ),
        )

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def version_delta(self) -> int:
        return (
            self.final_context.version
            - self.initial_context.version
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "initial_context": (
                self.initial_context.snapshot()
            ),
            "final_context": (
                self.final_context.snapshot()
            ),
            "steps": [
                {
                    "index": step.index,
                    "name": step.name,
                    "version_before": (
                        step.version_before
                    ),
                    "version_after": (
                        step.version_after
                    ),
                    "reward_key": step.reward_key,
                }
                for step in self.steps
            ],
            "metadata": dict(self.metadata),
            "step_count": self.step_count,
            "version_delta": self.version_delta,
        }

    @staticmethod
    def _normalize_metadata(
        metadata: (
            Mapping[
                str,
                str | int | float | bool | None,
            ]
            | None
        ),
    ) -> Mapping[
        str,
        str | int | float | bool | None,
    ]:
        if metadata is None:
            return MappingProxyType({})

        if not isinstance(metadata, Mapping):
            raise TypeError(
                "metadata must be a mapping"
            )

        normalized: dict[
            str,
            str | int | float | bool | None,
        ] = {}

        for key, value in metadata.items():
            if not isinstance(key, str):
                raise TypeError(
                    "metadata keys must be strings"
                )

            normalized_key = key.strip()

            if not normalized_key:
                raise ValueError(
                    "metadata keys must not be empty"
                )

            if not (
                value is None
                or isinstance(
                    value,
                    (str, int, float, bool),
                )
            ):
                raise TypeError(
                    "metadata values must be scalar"
                )

            if normalized_key in normalized:
                raise ValueError(
                    "duplicate normalized metadata key: "
                    f"{normalized_key}"
                )

            normalized[normalized_key] = value

        return MappingProxyType(normalized)
