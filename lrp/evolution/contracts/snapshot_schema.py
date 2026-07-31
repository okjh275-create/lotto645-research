from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from types import MappingProxyType
from typing import Mapping, TypeAlias

from lrp.evolution.contracts.learning_cycle import (
    LearningCycleResult,
)


SnapshotMetadataValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | None
)


@dataclass(frozen=True, slots=True)
class LearningCycleSnapshot:
    """Immutable persistence schema for one learning cycle."""

    snapshot_id: str
    result: LearningCycleResult
    created_at_utc: datetime
    schema_version: int = 1
    source: str = "learning_cycle"
    metadata: (
        Mapping[str, SnapshotMetadataValue]
        | None
    ) = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "snapshot_id",
            self._normalize_required_text(
                self.snapshot_id,
                field_name="snapshot_id",
            ),
        )

        if not isinstance(
            self.result,
            LearningCycleResult,
        ):
            raise TypeError(
                "result must be a LearningCycleResult"
            )

        object.__setattr__(
            self,
            "created_at_utc",
            self._normalize_datetime(
                self.created_at_utc,
            ),
        )
        object.__setattr__(
            self,
            "schema_version",
            self._normalize_positive_integer(
                self.schema_version,
                field_name="schema_version",
            ),
        )
        object.__setattr__(
            self,
            "source",
            self._normalize_required_text(
                self.source,
                field_name="source",
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            self._normalize_metadata(
                self.metadata,
            ),
        )

    @classmethod
    def create(
        cls,
        *,
        snapshot_id: str,
        result: LearningCycleResult,
        created_at_utc: datetime | None = None,
        schema_version: int = 1,
        source: str = "learning_cycle",
        metadata: (
            Mapping[str, SnapshotMetadataValue]
            | None
        ) = None,
    ) -> LearningCycleSnapshot:
        """Create a snapshot with a UTC creation time."""

        timestamp = (
            created_at_utc
            if created_at_utc is not None
            else datetime.now(timezone.utc)
        )

        return cls(
            snapshot_id=snapshot_id,
            result=result,
            created_at_utc=timestamp,
            schema_version=schema_version,
            source=source,
            metadata=metadata,
        )

    @property
    def cycle_id(self) -> str:
        return self.result.final_context.cycle_id

    @property
    def round_no(self) -> int:
        return self.result.final_context.round_no

    @property
    def context_version(self) -> int:
        return self.result.final_context.version

    @property
    def step_count(self) -> int:
        return self.result.step_count

    def to_payload(self) -> dict[str, object]:
        """Return a detached JSON-compatible payload."""

        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "created_at_utc": (
                self._format_datetime(
                    self.created_at_utc
                )
            ),
            "source": self.source,
            "cycle_id": self.cycle_id,
            "round_no": self.round_no,
            "context_version": (
                self.context_version
            ),
            "step_count": self.step_count,
            "result": self.result.snapshot(),
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

    @staticmethod
    def _normalize_datetime(
        value: datetime,
    ) -> datetime:
        if not isinstance(value, datetime):
            raise TypeError(
                "created_at_utc must be a datetime"
            )

        if (
            value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError(
                "created_at_utc must be timezone-aware"
            )

        return value.astimezone(timezone.utc)

    @classmethod
    def _normalize_metadata(
        cls,
        metadata: (
            Mapping[str, SnapshotMetadataValue]
            | None
        ),
    ) -> Mapping[str, SnapshotMetadataValue]:
        if metadata is None:
            return MappingProxyType({})

        if not isinstance(metadata, Mapping):
            raise TypeError(
                "metadata must be a mapping"
            )

        normalized: dict[
            str,
            SnapshotMetadataValue,
        ] = {}

        for key, value in metadata.items():
            normalized_key = (
                cls._normalize_required_text(
                    key,
                    field_name="metadata key",
                )
            )

            if not cls._is_metadata_value(
                value
            ):
                raise TypeError(
                    "metadata values must be scalar"
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
                    "duplicate normalized metadata "
                    f"key: {normalized_key}"
                )

            normalized[normalized_key] = value

        return MappingProxyType(normalized)

    @staticmethod
    def _is_metadata_value(
        value: object,
    ) -> bool:
        return (
            value is None
            or isinstance(
                value,
                (str, int, float, bool),
            )
        )

    @staticmethod
    def _format_datetime(
        value: datetime,
    ) -> str:
        return (
            value.astimezone(timezone.utc)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z")
        )
