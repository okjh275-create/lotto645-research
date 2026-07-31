from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, ClassVar, Mapping

from lrp.evolution.contracts import AdaptiveWeightProfile


@dataclass(frozen=True, slots=True)
class EvolutionSnapshot:
    """Persisted state of an evolution weight profile."""

    profile: AdaptiveWeightProfile
    saved_at: datetime
    schema_version: int = 1

    CURRENT_SCHEMA_VERSION: ClassVar[int] = 1

    def __post_init__(self) -> None:
        if not isinstance(
            self.profile,
            AdaptiveWeightProfile,
        ):
            raise TypeError(
                "profile must be an AdaptiveWeightProfile"
            )

        if not isinstance(self.saved_at, datetime):
            raise TypeError("saved_at must be a datetime")

        if self.saved_at.tzinfo is None:
            raise ValueError(
                "saved_at must be timezone-aware"
            )

        if self.saved_at.utcoffset() is None:
            raise ValueError(
                "saved_at must have a valid UTC offset"
            )

        if isinstance(self.schema_version, bool):
            raise TypeError(
                "schema_version must be an integer"
            )

        if not isinstance(self.schema_version, int):
            raise TypeError(
                "schema_version must be an integer"
            )

        if self.schema_version < 1:
            raise ValueError(
                "schema_version must be greater than "
                "or equal to 1"
            )

    @classmethod
    def create(
        cls,
        profile: AdaptiveWeightProfile,
        *,
        saved_at: datetime | None = None,
    ) -> EvolutionSnapshot:
        return cls(
            profile=profile,
            saved_at=saved_at or datetime.now(timezone.utc),
            schema_version=cls.CURRENT_SCHEMA_VERSION,
        )

    @property
    def revision(self) -> int:
        return self.profile.revision

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "saved_at": self.saved_at.isoformat(),
            "profile": self.profile.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> EvolutionSnapshot:
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")

        required = (
            "schema_version",
            "saved_at",
            "profile",
        )

        missing = [
            field_name
            for field_name in required
            if field_name not in payload
        ]

        if missing:
            names = ", ".join(missing)
            raise ValueError(
                f"missing snapshot fields: {names}"
            )

        schema_version = payload["schema_version"]

        if isinstance(schema_version, bool):
            raise TypeError(
                "schema_version must be an integer"
            )

        if not isinstance(schema_version, int):
            raise TypeError(
                "schema_version must be an integer"
            )

        if schema_version != cls.CURRENT_SCHEMA_VERSION:
            raise ValueError(
                "unsupported snapshot schema version: "
                f"{schema_version}"
            )

        saved_at_raw = payload["saved_at"]

        if not isinstance(saved_at_raw, str):
            raise ValueError(
                "saved_at must be an ISO-8601 string"
            )

        try:
            saved_at = datetime.fromisoformat(saved_at_raw)
        except ValueError as exc:
            raise ValueError(
                "saved_at must be a valid ISO-8601 string"
            ) from exc

        profile_raw = payload["profile"]

        if not isinstance(profile_raw, Mapping):
            raise ValueError(
                "profile must be a mapping"
            )

        return cls(
            profile=AdaptiveWeightProfile.from_dict(
                profile_raw
            ),
            saved_at=saved_at,
            schema_version=schema_version,
        )