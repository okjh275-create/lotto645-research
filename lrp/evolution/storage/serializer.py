from __future__ import annotations

import json
from typing import Any


from lrp.evolution.storage.snapshot import (
    EvolutionSnapshot,
)


class SnapshotSerializationError(ValueError):
    """Raised when a snapshot cannot be serialized or decoded."""


class EvolutionSnapshotSerializer:
    """Serialize evolution snapshots as deterministic JSON."""

    def dumps(
        self,
        snapshot: EvolutionSnapshot,
    ) -> str:
        if not isinstance(snapshot, EvolutionSnapshot):
            raise TypeError(
                "snapshot must be an EvolutionSnapshot"
            )

        try:
            return json.dumps(
                snapshot.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ) + "\n"
        except (TypeError, ValueError) as exc:
            raise SnapshotSerializationError(
                "failed to serialize evolution snapshot"
            ) from exc

    def loads(
        self,
        content: str,
    ) -> EvolutionSnapshot:
        if not isinstance(content, str):
            raise TypeError("content must be a string")

        try:
            payload: Any = json.loads(content)
        except json.JSONDecodeError as exc:
            raise SnapshotSerializationError(
                "invalid snapshot JSON"
            ) from exc

        if not isinstance(payload, dict):
            raise SnapshotSerializationError(
                "snapshot JSON root must be an object"
            )

        try:
            return EvolutionSnapshot.from_dict(payload)
        except (TypeError, ValueError, KeyError) as exc:
            raise SnapshotSerializationError(
                "invalid evolution snapshot payload"
            ) from exc