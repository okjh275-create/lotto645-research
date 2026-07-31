from __future__ import annotations

import json
from collections.abc import Mapping

from lrp.evolution.contracts.snapshot_schema import (
    LearningCycleSnapshot,
)
from lrp.evolution.serialization.snapshot_codec import (
    SnapshotCodec,
)


class JsonSnapshotSerializer:
    """Serialize learning snapshots to deterministic JSON."""

    def __init__(
        self,
        codec: SnapshotCodec | None = None,
    ) -> None:
        if (
            codec is not None
            and not isinstance(
                codec,
                SnapshotCodec,
            )
        ):
            raise TypeError(
                "codec must be a SnapshotCodec"
            )

        self._codec = (
            codec
            if codec is not None
            else SnapshotCodec()
        )

    def serialize(
        self,
        snapshot: LearningCycleSnapshot,
    ) -> str:
        payload = self._codec.encode(
            snapshot
        )

        return json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def deserialize(
        self,
        serialized: str,
    ) -> LearningCycleSnapshot:
        if not isinstance(serialized, str):
            raise TypeError(
                "serialized must be a string"
            )

        if not serialized.strip():
            raise ValueError(
                "serialized must not be empty"
            )

        try:
            payload = json.loads(serialized)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "serialized must contain valid JSON"
            ) from exc

        if not isinstance(payload, Mapping):
            raise TypeError(
                "snapshot JSON root must be an object"
            )

        return self._codec.decode(payload)
