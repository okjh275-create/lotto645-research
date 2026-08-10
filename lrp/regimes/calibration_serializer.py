from __future__ import annotations

import json

from lrp.regimes.calibration_snapshot import (
    RegimeCalibrationSnapshot,
)


class RegimeCalibrationSerializationError(ValueError):
    """Raised when regime calibration serialization fails."""


class RegimeCalibrationSnapshotSerializer:
    """Serialize regime calibration snapshots as JSON."""

    def dumps(
        self,
        snapshot: RegimeCalibrationSnapshot,
    ) -> str:
        if not isinstance(
            snapshot,
            RegimeCalibrationSnapshot,
        ):
            raise TypeError(
                "snapshot must be a RegimeCalibrationSnapshot"
            )

        return json.dumps(
            snapshot.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"

    def loads(
        self,
        content: str,
    ) -> RegimeCalibrationSnapshot:
        if not isinstance(content, str):
            raise TypeError(
                "content must be a string"
            )

        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RegimeCalibrationSerializationError(
                "invalid regime calibration JSON"
            ) from exc

        try:
            return RegimeCalibrationSnapshot.from_dict(
                payload
            )
        except (
            TypeError,
            ValueError,
            KeyError,
        ) as exc:
            raise RegimeCalibrationSerializationError(
                "invalid regime calibration snapshot"
            ) from exc
