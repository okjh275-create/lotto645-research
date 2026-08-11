from __future__ import annotations

import json

from lrp.regimes.bayesian_snapshot import (
    RegimeBayesianSnapshot,
)


class RegimeBayesianSerializationError(ValueError):
    """Raised when regime Bayesian serialization fails."""


class RegimeBayesianSnapshotSerializer:
    """Serialize regime Bayesian snapshots as JSON."""

    def dumps(
        self,
        snapshot: RegimeBayesianSnapshot,
    ) -> str:
        if not isinstance(
            snapshot,
            RegimeBayesianSnapshot,
        ):
            raise TypeError(
                "snapshot must be a RegimeBayesianSnapshot"
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
    ) -> RegimeBayesianSnapshot:
        if not isinstance(content, str):
            raise TypeError(
                "content must be a string"
            )

        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RegimeBayesianSerializationError(
                "invalid regime Bayesian JSON"
            ) from exc

        try:
            return RegimeBayesianSnapshot.from_dict(
                payload
            )
        except (
            TypeError,
            ValueError,
            KeyError,
        ) as exc:
            raise RegimeBayesianSerializationError(
                "invalid regime Bayesian snapshot"
            ) from exc