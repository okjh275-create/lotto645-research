"""Read-only consumer for durable replay result artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from lrp.operations.runtime import verify_manifest


@dataclass(frozen=True)
class DurableReplayResultArtifactConsumerRequest:
    artifact_root: str | Path
    end_round: int


class DurableReplayResultArtifactConsumer:
    def consume(
        self,
        *,
        request: DurableReplayResultArtifactConsumerRequest,
    ) -> Mapping[str, Any]:
        if not isinstance(
            request,
            DurableReplayResultArtifactConsumerRequest,
        ):
            raise TypeError(
                "request must be "
                "DurableReplayResultArtifactConsumerRequest"
            )

        directory = (
            Path(request.artifact_root)
            / "durable-replay-evaluations"
            / f"round_{request.end_round:04d}"
        )

        manifest_path = directory / "manifest.json"
        result_path = directory / "evaluation_result.json"

        verification = verify_manifest(manifest_path)

        if verification.get("status") != "PASS":
            raise ValueError(
                "result artifact manifest verification failed"
            )

        payload = json.loads(
            result_path.read_text(
                encoding="utf-8-sig"
            )
        )

        if not isinstance(payload, dict):
            raise TypeError(
                "result artifact top-level JSON must be an object"
            )

        return MappingProxyType(payload)
