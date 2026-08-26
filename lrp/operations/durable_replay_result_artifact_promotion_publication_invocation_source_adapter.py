from __future__ import annotations

from pathlib import Path

from lrp.operations.durable_replay_publication_invocation_json_file_carrier import (
    DurableReplayPublicationInvocationJsonFileCarrier,
)
from lrp.operations.durable_replay_publication_invocation_transport import (
    DurableReplayPublicationInvocationTransportCodec,
)
from lrp.operations.durable_replay_result_artifact_promotion_publication_request_source_adapter import (
    DurableReplayResultArtifactPromotionPublicationRequestSourceAdapter,
)


class DurableReplayResultArtifactPromotionPublicationInvocationSourceAdapter:
    def __init__(
        self,
        source_adapter: DurableReplayResultArtifactPromotionPublicationRequestSourceAdapter
        | None = None,
        transport_codec: DurableReplayPublicationInvocationTransportCodec
        | None = None,
        file_carrier: DurableReplayPublicationInvocationJsonFileCarrier
        | None = None,
    ) -> None:
        self._source_adapter = (
            source_adapter
            if source_adapter is not None
            else DurableReplayResultArtifactPromotionPublicationRequestSourceAdapter()
        )
        self._transport_codec = (
            transport_codec
            if transport_codec is not None
            else DurableReplayPublicationInvocationTransportCodec()
        )
        self._file_carrier = (
            file_carrier
            if file_carrier is not None
            else DurableReplayPublicationInvocationJsonFileCarrier()
        )

    def adapt(
        self,
        artifact_root: str | Path,
        end_round: int,
        *,
        source_decision: str | Path,
        registry_root: str | Path,
        output_path: str | Path,
    ) -> Path:
        request = self._source_adapter.adapt(
            artifact_root,
            end_round,
            source_decision=source_decision,
            registry_root=registry_root,
        )
        transport = self._transport_codec.encode(
            request
        )
        return self._file_carrier.write(
            output_path,
            transport,
        )