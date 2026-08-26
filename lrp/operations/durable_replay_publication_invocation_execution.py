from __future__ import annotations

from pathlib import Path

from lrp.operations.durable_replay_publication_invocation_json_file_carrier import (
    DurableReplayPublicationInvocationJsonFileCarrier,
)
from lrp.operations.durable_replay_publication_invocation_transport import (
    DurableReplayPublicationInvocationTransportCodec,
)
from lrp.operations.durable_replay_publication_lifecycle_entrypoint import (
    DurableReplayPublicationLifecycleEntrypoint,
)
from lrp.production.production_lifecycle import ProductionLifecycleStageResult


class DurableReplayPublicationInvocationExecutionService:
    def __init__(
        self,
        file_carrier: DurableReplayPublicationInvocationJsonFileCarrier,
        transport_codec: DurableReplayPublicationInvocationTransportCodec,
        lifecycle_entrypoint: DurableReplayPublicationLifecycleEntrypoint,
    ) -> None:
        self._file_carrier = file_carrier
        self._transport_codec = transport_codec
        self._lifecycle_entrypoint = lifecycle_entrypoint

    def execute(
        self,
        path: str | Path,
    ) -> ProductionLifecycleStageResult:
        transport = self._file_carrier.read(path)
        request = self._transport_codec.decode(transport)
        return self._lifecycle_entrypoint.run(request)