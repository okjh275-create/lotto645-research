from __future__ import annotations

import argparse
from collections.abc import Sequence

from lrp.operations.durable_replay_publication_invocation_execution import (
    DurableReplayPublicationInvocationExecutionService,
)
from lrp.operations.durable_replay_publication_invocation_json_file_carrier import (
    DurableReplayPublicationInvocationJsonFileCarrier,
)
from lrp.operations.durable_replay_publication_invocation_transport import (
    DurableReplayPublicationInvocationTransportCodec,
)
from lrp.operations.durable_replay_publication_lifecycle_adaptation import (
    DurableReplayPublicationLifecycleAdaptationService,
)
from lrp.operations.durable_replay_publication_lifecycle_entrypoint import (
    DurableReplayPublicationLifecycleEntrypoint,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute one explicit durable replay publication "
            "invocation JSON file."
        )
    )
    parser.add_argument("--input", required=True)
    return parser


def _build_execution_service() -> (
    DurableReplayPublicationInvocationExecutionService
):
    file_carrier = DurableReplayPublicationInvocationJsonFileCarrier()
    transport_codec = DurableReplayPublicationInvocationTransportCodec()

    adaptation_service = (
        DurableReplayPublicationLifecycleAdaptationService()
    )

    lifecycle_entrypoint = (
        DurableReplayPublicationLifecycleEntrypoint(
            adaptation_service=adaptation_service
        )
    )

    return DurableReplayPublicationInvocationExecutionService(
        file_carrier=file_carrier,
        transport_codec=transport_codec,
        lifecycle_entrypoint=lifecycle_entrypoint,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    service = _build_execution_service()
    service.execute(args.input)
    return 0