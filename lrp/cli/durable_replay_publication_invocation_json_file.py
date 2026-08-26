from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from lrp.operations.durable_replay_publication_invocation_json_file_carrier import (
    DurableReplayPublicationInvocationJsonFileCarrier,
)
from lrp.operations.durable_replay_publication_invocation_json_presentation import (
    DurableReplayPublicationInvocationJsonCodec,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="durable-replay-publication-invocation-json-file",
    )
    parser.add_argument("--input", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    carrier = DurableReplayPublicationInvocationJsonFileCarrier()
    transport = carrier.read(args.input)

    codec = DurableReplayPublicationInvocationJsonCodec()
    payload = codec.encode(transport)

    sys.stdout.write(payload + "\n")
    return 0
