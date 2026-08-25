from __future__ import annotations

from collections.abc import Mapping
import json

from lrp.operations.durable_replay_publication_invocation_transport import (
    DurableReplayPublicationInvocationTransport,
    DurableReplayPublicationInvocationTransportCodec,
)


class DurableReplayPublicationInvocationJsonCodec:
    """Canonical JSON-text presentation over the durable replay transport."""

    def __init__(self) -> None:
        self._transport_codec = DurableReplayPublicationInvocationTransportCodec()

    def encode(
        self,
        transport: DurableReplayPublicationInvocationTransport,
    ) -> str:
        mapping = self._transport_codec.to_mapping(transport)
        try:
            return json.dumps(
                mapping,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("transport is not canonically JSON encodable") from exc

    def decode(
        self,
        payload: str,
    ) -> DurableReplayPublicationInvocationTransport:
        if not isinstance(payload, str):
            raise TypeError("payload must be str")

        try:
            decoded = json.loads(
                payload,
                object_pairs_hook=self._object_from_pairs,
                parse_constant=self._reject_non_finite_constant,
            )
        except json.JSONDecodeError as exc:
            raise ValueError("payload must be valid JSON") from exc

        if not isinstance(decoded, Mapping):
            raise TypeError("JSON root must be an object")

        return self._transport_codec.from_mapping(decoded)

    @staticmethod
    def _object_from_pairs(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    @staticmethod
    def _reject_non_finite_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant: {value}")
