from __future__ import annotations

from pathlib import Path

from lrp.operations.durable_replay_publication_invocation_json_presentation import (
    DurableReplayPublicationInvocationJsonCodec,
)
from lrp.operations.durable_replay_publication_invocation_transport import (
    DurableReplayPublicationInvocationTransport,
)


class DurableReplayPublicationInvocationJsonFileCarrier:
    def __init__(
        self,
        json_codec: DurableReplayPublicationInvocationJsonCodec | None = None,
    ) -> None:
        self._json_codec = json_codec or DurableReplayPublicationInvocationJsonCodec()

    @staticmethod
    def _path(value: str | Path) -> Path:
        if not isinstance(value, (str, Path)):
            raise TypeError("path must be str or Path")

        path = Path(value)
        if str(path) in ("", "."):
            raise ValueError("path must identify an explicit file target")
        return path

    @staticmethod
    def _strip_file_envelope(text: str) -> str:
        if text.startswith("\ufeff"):
            raise ValueError("UTF-8 BOM is not allowed")

        if text.endswith("\r\n"):
            payload = text[:-2]
            if payload.endswith("\n") or payload.endswith("\r"):
                raise ValueError("multiple trailing newlines are not allowed")
            return payload

        if text.endswith("\n"):
            payload = text[:-1]
            if payload.endswith("\n") or payload.endswith("\r"):
                raise ValueError("multiple trailing newlines are not allowed")
            return payload

        if text.endswith("\r"):
            raise ValueError("bare carriage return is not allowed")

        return text

    def write(
        self,
        path: str | Path,
        transport: DurableReplayPublicationInvocationTransport,
    ) -> Path:
        target = self._path(path)
        payload = self._json_codec.encode(transport)

        with target.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload + "\n")

        return target

    def read(
        self,
        path: str | Path,
    ) -> DurableReplayPublicationInvocationTransport:
        source = self._path(path)

        if not source.exists():
            raise FileNotFoundError(str(source))
        if not source.is_file():
            raise IsADirectoryError(str(source))

        with source.open("r", encoding="utf-8", newline="") as handle:
            text = handle.read()

        payload = self._strip_file_envelope(text)
        return self._json_codec.decode(payload)
