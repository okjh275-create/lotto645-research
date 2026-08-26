from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import pytest

from lrp.operations.durable_replay_promotion_publication_request import (
    DurableReplayPromotionPublicationRequest,
)
from lrp.operations.durable_replay_publication_invocation_transport import (
    DurableReplayPublicationInvocationTransport,
    DurableReplayPublicationInvocationTransportCodec,
)
from lrp.operations.durable_replay_publication_invocation_json_file_carrier import (
    DurableReplayPublicationInvocationJsonFileCarrier,
)


def _transport() -> DurableReplayPublicationInvocationTransport:
    request = DurableReplayPromotionPublicationRequest(
        status="PASS",
        round_count=9,
        candidate_model_name="candidate-model",
        baseline_model_name="baseline-model",
        recommendation="eligible",
        action="prepare_publish",
        window=MappingProxyType(
            {
                "start_round": 1223,
                "end_round": 1231,
                "unicode": "서울-✓",
                "nested": {"a": 1, "z": 2},
            }
        ),
        source_decision=Path("explicit/source_decision.json"),
        registry_root=Path("explicit/registry_root"),
    )
    return DurableReplayPublicationInvocationTransportCodec().encode(request)


class _CodecStub:
    def __init__(
        self,
        *,
        encoded: str = '{"x":1}',
        decoded: object = None,
        encode_error: Exception | None = None,
        decode_error: Exception | None = None,
    ) -> None:
        self.encoded = encoded
        self.decoded = _transport() if decoded is None else decoded
        self.encode_error = encode_error
        self.decode_error = decode_error
        self.encode_calls = []
        self.decode_calls = []

    def encode(self, transport):
        self.encode_calls.append(transport)
        if self.encode_error is not None:
            raise self.encode_error
        return self.encoded

    def decode(self, payload):
        self.decode_calls.append(payload)
        if self.decode_error is not None:
            raise self.decode_error
        return self.decoded


def test_valid_write_read_round_trip(tmp_path: Path) -> None:
    carrier = DurableReplayPublicationInvocationJsonFileCarrier()
    target = tmp_path / "carrier.json"
    transport = _transport()

    written = carrier.write(target, transport)
    restored = carrier.read(target)

    assert written == target
    assert restored == transport


@pytest.mark.parametrize("bad_path", [None, 1, 1.5, object(), [], {}])
def test_write_rejects_invalid_path_type_before_encode(
    bad_path: object,
) -> None:
    stub = _CodecStub()
    carrier = DurableReplayPublicationInvocationJsonFileCarrier(json_codec=stub)

    with pytest.raises(TypeError):
        carrier.write(bad_path, _transport())  # type: ignore[arg-type]

    assert stub.encode_calls == []


@pytest.mark.parametrize("bad_path", [None, 1, 1.5, object(), [], {}])
def test_read_rejects_invalid_path_type_before_decode(
    bad_path: object,
) -> None:
    stub = _CodecStub()
    carrier = DurableReplayPublicationInvocationJsonFileCarrier(json_codec=stub)

    with pytest.raises(TypeError):
        carrier.read(bad_path)  # type: ignore[arg-type]

    assert stub.decode_calls == []


@pytest.mark.parametrize("bad_path", ["", "."])
def test_write_rejects_ambiguous_target_before_encode(
    bad_path: str,
) -> None:
    stub = _CodecStub()
    carrier = DurableReplayPublicationInvocationJsonFileCarrier(json_codec=stub)

    with pytest.raises(ValueError):
        carrier.write(bad_path, _transport())

    assert stub.encode_calls == []


@pytest.mark.parametrize("bad_path", ["", "."])
def test_read_rejects_ambiguous_source_before_decode(
    bad_path: str,
) -> None:
    stub = _CodecStub()
    carrier = DurableReplayPublicationInvocationJsonFileCarrier(json_codec=stub)

    with pytest.raises(ValueError):
        carrier.read(bad_path)

    assert stub.decode_calls == []


def test_write_missing_parent_fails_without_encoding_side_effect(
    tmp_path: Path,
) -> None:
    stub = _CodecStub()
    carrier = DurableReplayPublicationInvocationJsonFileCarrier(json_codec=stub)
    target = tmp_path / "missing" / "carrier.json"

    with pytest.raises(FileNotFoundError):
        carrier.write(target, _transport())

    assert not target.parent.exists()


def test_write_existing_target_preserves_existing_bytes(tmp_path: Path) -> None:
    carrier = DurableReplayPublicationInvocationJsonFileCarrier()
    target = tmp_path / "carrier.json"
    target.write_bytes(b"sentinel")

    with pytest.raises(FileExistsError):
        carrier.write(target, _transport())

    assert target.read_bytes() == b"sentinel"


def test_write_codec_exception_propagates_and_creates_no_file(
    tmp_path: Path,
) -> None:
    expected = RuntimeError("encode-failure")
    stub = _CodecStub(encode_error=expected)
    carrier = DurableReplayPublicationInvocationJsonFileCarrier(json_codec=stub)
    target = tmp_path / "carrier.json"

    with pytest.raises(RuntimeError) as captured:
        carrier.write(target, _transport())

    assert captured.value is expected
    assert not target.exists()


def test_write_calls_encode_exactly_once(tmp_path: Path) -> None:
    stub = _CodecStub()
    carrier = DurableReplayPublicationInvocationJsonFileCarrier(json_codec=stub)
    target = tmp_path / "carrier.json"
    transport = _transport()

    carrier.write(target, transport)

    assert stub.encode_calls == [transport]


def test_write_is_deterministic_for_same_payload_in_fresh_paths(
    tmp_path: Path,
) -> None:
    carrier = DurableReplayPublicationInvocationJsonFileCarrier()
    transport = _transport()
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    carrier.write(first, transport)
    carrier.write(second, transport)

    assert first.read_bytes() == second.read_bytes()


def test_write_has_no_bom_and_exact_single_lf(tmp_path: Path) -> None:
    carrier = DurableReplayPublicationInvocationJsonFileCarrier()
    target = tmp_path / "carrier.json"

    carrier.write(target, _transport())
    raw = target.read_bytes()

    assert not raw.startswith(b"\xef\xbb\xbf")
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    assert not raw.endswith(b"\r\n")


@pytest.mark.parametrize(
    ("raw_suffix", "accepted"),
    [
        (b"", True),
        (b"\n", True),
        (b"\r\n", True),
        (b"\r", False),
        (b"\n\n", False),
        (b"\r\n\r\n", False),
        (b"\n\r\n", False),
    ],
)
def test_read_physical_newline_envelope(
    tmp_path: Path,
    raw_suffix: bytes,
    accepted: bool,
) -> None:
    class Stub:
        def encode(self, value):
            raise AssertionError("encode not expected")

        def decode(self, payload):
            return payload

    carrier = DurableReplayPublicationInvocationJsonFileCarrier(json_codec=Stub())
    source = tmp_path / "carrier.json"
    source.write_bytes(b'{"x":1}' + raw_suffix)

    if accepted:
        assert carrier.read(source) == '{"x":1}'
    else:
        with pytest.raises(ValueError):
            carrier.read(source)


def test_read_rejects_bom_before_decode(tmp_path: Path) -> None:
    stub = _CodecStub()
    carrier = DurableReplayPublicationInvocationJsonFileCarrier(json_codec=stub)
    source = tmp_path / "carrier.json"
    source.write_bytes(b"\xef\xbb\xbf" + b'{"x":1}')

    with pytest.raises(ValueError):
        carrier.read(source)

    assert stub.decode_calls == []


def test_read_invalid_utf8_fails_before_decode(tmp_path: Path) -> None:
    stub = _CodecStub()
    carrier = DurableReplayPublicationInvocationJsonFileCarrier(json_codec=stub)
    source = tmp_path / "carrier.json"
    source.write_bytes(b'{"x":"\xff"}')

    with pytest.raises(UnicodeDecodeError):
        carrier.read(source)

    assert stub.decode_calls == []


def test_read_missing_source_fails_before_decode(tmp_path: Path) -> None:
    stub = _CodecStub()
    carrier = DurableReplayPublicationInvocationJsonFileCarrier(json_codec=stub)

    with pytest.raises(FileNotFoundError):
        carrier.read(tmp_path / "missing.json")

    assert stub.decode_calls == []


def test_read_directory_fails_before_decode(tmp_path: Path) -> None:
    stub = _CodecStub()
    carrier = DurableReplayPublicationInvocationJsonFileCarrier(json_codec=stub)
    directory = tmp_path / "dir"
    directory.mkdir()

    with pytest.raises(IsADirectoryError):
        carrier.read(directory)

    assert stub.decode_calls == []


def test_read_calls_decode_exactly_once_with_envelope_removed(
    tmp_path: Path,
) -> None:
    stub = _CodecStub()
    carrier = DurableReplayPublicationInvocationJsonFileCarrier(json_codec=stub)
    source = tmp_path / "carrier.json"
    source.write_bytes(b'{"x":1}\r\n')

    result = carrier.read(source)

    assert stub.decode_calls == ['{"x":1}']
    assert result == stub.decoded


def test_read_codec_exception_propagates_unchanged(tmp_path: Path) -> None:
    expected = RuntimeError("decode-failure")
    stub = _CodecStub(decode_error=expected)
    carrier = DurableReplayPublicationInvocationJsonFileCarrier(json_codec=stub)
    source = tmp_path / "carrier.json"
    source.write_bytes(b'{"x":1}\n')

    with pytest.raises(RuntimeError) as captured:
        carrier.read(source)

    assert captured.value is expected


def test_read_result_identity_is_preserved(tmp_path: Path) -> None:
    sentinel = _transport()
    stub = _CodecStub(decoded=sentinel)
    carrier = DurableReplayPublicationInvocationJsonFileCarrier(json_codec=stub)
    source = tmp_path / "carrier.json"
    source.write_bytes(b'{"x":1}')

    result = carrier.read(source)

    assert result is sentinel


def test_relative_path_is_preserved_on_write(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    carrier = DurableReplayPublicationInvocationJsonFileCarrier()
    relative = Path("carrier.json")

    result = carrier.write(relative, _transport())

    assert result == relative
    assert not result.is_absolute()
    assert relative.exists()


def test_product_has_exact_dependency_boundary() -> None:
    source = Path(
        "lrp/operations/durable_replay_publication_invocation_json_file_carrier.py"
    ).read_text(encoding="utf-8-sig")

    assert "from pathlib import Path" in source
    assert (
        "lrp.operations.durable_replay_publication_invocation_transport"
        in source
    )
    assert (
        "lrp.operations.durable_replay_publication_invocation_json_presentation"
        in source
    )


def test_product_has_no_json_cli_execution_discovery_policy_or_replace_surface() -> None:
    source = Path(
        "lrp/operations/durable_replay_publication_invocation_json_file_carrier.py"
    ).read_text(encoding="utf-8-sig")

    forbidden = (
        "json.dumps",
        "json.loads",
        "argparse",
        "lrp.cli",
        "stdin",
        "stdout",
        "DurableReplayPublicationLifecycleEntrypoint",
        "ProductionChampionRegistryPublisher",
        "run_publication_stage",
        "discover",
        "latest",
        "getenv",
        "environ",
        "resolve()",
        "expanduser",
        "candidate_advantage_count",
        "baseline_advantage_count",
        "baseline_delta",
        "eligibility",
        "promotion_policy",
        "rollback",
        "mkdir(",
        "replace(",
        "os.replace",
    )
    for token in forbidden:
        assert token not in source
