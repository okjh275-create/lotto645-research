from __future__ import annotations

import importlib
import importlib.util
import inspect
from dataclasses import fields
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
from lrp.operations.durable_replay_publication_invocation_json_presentation import (
    DurableReplayPublicationInvocationJsonCodec,
)

MODULE_NAME = "lrp.operations.durable_replay_publication_invocation_json_file_carrier"
CLASS_NAME = "DurableReplayPublicationInvocationJsonFileCarrier"


def _module():
    return importlib.import_module(MODULE_NAME)


def _carrier_class():
    return getattr(_module(), CLASS_NAME)


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
                "mode": "durable-replay",
                "unicode": "서울-✓",
            }
        ),
        source_decision=Path("explicit/source_decision.json"),
        registry_root=Path("explicit/registry_root"),
    )
    return DurableReplayPublicationInvocationTransportCodec().encode(request)


def test_json_file_carrier_product_module_exists() -> None:
    assert importlib.util.find_spec(MODULE_NAME) is not None


def test_json_file_carrier_class_exists() -> None:
    assert hasattr(_module(), CLASS_NAME)


def test_public_methods_are_exact() -> None:
    cls = _carrier_class()
    public = {
        name
        for name, value in vars(cls).items()
        if callable(value) and not name.startswith("_")
    }
    assert public == {"write", "read"}


def test_write_signature_is_exact() -> None:
    sig = inspect.signature(_carrier_class().write)
    assert tuple(sig.parameters) == ("self", "path", "transport")
    assert sig.parameters["path"].annotation in {
        "str | Path",
        str | Path,
    }
    assert sig.parameters["transport"].annotation in {
        "DurableReplayPublicationInvocationTransport",
        DurableReplayPublicationInvocationTransport,
    }
    assert sig.return_annotation in {"Path", Path}


def test_read_signature_is_exact() -> None:
    sig = inspect.signature(_carrier_class().read)
    assert tuple(sig.parameters) == ("self", "path")
    assert sig.parameters["path"].annotation in {
        "str | Path",
        str | Path,
    }
    assert sig.return_annotation in {
        "DurableReplayPublicationInvocationTransport",
        DurableReplayPublicationInvocationTransport,
    }


def test_service_owns_or_receives_bf_codec_dependency() -> None:
    cls = _carrier_class()
    sig = inspect.signature(cls)
    assert "json_codec" in sig.parameters


def test_write_delegates_exact_transport_to_bf_once(tmp_path: Path) -> None:
    transport = _transport()

    class Stub:
        def __init__(self) -> None:
            self.calls = []

        def encode(self, value):
            self.calls.append(value)
            return '{"x":1}'

        def decode(self, payload):
            raise AssertionError("decode not expected")

    stub = Stub()
    carrier = _carrier_class()(json_codec=stub)

    target = tmp_path / "carrier.json"
    result = carrier.write(target, transport)

    assert stub.calls == [transport]
    assert result == target


def test_write_uses_exclusive_create_and_blocks_overwrite(tmp_path: Path) -> None:
    carrier = _carrier_class()()
    target = tmp_path / "carrier.json"
    transport = _transport()

    carrier.write(target, transport)

    with pytest.raises(FileExistsError):
        carrier.write(target, transport)


def test_write_uses_utf8_without_bom_and_exact_one_lf(tmp_path: Path) -> None:
    carrier = _carrier_class()()
    target = tmp_path / "carrier.json"
    transport = _transport()

    carrier.write(target, transport)
    raw = target.read_bytes()

    assert not raw.startswith(b"\xef\xbb\xbf")
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    assert "서울-✓".encode("utf-8") in raw


@pytest.mark.parametrize("path_value", ["", "."])
def test_write_rejects_empty_or_dot_path(path_value: str) -> None:
    carrier = _carrier_class()()
    with pytest.raises((TypeError, ValueError)):
        carrier.write(path_value, _transport())


def test_write_rejects_missing_parent_without_mkdir(tmp_path: Path) -> None:
    carrier = _carrier_class()()
    target = tmp_path / "missing" / "carrier.json"

    with pytest.raises(FileNotFoundError):
        carrier.write(target, _transport())

    assert not target.parent.exists()


def test_read_delegates_payload_to_bf_decode_once(tmp_path: Path) -> None:
    class Stub:
        def __init__(self) -> None:
            self.payloads = []

        def encode(self, value):
            raise AssertionError("encode not expected")

        def decode(self, payload):
            self.payloads.append(payload)
            return _transport()

    stub = Stub()
    carrier = _carrier_class()(json_codec=stub)
    source = tmp_path / "carrier.json"
    source.write_text('{"x":1}\n', encoding="utf-8", newline="\n")

    result = carrier.read(source)

    assert stub.payloads == ['{"x":1}']
    assert result == _transport()


@pytest.mark.parametrize(
    ("suffix", "accepted"),
    [
        ("", True),
        ("\n", True),
        ("\r\n", True),
        ("\n\n", False),
        ("\n\r\n", False),
        ("\r\n\r\n", False),
        ("\r", False),
    ],
)
def test_read_newline_envelope_policy(
    tmp_path: Path,
    suffix: str,
    accepted: bool,
) -> None:
    carrier = _carrier_class()()
    source = tmp_path / "carrier.json"
    payload = DurableReplayPublicationInvocationJsonCodec().encode(_transport())
    source.write_bytes((payload + suffix).encode("utf-8"))

    if accepted:
        assert carrier.read(source) == _transport()
    else:
        with pytest.raises((TypeError, ValueError)):
            carrier.read(source)


def test_read_rejects_utf8_bom(tmp_path: Path) -> None:
    carrier = _carrier_class()()
    source = tmp_path / "carrier.json"
    payload = DurableReplayPublicationInvocationJsonCodec().encode(_transport())
    source.write_bytes(b"\xef\xbb\xbf" + payload.encode("utf-8"))

    with pytest.raises((TypeError, ValueError, UnicodeError)):
        carrier.read(source)


def test_read_rejects_invalid_utf8(tmp_path: Path) -> None:
    carrier = _carrier_class()()
    source = tmp_path / "carrier.json"
    source.write_bytes(b'{"x":"\xff"}')

    with pytest.raises(UnicodeDecodeError):
        carrier.read(source)


def test_read_rejects_missing_source(tmp_path: Path) -> None:
    carrier = _carrier_class()()
    with pytest.raises(FileNotFoundError):
        carrier.read(tmp_path / "missing.json")


def test_read_rejects_directory(tmp_path: Path) -> None:
    carrier = _carrier_class()()
    directory = tmp_path / "dir"
    directory.mkdir()

    with pytest.raises((IsADirectoryError, ValueError)):
        carrier.read(directory)


def test_path_text_is_not_normalized_or_expanded(monkeypatch, tmp_path: Path) -> None:
    carrier = _carrier_class()()
    transport = _transport()

    monkeypatch.chdir(tmp_path)
    relative = Path("carrier.json")
    result = carrier.write(relative, transport)

    assert result == relative
    assert result == Path("carrier.json")
    assert not result.is_absolute()


def test_carrier_declares_no_second_result_model() -> None:
    module = _module()
    classes = [
        value
        for value in vars(module).values()
        if inspect.isclass(value) and value.__module__ == MODULE_NAME
    ]
    assert [cls.__name__ for cls in classes] == [CLASS_NAME]


def test_carrier_has_no_json_domain_cli_execution_or_mutation_surface() -> None:
    source = Path(inspect.getsourcefile(_carrier_class()) or "").read_text(
        encoding="utf-8-sig"
    )

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
        "candidate_advantage_count",
        "baseline_advantage_count",
        "baseline_delta",
        "eligibility",
        "promotion_policy",
        "rollback",
        "resolve()",
        "expanduser",
        "getenv",
        "environ",
        "mkdir(",
        "replace(",
        "os.replace",
    )
    for token in forbidden:
        assert token not in source


def test_carrier_depends_on_exact_existing_owners() -> None:
    source = Path(inspect.getsourcefile(_carrier_class()) or "").read_text(
        encoding="utf-8-sig"
    )

    assert "pathlib" in source
    assert (
        "lrp.operations.durable_replay_publication_invocation_transport"
        in source
    )
    assert (
        "lrp.operations.durable_replay_publication_invocation_json_presentation"
        in source
    )
