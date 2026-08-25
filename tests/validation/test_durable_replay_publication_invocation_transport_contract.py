from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError, fields, is_dataclass
import importlib
import importlib.util
import inspect
from pathlib import Path
from types import MappingProxyType
from typing import get_type_hints

import pytest

from lrp.operations.durable_replay_promotion_publication_request import (
    DurableReplayPromotionPublicationRequest,
)

MODULE_NAME = "lrp.operations.durable_replay_publication_invocation_transport"
TRANSPORT_NAME = "DurableReplayPublicationInvocationTransport"
CODEC_NAME = "DurableReplayPublicationInvocationTransportCodec"

EXPECTED_FIELDS = (
    "status",
    "round_count",
    "candidate_model_name",
    "baseline_model_name",
    "recommendation",
    "action",
    "window",
    "source_decision",
    "registry_root",
)


def _module():
    return importlib.import_module(MODULE_NAME)


def _transport_type():
    return getattr(_module(), TRANSPORT_NAME)


def _codec_type():
    return getattr(_module(), CODEC_NAME)


def _request() -> DurableReplayPromotionPublicationRequest:
    return DurableReplayPromotionPublicationRequest(
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
                "flags": ["a", "b"],
                "nested": {"x": 1, "enabled": True},
            }
        ),
        source_decision=Path("explicit/source_decision.json"),
        registry_root=Path("explicit/registry_root"),
    )


def test_invocation_transport_product_module_exists() -> None:
    assert importlib.util.find_spec(MODULE_NAME) is not None


def test_transport_model_is_frozen_dataclass() -> None:
    transport_type = _transport_type()
    assert is_dataclass(transport_type)
    params = transport_type.__dataclass_params__
    assert params.frozen is True


def test_transport_fields_are_exact() -> None:
    transport_type = _transport_type()
    assert tuple(field.name for field in fields(transport_type)) == EXPECTED_FIELDS


def test_transport_annotations_are_transport_safe() -> None:
    transport_type = _transport_type()
    hints = get_type_hints(transport_type)
    assert hints["status"] is str
    assert hints["round_count"] is int
    assert hints["candidate_model_name"] is str
    assert hints["baseline_model_name"] is str
    assert hints["recommendation"] is str
    assert hints["action"] is str
    assert hints["source_decision"] is str
    assert hints["registry_root"] is str
    assert "Mapping" in str(hints["window"])


def test_codec_class_exists() -> None:
    codec_type = _codec_type()
    assert inspect.isclass(codec_type)


def test_codec_public_methods_are_exact() -> None:
    codec_type = _codec_type()
    public = {
        name
        for name, value in vars(codec_type).items()
        if callable(value) and not name.startswith("_")
    }
    assert public == {"encode", "to_mapping", "from_mapping", "decode"}


def test_encode_signature_is_exact() -> None:
    codec_type = _codec_type()
    sig = inspect.signature(codec_type.encode)
    assert tuple(sig.parameters) == ("self", "request")
    assert (
        sig.parameters["request"].annotation
        == "DurableReplayPromotionPublicationRequest"
        or "DurableReplayPromotionPublicationRequest"
        in str(sig.parameters["request"].annotation)
    )
    assert TRANSPORT_NAME in str(sig.return_annotation)


def test_to_mapping_signature_is_exact() -> None:
    codec_type = _codec_type()
    sig = inspect.signature(codec_type.to_mapping)
    assert tuple(sig.parameters) == ("self", "transport")
    assert TRANSPORT_NAME in str(sig.parameters["transport"].annotation)
    assert "Mapping" in str(sig.return_annotation)


def test_from_mapping_signature_is_exact() -> None:
    codec_type = _codec_type()
    sig = inspect.signature(codec_type.from_mapping)
    assert tuple(sig.parameters) == ("self", "mapping")
    assert "Mapping" in str(sig.parameters["mapping"].annotation)
    assert TRANSPORT_NAME in str(sig.return_annotation)


def test_decode_signature_is_exact() -> None:
    codec_type = _codec_type()
    sig = inspect.signature(codec_type.decode)
    assert tuple(sig.parameters) == ("self", "transport")
    assert TRANSPORT_NAME in str(sig.parameters["transport"].annotation)
    assert "DurableReplayPromotionPublicationRequest" in str(sig.return_annotation)


def test_encode_projects_all_request_fields_exactly() -> None:
    codec = _codec_type()()
    request = _request()
    transport = codec.encode(request)

    assert transport.status == request.status
    assert transport.round_count == request.round_count
    assert transport.candidate_model_name == request.candidate_model_name
    assert transport.baseline_model_name == request.baseline_model_name
    assert transport.recommendation == request.recommendation
    assert transport.action == request.action
    assert dict(transport.window) == dict(request.window)
    assert transport.source_decision == str(request.source_decision)
    assert transport.registry_root == str(request.registry_root)


def test_transport_window_is_read_only_and_detached() -> None:
    codec = _codec_type()()
    request = _request()
    transport = codec.encode(request)

    assert isinstance(transport.window, Mapping)
    assert transport.window is not request.window

    with pytest.raises(TypeError):
        transport.window["x"] = 1


def test_transport_model_is_immutable() -> None:
    codec = _codec_type()()
    transport = codec.encode(_request())

    with pytest.raises(FrozenInstanceError):
        transport.status = "FAIL"


def test_to_mapping_returns_exact_detached_plain_mapping() -> None:
    codec = _codec_type()()
    transport = codec.encode(_request())
    payload = codec.to_mapping(transport)

    assert isinstance(payload, Mapping)
    assert tuple(payload.keys()) == EXPECTED_FIELDS
    assert payload["status"] == transport.status
    assert payload["round_count"] == transport.round_count
    assert payload["candidate_model_name"] == transport.candidate_model_name
    assert payload["baseline_model_name"] == transport.baseline_model_name
    assert payload["recommendation"] == transport.recommendation
    assert payload["action"] == transport.action
    assert payload["window"] == dict(transport.window)
    assert payload["source_decision"] == transport.source_decision
    assert payload["registry_root"] == transport.registry_root
    assert payload["window"] is not transport.window


def test_from_mapping_accepts_exact_field_set() -> None:
    codec = _codec_type()()
    transport = codec.encode(_request())
    payload = codec.to_mapping(transport)
    rebuilt = codec.from_mapping(payload)

    assert rebuilt == transport
    assert rebuilt.window is not transport.window
    assert dict(rebuilt.window) == dict(transport.window)


@pytest.mark.parametrize(
    "missing_field",
    EXPECTED_FIELDS,
)
def test_from_mapping_rejects_missing_fields(missing_field: str) -> None:
    codec = _codec_type()()
    payload = dict(codec.to_mapping(codec.encode(_request())))
    payload.pop(missing_field)

    with pytest.raises((TypeError, ValueError, KeyError)):
        codec.from_mapping(payload)


def test_from_mapping_rejects_unknown_fields() -> None:
    codec = _codec_type()()
    payload = dict(codec.to_mapping(codec.encode(_request())))
    payload["unexpected"] = "value"

    with pytest.raises((TypeError, ValueError, KeyError)):
        codec.from_mapping(payload)


def test_decode_restores_ba_equivalent_request() -> None:
    codec = _codec_type()()
    original = _request()
    decoded = codec.decode(codec.encode(original))

    assert isinstance(decoded, DurableReplayPromotionPublicationRequest)
    assert decoded.status == original.status
    assert decoded.round_count == original.round_count
    assert decoded.candidate_model_name == original.candidate_model_name
    assert decoded.baseline_model_name == original.baseline_model_name
    assert decoded.recommendation == original.recommendation
    assert decoded.action == original.action
    assert dict(decoded.window) == dict(original.window)
    assert str(decoded.source_decision) == str(original.source_decision)
    assert str(decoded.registry_root) == str(original.registry_root)


def test_decode_window_is_read_only_and_detached() -> None:
    codec = _codec_type()()
    original = _request()
    decoded = codec.decode(codec.encode(original))

    assert decoded.window is not original.window
    with pytest.raises(TypeError):
        decoded.window["x"] = 1


def test_codec_round_trip_is_deterministic() -> None:
    codec = _codec_type()()
    request = _request()

    a = codec.to_mapping(codec.encode(request))
    b = codec.to_mapping(codec.encode(request))

    assert a == b
    assert codec.from_mapping(a) == codec.from_mapping(b)
    assert codec.decode(codec.from_mapping(a)) == codec.decode(
        codec.from_mapping(b)
    )


def test_codec_does_not_mutate_input_request_or_window() -> None:
    codec = _codec_type()()
    request = _request()
    before = dict(request.window)

    codec.encode(request)

    assert dict(request.window) == before


def test_codec_declares_no_filesystem_cli_or_execution_surface() -> None:
    source = inspect.getsource(_module()).lower()

    forbidden = (
        "open(",
        "read_text",
        "write_text",
        "read_bytes",
        "write_bytes",
        "json.dump",
        "json.load",
        "argparse",
        "stdin",
        "stdout",
        "lrp.cli",
        "durablereplaypublicationlifecycleentrypoint",
        ".run(",
        "productionchampionregistrypublisher",
        ".publish(",
        "run_publication_stage",
    )

    for token in forbidden:
        assert token not in source


def test_codec_declares_no_identity_discovery_or_policy_surface() -> None:
    source = inspect.getsource(_module()).lower()

    forbidden = (
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
    )

    for token in forbidden:
        assert token not in source


def test_transport_depends_only_on_ba_request_owner() -> None:
    source = inspect.getsource(_module())

    assert (
        "lrp.operations.durable_replay_promotion_publication_request"
        in source
    )
    assert "lrp.production" not in source
    assert "lrp.cli" not in source