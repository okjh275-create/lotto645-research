from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
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


MODULE_NAME = (
    "lrp.operations."
    "durable_replay_publication_invocation_json_presentation"
)
CODEC_NAME = "DurableReplayPublicationInvocationJsonCodec"


def _module():
    return importlib.import_module(MODULE_NAME)


def _codec_class():
    return getattr(_module(), CODEC_NAME)


def _codec():
    return _codec_class()()


def _transport(
    *,
    candidate_model_name: str = "후보-model",
    window: object | None = None,
) -> DurableReplayPublicationInvocationTransport:
    request = DurableReplayPromotionPublicationRequest(
        status="PASS",
        round_count=9,
        candidate_model_name=candidate_model_name,
        baseline_model_name="baseline-model",
        recommendation="eligible",
        action="prepare_publish",
        window=(
            MappingProxyType(
                {
                    "start_round": 1223,
                    "end_round": 1231,
                    "mode": "durable-replay",
                    "nested": {"z": 2, "a": 1},
                    "labels": ["가", "나", "α", "β"],
                }
            )
            if window is None
            else window
        ),
        source_decision=Path("explicit/source_decision.json"),
        registry_root=Path("explicit/registry_root"),
    )
    return DurableReplayPublicationInvocationTransportCodec().encode(request)


def test_json_presentation_product_module_exists() -> None:
    assert importlib.util.find_spec(MODULE_NAME) is not None


def test_json_codec_class_exists() -> None:
    assert hasattr(_module(), CODEC_NAME)


def test_json_codec_public_methods_are_exact() -> None:
    cls = _codec_class()
    public = {
        name
        for name, value in vars(cls).items()
        if not name.startswith("_") and callable(value)
    }
    assert public == {"encode", "decode"}


def test_encode_signature_is_exact() -> None:
    sig = inspect.signature(_codec_class().encode)
    assert tuple(sig.parameters) == ("self", "transport")
    assert (
        sig.parameters["transport"].annotation
        in {
            "DurableReplayPublicationInvocationTransport",
            DurableReplayPublicationInvocationTransport,
        }
    )
    assert sig.return_annotation in {"str", str}


def test_decode_signature_is_exact() -> None:
    sig = inspect.signature(_codec_class().decode)
    assert tuple(sig.parameters) == ("self", "payload")
    assert sig.parameters["payload"].annotation in {"str", str}
    assert (
        sig.return_annotation
        in {
            "DurableReplayPublicationInvocationTransport",
            DurableReplayPublicationInvocationTransport,
        }
    )


def test_encode_returns_str() -> None:
    assert isinstance(_codec().encode(_transport()), str)


def test_encode_is_deterministic_for_same_transport() -> None:
    codec = _codec()
    transport = _transport()
    assert codec.encode(transport) == codec.encode(transport)


def test_encode_is_canonical_across_mapping_key_order() -> None:
    codec = _codec()
    a = _transport(
        window={
            "mode": "durable-replay",
            "end_round": 1231,
            "start_round": 1223,
            "nested": {"z": 2, "a": 1},
        }
    )
    b = _transport(
        window={
            "nested": {"a": 1, "z": 2},
            "start_round": 1223,
            "end_round": 1231,
            "mode": "durable-replay",
        }
    )
    assert codec.encode(a) == codec.encode(b)


def test_encode_uses_compact_canonical_json() -> None:
    payload = _codec().encode(_transport(candidate_model_name="candidate-model"))
    parsed = json.loads(payload)
    assert payload == json.dumps(
        parsed,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def test_unicode_is_preserved_as_literal_json_text() -> None:
    payload = _codec().encode(_transport())
    assert "후보-model" in payload
    assert "가" in payload
    assert "α" in payload


def test_encode_projects_exact_be_mapping() -> None:
    transport = _transport(candidate_model_name="candidate-model")
    expected = DurableReplayPublicationInvocationTransportCodec().to_mapping(
        transport
    )
    payload = _codec().encode(transport)
    assert json.loads(payload) == expected


def test_decode_round_trip_restores_equal_transport() -> None:
    codec = _codec()
    transport = _transport()
    assert codec.decode(codec.encode(transport)) == transport


@pytest.mark.parametrize("bad_payload", [None, 1, b"{}", {}, []])
def test_decode_rejects_non_string_payload(bad_payload: object) -> None:
    with pytest.raises(TypeError):
        _codec().decode(bad_payload)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "payload",
    [
        "",
        "{",
        '{"status":',
        '{"status":"PASS",}',
        '{"window":[1,2}',
    ],
)
def test_decode_rejects_malformed_json(payload: str) -> None:
    with pytest.raises(ValueError):
        _codec().decode(payload)


@pytest.mark.parametrize(
    "payload",
    [
        "[]",
        '"payload"',
        "1",
        "true",
        "null",
    ],
)
def test_decode_rejects_non_object_root(payload: str) -> None:
    with pytest.raises(TypeError):
        _codec().decode(payload)


@pytest.mark.parametrize(
    "payload",
    [
        '{"status":"PASS","status":"FAIL"}',
        '{"window":{"start_round":1223,"start_round":1224}}',
    ],
)
def test_decode_rejects_duplicate_object_keys(payload: str) -> None:
    with pytest.raises(ValueError):
        _codec().decode(payload)


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf")],
)
def test_encode_rejects_non_finite_numbers(value: float) -> None:
    valid = _transport()
    transport = DurableReplayPublicationInvocationTransport(
        status=valid.status,
        round_count=valid.round_count,
        candidate_model_name=valid.candidate_model_name,
        baseline_model_name=valid.baseline_model_name,
        recommendation=valid.recommendation,
        action=valid.action,
        window=MappingProxyType({"value": value}),
        source_decision=valid.source_decision,
        registry_root=valid.registry_root,
    )
    with pytest.raises((TypeError, ValueError)):
        _codec().encode(transport)


@pytest.mark.parametrize(
    "token",
    ["NaN", "Infinity", "-Infinity"],
)
def test_decode_rejects_non_finite_constants(token: str) -> None:
    payload = (
        '{"action":"prepare_publish",'
        '"baseline_model_name":"baseline-model",'
        '"candidate_model_name":"candidate-model",'
        '"recommendation":"eligible",'
        '"registry_root":"explicit\\\\registry_root",'
        '"round_count":9,'
        '"source_decision":"explicit\\\\source_decision.json",'
        '"status":"PASS",'
        '"window":{"value":' + token + "}}"
    )
    with pytest.raises(ValueError):
        _codec().decode(payload)


def test_decode_delegates_missing_field_validation_to_be() -> None:
    mapping = DurableReplayPublicationInvocationTransportCodec().to_mapping(
        _transport()
    )
    mapping.pop("status")
    payload = json.dumps(mapping, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with pytest.raises(ValueError):
        _codec().decode(payload)


def test_decode_delegates_unknown_field_validation_to_be() -> None:
    mapping = DurableReplayPublicationInvocationTransportCodec().to_mapping(
        _transport()
    )
    mapping["unexpected"] = "x"
    payload = json.dumps(mapping, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with pytest.raises(ValueError):
        _codec().decode(payload)


def test_decode_delegates_scalar_type_validation_to_be() -> None:
    mapping = DurableReplayPublicationInvocationTransportCodec().to_mapping(
        _transport()
    )
    mapping["round_count"] = "9"
    payload = json.dumps(mapping, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with pytest.raises(TypeError):
        _codec().decode(payload)


def test_decode_delegates_window_validation_to_be() -> None:
    mapping = DurableReplayPublicationInvocationTransportCodec().to_mapping(
        _transport()
    )
    mapping["window"] = []
    payload = json.dumps(mapping, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with pytest.raises(TypeError):
        _codec().decode(payload)


def test_product_depends_on_be_transport_owner() -> None:
    source = inspect.getsource(_module())
    assert "durable_replay_publication_invocation_transport" in source
    assert "DurableReplayPublicationInvocationTransportCodec" in source


def test_product_has_no_filesystem_cli_execution_or_publisher_surface() -> None:
    source = inspect.getsource(_module()).lower()
    forbidden = (
        "open(",
        "read_text",
        "write_text",
        "read_bytes",
        "write_bytes",
        "stdin",
        "stdout",
        "argparse",
        "lrp.cli",
        "durablereplaypublicationlifecycleentrypoint",
        ".run(",
        "productionchampionregistrypublisher",
        ".publish(",
        "run_publication_stage",
    )
    for token in forbidden:
        assert token not in source


def test_product_has_no_identity_discovery_policy_or_rollback_surface() -> None:
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
