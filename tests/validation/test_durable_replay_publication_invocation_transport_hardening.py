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


_UNSET = object()


def _request(
    *,
    window: object = _UNSET,
    source_decision: object = "explicit/source_decision.json",
    registry_root: object = "explicit/registry_root",
) -> DurableReplayPromotionPublicationRequest:
    if window is _UNSET:
        window = MappingProxyType(
            {
                "start_round": 1223,
                "end_round": 1231,
                "mode": "durable-replay",
                "flags": ["a", "b"],
                "nested": {"x": 1, "enabled": True},
            }
        )

    return DurableReplayPromotionPublicationRequest(
        status="PASS",
        round_count=9,
        candidate_model_name="candidate-model",
        baseline_model_name="baseline-model",
        recommendation="eligible",
        action="prepare_publish",
        window=window,
        source_decision=source_decision,
        registry_root=registry_root,
    )


def test_valid_transport_round_trip_with_path_inputs() -> None:
    codec = DurableReplayPublicationInvocationTransportCodec()
    request = _request(
        source_decision=Path("explicit/source_decision.json"),
        registry_root=Path("explicit/registry_root"),
    )

    transport = codec.encode(request)
    decoded = codec.decode(codec.from_mapping(codec.to_mapping(transport)))

    assert isinstance(transport, DurableReplayPublicationInvocationTransport)
    assert transport.source_decision == str(Path("explicit/source_decision.json"))
    assert transport.registry_root == str(Path("explicit/registry_root"))
    assert str(decoded.source_decision) == str(Path("explicit/source_decision.json"))
    assert str(decoded.registry_root) == str(Path("explicit/registry_root"))


@pytest.mark.parametrize("bad_request", [None, 1, "request", object()])
def test_encode_rejects_invalid_request_type(bad_request: object) -> None:
    codec = DurableReplayPublicationInvocationTransportCodec()

    with pytest.raises(TypeError):
        codec.encode(bad_request)


@pytest.mark.parametrize(
    "bad_window",
    [None, [], (), "window", 123],
)
def test_encode_rejects_non_mapping_window(bad_window: object) -> None:
    codec = DurableReplayPublicationInvocationTransportCodec()
    request = _request(window=bad_window)

    with pytest.raises(TypeError):
        codec.encode(request)


@pytest.mark.parametrize(
    "bad_nested",
    [
        {"bad": object()},
        {"bad": {1, 2}},
        {"bad": (1, 2)},
        {"bad": complex(1, 2)},
        {"bad": float("nan")},
        {"bad": float("inf")},
        {"bad": float("-inf")},
    ],
)
def test_encode_rejects_non_json_compatible_window_values(
    bad_nested: dict[str, object],
) -> None:
    codec = DurableReplayPublicationInvocationTransportCodec()
    request = _request(window=bad_nested)

    with pytest.raises(TypeError):
        codec.encode(request)


def test_encode_rejects_non_string_nested_mapping_keys() -> None:
    codec = DurableReplayPublicationInvocationTransportCodec()
    request = _request(window={1: "value"})

    with pytest.raises(TypeError):
        codec.encode(request)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("status", None),
        ("round_count", "9"),
        ("round_count", True),
        ("candidate_model_name", 1),
        ("baseline_model_name", 1),
        ("recommendation", 1),
        ("action", 1),
        ("source_decision", None),
        ("registry_root", None),
    ],
)
def test_from_mapping_rejects_invalid_scalar_types(
    field_name: str,
    bad_value: object,
) -> None:
    codec = DurableReplayPublicationInvocationTransportCodec()
    payload = dict(codec.to_mapping(codec.encode(_request())))
    payload[field_name] = bad_value

    with pytest.raises(TypeError):
        codec.from_mapping(payload)


@pytest.mark.parametrize(
    "bad_window",
    [None, [], (), "window", 123],
)
def test_from_mapping_rejects_non_mapping_window(
    bad_window: object,
) -> None:
    codec = DurableReplayPublicationInvocationTransportCodec()
    payload = dict(codec.to_mapping(codec.encode(_request())))
    payload["window"] = bad_window

    with pytest.raises(TypeError):
        codec.from_mapping(payload)


@pytest.mark.parametrize(
    "bad_nested",
    [
        {"bad": object()},
        {"bad": {1, 2}},
        {"bad": (1, 2)},
        {"bad": complex(1, 2)},
        {"bad": float("nan")},
        {"bad": float("inf")},
        {"bad": float("-inf")},
    ],
)
def test_from_mapping_rejects_non_json_compatible_nested_values(
    bad_nested: dict[str, object],
) -> None:
    codec = DurableReplayPublicationInvocationTransportCodec()
    payload = dict(codec.to_mapping(codec.encode(_request())))
    payload["window"] = bad_nested

    with pytest.raises(TypeError):
        codec.from_mapping(payload)


def test_to_mapping_rejects_invalid_transport_type() -> None:
    codec = DurableReplayPublicationInvocationTransportCodec()

    with pytest.raises(TypeError):
        codec.to_mapping(object())


def test_decode_rejects_invalid_transport_type() -> None:
    codec = DurableReplayPublicationInvocationTransportCodec()

    with pytest.raises(TypeError):
        codec.decode(object())


def test_mapping_round_trip_deep_detaches_nested_values() -> None:
    codec = DurableReplayPublicationInvocationTransportCodec()
    transport = codec.encode(_request())
    payload = codec.to_mapping(transport)

    payload_window = payload["window"]
    assert isinstance(payload_window, dict)
    nested = payload_window["nested"]
    flags = payload_window["flags"]
    assert isinstance(nested, dict)
    assert isinstance(flags, list)

    nested["x"] = 999
    flags.append("c")

    assert transport.window["nested"]["x"] == 1
    assert transport.window["flags"] == ["a", "b"]


def test_from_mapping_deep_detaches_source_mapping() -> None:
    codec = DurableReplayPublicationInvocationTransportCodec()
    payload = dict(codec.to_mapping(codec.encode(_request())))
    window = payload["window"]
    assert isinstance(window, dict)

    rebuilt = codec.from_mapping(payload)

    nested = window["nested"]
    flags = window["flags"]
    assert isinstance(nested, dict)
    assert isinstance(flags, list)

    nested["x"] = 999
    flags.append("c")

    assert rebuilt.window["nested"]["x"] == 1
    assert rebuilt.window["flags"] == ["a", "b"]


def test_decode_deep_detaches_transport_window() -> None:
    codec = DurableReplayPublicationInvocationTransportCodec()
    transport = codec.encode(_request())
    decoded = codec.decode(transport)

    decoded_nested = decoded.window["nested"]
    decoded_flags = decoded.window["flags"]

    assert decoded_nested == {"x": 1, "enabled": True}
    assert decoded_flags == ["a", "b"]
    assert decoded.window is not transport.window


def test_path_text_is_preserved_without_normalization() -> None:
    codec = DurableReplayPublicationInvocationTransportCodec()
    request = _request(
        source_decision=r".\relative\..\decision.json",
        registry_root=r".\registry\..\registry-root",
    )

    transport = codec.encode(request)

    assert transport.source_decision == r".\relative\..\decision.json"
    assert transport.registry_root == r".\registry\..\registry-root"


def test_codec_is_deterministic_for_same_explicit_input() -> None:
    codec = DurableReplayPublicationInvocationTransportCodec()
    request = _request()

    a = codec.to_mapping(codec.encode(request))
    b = codec.to_mapping(codec.encode(request))

    assert a == b


def test_codec_does_not_mutate_request_or_nested_window() -> None:
    codec = DurableReplayPublicationInvocationTransportCodec()
    nested = {"x": 1}
    flags = ["a", "b"]
    source_window = {
        "nested": nested,
        "flags": flags,
    }
    request = _request(window=source_window)

    codec.encode(request)

    assert source_window == {
        "nested": {"x": 1},
        "flags": ["a", "b"],
    }
    assert nested == {"x": 1}
    assert flags == ["a", "b"]


def test_product_has_exact_dependency_boundary() -> None:
    import inspect
    import lrp.operations.durable_replay_publication_invocation_transport as module

    source = inspect.getsource(module)

    assert (
        "lrp.operations.durable_replay_promotion_publication_request"
        in source
    )
    assert "lrp.production" not in source
    assert "lrp.cli" not in source


def test_product_has_no_io_execution_discovery_or_policy_surface() -> None:
    import inspect
    import lrp.operations.durable_replay_publication_invocation_transport as module

    source = inspect.getsource(module).lower()

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
        "durablereplaypublicationlifecycleentrypoint",
        ".run(",
        "productionchampionregistrypublisher",
        ".publish(",
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
    )

    for token in forbidden:
        assert token not in source