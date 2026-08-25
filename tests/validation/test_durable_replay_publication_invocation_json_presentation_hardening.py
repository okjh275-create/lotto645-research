from __future__ import annotations

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
from lrp.operations.durable_replay_publication_invocation_json_presentation import (
    DurableReplayPublicationInvocationJsonCodec,
)


def _valid_transport(
    *,
    candidate_model_name: str = "candidate-model",
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
                    "nested": {
                        "labels": ["가", "나", "α", "β", "✓"],
                        "ratio": 1.25,
                        "enabled": True,
                        "none": None,
                    },
                }
            )
            if window is None
            else window
        ),
        source_decision=Path("explicit/source_decision.json"),
        registry_root=Path("explicit/registry_root"),
    )
    return DurableReplayPublicationInvocationTransportCodec().encode(request)


def _mapping(
    transport: DurableReplayPublicationInvocationTransport | None = None,
) -> dict[str, object]:
    if transport is None:
        transport = _valid_transport()
    return DurableReplayPublicationInvocationTransportCodec().to_mapping(
        transport
    )


def _json_payload(mapping: dict[str, object]) -> str:
    return json.dumps(
        mapping,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def test_valid_round_trip_preserves_all_transport_fields() -> None:
    codec = DurableReplayPublicationInvocationJsonCodec()
    transport = _valid_transport(candidate_model_name="후보-model")

    decoded = codec.decode(codec.encode(transport))

    assert decoded == transport


@pytest.mark.parametrize(
    "payload",
    [
        "",
        " ",
        "\n",
        "{",
        "}",
        "[",
        "]",
        '{"status":',
        '{"status":"PASS",}',
        '{"status":"PASS" "round_count":9}',
        '{"window":{"a":1,}}',
        '{"window":[1,2}',
        '{"x":"\\uZZZZ"}',
    ],
)
def test_malformed_json_fails_closed(payload: str) -> None:
    with pytest.raises(ValueError):
        DurableReplayPublicationInvocationJsonCodec().decode(payload)


@pytest.mark.parametrize(
    "payload",
    [
        "[]",
        "[1,2,3]",
        '"text"',
        "1",
        "-1",
        "1.5",
        "true",
        "false",
        "null",
    ],
)
def test_non_object_root_fails_closed(payload: str) -> None:
    with pytest.raises(TypeError):
        DurableReplayPublicationInvocationJsonCodec().decode(payload)


@pytest.mark.parametrize(
    "payload",
    [
        '{"status":"PASS","status":"FAIL"}',
        '{"window":{"x":1,"x":2}}',
        '{"window":{"nested":{"x":1,"x":2}}}',
        '{"window":{"items":[{"x":1,"x":2}]}}',
        '{"source_decision":"a","source_decision":"b"}',
        '{"registry_root":"a","registry_root":"b"}',
    ],
)
def test_duplicate_keys_fail_closed_at_any_depth(payload: str) -> None:
    with pytest.raises(ValueError):
        DurableReplayPublicationInvocationJsonCodec().decode(payload)


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_json_constants_fail_closed_on_decode(token: str) -> None:
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
        DurableReplayPublicationInvocationJsonCodec().decode(payload)


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_non_finite_values_fail_closed_on_encode(value: float) -> None:
    valid = _valid_transport()
    malformed = DurableReplayPublicationInvocationTransport(
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
        DurableReplayPublicationInvocationJsonCodec().encode(malformed)


def test_unicode_round_trip_is_lossless() -> None:
    codec = DurableReplayPublicationInvocationJsonCodec()
    transport = _valid_transport(candidate_model_name="후보-모델-서울-✓")

    payload = codec.encode(transport)
    decoded = codec.decode(payload)

    assert "후보-모델-서울-✓" in payload
    assert "가" in payload
    assert "α" in payload
    assert decoded == transport


def test_canonical_json_is_independent_of_mapping_order() -> None:
    codec = DurableReplayPublicationInvocationJsonCodec()

    a = _valid_transport(
        window={
            "z": 3,
            "a": 1,
            "nested": {"b": 2, "a": 1},
        }
    )
    b = _valid_transport(
        window={
            "nested": {"a": 1, "b": 2},
            "a": 1,
            "z": 3,
        }
    )

    assert codec.encode(a) == codec.encode(b)


def test_canonical_json_is_repeatable() -> None:
    codec = DurableReplayPublicationInvocationJsonCodec()
    transport = _valid_transport()

    payloads = [codec.encode(transport) for _ in range(5)]

    assert len(set(payloads)) == 1


@pytest.mark.parametrize(
    ("field", "bad_value", "exc_type"),
    [
        ("status", None, TypeError),
        ("round_count", "9", TypeError),
        ("round_count", True, TypeError),
        ("candidate_model_name", 1, TypeError),
        ("baseline_model_name", 1, TypeError),
        ("recommendation", 1, TypeError),
        ("action", 1, TypeError),
        ("source_decision", None, TypeError),
        ("registry_root", None, TypeError),
    ],
)
def test_scalar_validation_is_delegated_to_be(
    field: str,
    bad_value: object,
    exc_type: type[Exception],
) -> None:
    mapping = _mapping()
    mapping[field] = bad_value

    with pytest.raises(exc_type):
        DurableReplayPublicationInvocationJsonCodec().decode(
            _json_payload(mapping)
        )


@pytest.mark.parametrize(
    "field",
    [
        "status",
        "round_count",
        "candidate_model_name",
        "baseline_model_name",
        "recommendation",
        "action",
        "window",
        "source_decision",
        "registry_root",
    ],
)
def test_missing_field_validation_is_delegated_to_be(field: str) -> None:
    mapping = _mapping()
    mapping.pop(field)

    with pytest.raises(ValueError):
        DurableReplayPublicationInvocationJsonCodec().decode(
            _json_payload(mapping)
        )


@pytest.mark.parametrize(
    "extra",
    ["unexpected", "extra", "sourceDecision"],
)
def test_unknown_field_validation_is_delegated_to_be(extra: str) -> None:
    mapping = _mapping()
    mapping[extra] = "x"

    with pytest.raises(ValueError):
        DurableReplayPublicationInvocationJsonCodec().decode(
            _json_payload(mapping)
        )


@pytest.mark.parametrize(
    "bad_window",
    [
        None,
        [],
        "window",
        123,
        True,
    ],
)
def test_window_type_validation_is_delegated_to_be(
    bad_window: object,
) -> None:
    mapping = _mapping()
    mapping["window"] = bad_window

    with pytest.raises(TypeError):
        DurableReplayPublicationInvocationJsonCodec().decode(
            _json_payload(mapping)
        )


def test_json_codec_does_not_mutate_transport() -> None:
    codec = DurableReplayPublicationInvocationJsonCodec()
    transport = _valid_transport()
    before = _mapping(transport)

    codec.encode(transport)

    after = _mapping(transport)
    assert after == before


def test_json_codec_has_exact_dependency_boundary() -> None:
    import inspect
    import lrp.operations.durable_replay_publication_invocation_json_presentation as module

    source = inspect.getsource(module)

    assert "import json" in source
    assert (
        "lrp.operations.durable_replay_publication_invocation_transport"
        in source
    )
    assert "pathlib" not in source
    assert "argparse" not in source
    assert "lrp.cli" not in source


def test_json_codec_has_no_io_execution_discovery_policy_or_rollback_surface() -> None:
    import inspect
    import lrp.operations.durable_replay_publication_invocation_json_presentation as module

    source = inspect.getsource(module).lower()

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
