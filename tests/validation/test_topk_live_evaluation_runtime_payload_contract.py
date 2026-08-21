from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
from pathlib import Path

import pytest

from lrp.contracts.exceptions import ContractError
from lrp.evaluation.topk_live_evaluation_source_snapshot import (
    snapshot_to_dict,
)


PRODUCT_MODULE = (
    "lrp.evaluation."
    "topk_live_evaluation_runtime_payload"
)

RUNTIME_FIXTURE = Path(
    "tests/validation/"
    "test_topk_live_evaluation_runtime_contract.py"
)


def _product():
    return importlib.import_module(
        PRODUCT_MODULE
    )


def _foundation():
    spec = importlib.util.spec_from_file_location(
        "ag25_runtime_fixture",
        RUNTIME_FIXTURE,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Unable to load canonical runtime fixture"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def _runtime_result():
    fixture = _foundation()

    runtime = importlib.import_module(
        "lrp.evaluation."
        "topk_live_evaluation_runtime"
    )

    return (
        runtime.TopKLiveEvaluationRuntimeService()
        .execute(
            request=fixture._request()
        )
    )


def _parameter_names(
    value: object,
) -> tuple[str, ...]:
    return tuple(
        inspect.signature(
            value
        ).parameters
    )


def test_runtime_result_to_dict_signature_is_exact() -> None:
    product = _product()

    assert _parameter_names(
        product.runtime_result_to_dict
    ) == (
        "result",
    )


def test_runtime_result_to_json_signature_is_exact() -> None:
    product = _product()

    assert _parameter_names(
        product.runtime_result_to_json
    ) == (
        "result",
    )


def test_runtime_payload_top_level_schema_is_exact() -> None:
    product = _product()

    payload = product.runtime_result_to_dict(
        _runtime_result()
    )

    assert tuple(payload) == (
        "schema_version",
        "round",
        "candidate",
        "baseline",
        "evaluation",
    )


def test_runtime_payload_schema_version_is_exact() -> None:
    product = _product()

    payload = product.runtime_result_to_dict(
        _runtime_result()
    )

    assert payload["schema_version"] == "1.0"


def test_runtime_payload_round_is_source_round() -> None:
    product = _product()

    result = _runtime_result()

    payload = product.runtime_result_to_dict(
        result
    )

    assert payload["round"] == 1233

    assert (
        payload["round"]
        == result.source_pair.candidate.round_no
    )

    assert (
        payload["round"]
        == result.source_pair.baseline.round_no
    )


def test_runtime_payload_candidate_is_canonical_snapshot_projection() -> None:
    product = _product()

    result = _runtime_result()

    payload = product.runtime_result_to_dict(
        result
    )

    assert payload["candidate"] == snapshot_to_dict(
        result.source_pair.candidate
    )


def test_runtime_payload_baseline_is_canonical_snapshot_projection() -> None:
    product = _product()

    result = _runtime_result()

    payload = product.runtime_result_to_dict(
        result
    )

    assert payload["baseline"] == snapshot_to_dict(
        result.source_pair.baseline
    )


def test_runtime_payload_evaluation_schema_is_exact() -> None:
    product = _product()

    payload = product.runtime_result_to_dict(
        _runtime_result()
    )

    assert tuple(
        payload["evaluation"]
    ) == (
        "candidate_model_name",
        "baseline_model_name",
        "round_count",
        "walk_forward",
    )


def test_runtime_payload_evaluation_identity_is_exact() -> None:
    product = _product()

    result = _runtime_result()

    payload = product.runtime_result_to_dict(
        result
    )

    replay = result.evaluation.evaluation

    assert (
        payload["evaluation"][
            "candidate_model_name"
        ]
        == replay.candidate_model_name
    )

    assert (
        payload["evaluation"][
            "baseline_model_name"
        ]
        == replay.baseline_model_name
    )

    assert (
        payload["evaluation"][
            "candidate_model_name"
        ]
        == result.source_pair.candidate.model_name
    )

    assert (
        payload["evaluation"][
            "baseline_model_name"
        ]
        == result.source_pair.baseline.model_name
    )


def test_runtime_payload_evaluation_round_count_is_exact() -> None:
    product = _product()

    result = _runtime_result()

    payload = product.runtime_result_to_dict(
        result
    )

    replay = result.evaluation.evaluation

    assert (
        payload["evaluation"]["round_count"]
        == replay.round_count
    )

    assert (
        payload["evaluation"]["round_count"]
        == 1
    )


def test_runtime_payload_walk_forward_uses_existing_projection() -> None:
    product = _product()

    result = _runtime_result()

    payload = product.runtime_result_to_dict(
        result
    )

    replay = result.evaluation.evaluation

    assert (
        payload["evaluation"]["walk_forward"]
        == replay.evaluation.as_dict()
    )


def test_runtime_payload_dict_is_json_safe() -> None:
    product = _product()

    payload = product.runtime_result_to_dict(
        _runtime_result()
    )

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
    )

    assert isinstance(
        encoded,
        str,
    )


def test_runtime_payload_json_round_trip_is_exact() -> None:
    product = _product()

    result = _runtime_result()

    payload = product.runtime_result_to_dict(
        result
    )

    encoded = product.runtime_result_to_json(
        result
    )

    assert json.loads(
        encoded
    ) == payload


def test_runtime_payload_dict_is_semantically_deterministic() -> None:
    product = _product()

    result = _runtime_result()

    first = product.runtime_result_to_dict(
        result
    )

    second = product.runtime_result_to_dict(
        result
    )

    assert first == second


def test_runtime_payload_json_is_exactly_deterministic() -> None:
    product = _product()

    result = _runtime_result()

    first = product.runtime_result_to_json(
        result
    )

    second = product.runtime_result_to_json(
        result
    )

    assert first == second


def test_runtime_payload_rejects_invalid_result_for_dict() -> None:
    product = _product()

    with pytest.raises(ContractError):
        product.runtime_result_to_dict(
            object()
        )


def test_runtime_payload_rejects_invalid_result_for_json() -> None:
    product = _product()

    with pytest.raises(ContractError):
        product.runtime_result_to_json(
            object()
        )


def test_runtime_payload_does_not_mutate_runtime_result() -> None:
    product = _product()

    result = _runtime_result()

    evaluation = result.evaluation
    source_pair = result.source_pair

    candidate = source_pair.candidate
    baseline = source_pair.baseline

    product.runtime_result_to_dict(
        result
    )

    product.runtime_result_to_json(
        result
    )

    assert result.evaluation is evaluation
    assert result.source_pair is source_pair

    assert result.source_pair.candidate is candidate
    assert result.source_pair.baseline is baseline


def test_runtime_payload_product_has_no_side_effect_dependency() -> None:
    path = Path(
        "lrp/evaluation/"
        "topk_live_evaluation_runtime_payload.py"
    )

    source = path.read_text(
        encoding="utf-8-sig"
    )

    forbidden = (
        "open(",
        "Path(",
        "write_text(",
        "write_bytes(",
        "mkdir(",
        "unlink(",
        "subprocess",
        "sqlite3",
        "write_operation_artifact",
        "write_prediction_artifacts",
        "atomic_write",
        "append_operation_log",
        "datetime.now",
        "datetime.utcnow",
        "time.time",
        "random",
        "secrets",
        "uuid",
    )

    assert not any(
        token in source
        for token in forbidden
    )


def test_runtime_payload_product_does_not_serialize_internal_runtime_objects() -> None:
    path = Path(
        "lrp/evaluation/"
        "topk_live_evaluation_runtime_payload.py"
    )

    source = path.read_text(
        encoding="utf-8-sig"
    )

    forbidden = (
        "dataclasses.asdict",
        "asdict(",
        "candidate_binding",
        "baseline_binding",
        "candidate_replay_prediction",
        "baseline_replay_prediction",
        "TopKLivePredictionBindingResult",
        "TopKReplayPrediction",
    )

    assert not any(
        token in source
        for token in forbidden
    )