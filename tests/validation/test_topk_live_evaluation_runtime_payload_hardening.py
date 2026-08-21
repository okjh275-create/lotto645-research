from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from lrp.contracts.exceptions import ContractError
from lrp.evaluation.topk_live_evaluation_runtime import (
    TopKLiveEvaluationRuntimeResult,
    TopKLiveEvaluationRuntimeService,
)
from lrp.evaluation.topk_live_evaluation_runtime_payload import (
    runtime_result_to_dict,
    runtime_result_to_json,
)


_RUNTIME_FIXTURE = Path(
    "tests/validation/"
    "test_topk_live_evaluation_runtime_contract.py"
)


def _foundation():
    spec = importlib.util.spec_from_file_location(
        "ag28_runtime_fixture",
        _RUNTIME_FIXTURE,
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

    return (
        TopKLiveEvaluationRuntimeService()
        .execute(
            request=fixture._request()
        )
    )


@pytest.mark.parametrize(
    "serializer",
    (
        runtime_result_to_dict,
        runtime_result_to_json,
    ),
)
def test_payload_rejects_malformed_source_pair(
    serializer,
) -> None:
    result = _runtime_result()

    malformed = TopKLiveEvaluationRuntimeResult(
        evaluation=result.evaluation,
        source_pair=object(),
    )

    with pytest.raises(ContractError):
        serializer(
            malformed
        )


@pytest.mark.parametrize(
    "serializer",
    (
        runtime_result_to_dict,
        runtime_result_to_json,
    ),
)
def test_payload_rejects_malformed_live_evaluation(
    serializer,
) -> None:
    result = _runtime_result()

    malformed = TopKLiveEvaluationRuntimeResult(
        evaluation=object(),
        source_pair=result.source_pair,
    )

    with pytest.raises(ContractError):
        serializer(
            malformed
        )


@pytest.mark.parametrize(
    "slot",
    (
        "candidate",
        "baseline",
    ),
)
def test_payload_rejects_malformed_snapshot_slot(
    slot: str,
) -> None:
    result = _runtime_result()

    if slot == "candidate":
        pair = SimpleNamespace(
            candidate=object(),
            baseline=result.source_pair.baseline,
        )
    else:
        pair = SimpleNamespace(
            candidate=result.source_pair.candidate,
            baseline=object(),
        )

    malformed = TopKLiveEvaluationRuntimeResult(
        evaluation=result.evaluation,
        source_pair=pair,
    )

    with pytest.raises(ContractError):
        runtime_result_to_dict(
            malformed
        )


@pytest.mark.parametrize(
    "serializer",
    (
        runtime_result_to_dict,
        runtime_result_to_json,
    ),
)
def test_payload_rejects_malformed_replay_evaluation(
    serializer,
) -> None:
    result = _runtime_result()

    live_evaluation = SimpleNamespace(
        evaluation=object(),
    )

    malformed = TopKLiveEvaluationRuntimeResult(
        evaluation=live_evaluation,
        source_pair=result.source_pair,
    )

    with pytest.raises(ContractError):
        serializer(
            malformed
        )


@pytest.mark.parametrize(
    "serializer",
    (
        runtime_result_to_dict,
        runtime_result_to_json,
    ),
)
def test_payload_rejects_malformed_walk_forward_projection(
    serializer,
) -> None:
    result = _runtime_result()

    replay = result.evaluation.evaluation

    malformed_replay = SimpleNamespace(
        candidate_model_name=(
            replay.candidate_model_name
        ),
        baseline_model_name=(
            replay.baseline_model_name
        ),
        round_count=(
            replay.round_count
        ),
        evaluation=object(),
    )

    live_evaluation = SimpleNamespace(
        evaluation=malformed_replay,
    )

    malformed = TopKLiveEvaluationRuntimeResult(
        evaluation=live_evaluation,
        source_pair=result.source_pair,
    )

    with pytest.raises(ContractError):
        serializer(
            malformed
        )


def test_payload_rejects_round_identity_tampering() -> None:
    result = _runtime_result()

    baseline = replace(
        result.source_pair.baseline,
        round_no=1234,
    )

    pair = SimpleNamespace(
        candidate=result.source_pair.candidate,
        baseline=baseline,
    )

    malformed = TopKLiveEvaluationRuntimeResult(
        evaluation=result.evaluation,
        source_pair=pair,
    )

    with pytest.raises(ContractError):
        runtime_result_to_dict(
            malformed
        )


def test_payload_rejects_candidate_model_identity_tampering() -> None:
    result = _runtime_result()

    candidate = replace(
        result.source_pair.candidate,
        model_name="tampered-candidate",
    )

    pair = SimpleNamespace(
        candidate=candidate,
        baseline=result.source_pair.baseline,
    )

    malformed = TopKLiveEvaluationRuntimeResult(
        evaluation=result.evaluation,
        source_pair=pair,
    )

    with pytest.raises(ContractError):
        runtime_result_to_dict(
            malformed
        )


def test_payload_rejects_baseline_model_identity_tampering() -> None:
    result = _runtime_result()

    baseline = replace(
        result.source_pair.baseline,
        model_name="tampered-baseline",
    )

    pair = SimpleNamespace(
        candidate=result.source_pair.candidate,
        baseline=baseline,
    )

    malformed = TopKLiveEvaluationRuntimeResult(
        evaluation=result.evaluation,
        source_pair=pair,
    )

    with pytest.raises(ContractError):
        runtime_result_to_dict(
            malformed
        )