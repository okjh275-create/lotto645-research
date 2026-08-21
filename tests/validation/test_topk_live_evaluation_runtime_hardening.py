from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from lrp.contracts.exceptions import ContractError
from lrp.evaluation import EvaluationWindow
from lrp.evaluation.topk_live_evaluation_runtime import (
    TopKLiveEvaluationRuntimeRequest,
    TopKLiveEvaluationRuntimeService,
)


_FOUNDATION_PATH = Path(
    "tests/validation/"
    "test_topk_live_evaluation_runtime_contract.py"
)


def _foundation():
    spec = importlib.util.spec_from_file_location(
        "ag22_runtime_foundation",
        _FOUNDATION_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Unable to load runtime foundation fixture"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def _clone_request(
    base,
    *,
    window=None,
    candidate_prediction_result=None,
    candidate_history_rows=None,
    candidate_model_name=None,
    candidate_source_artifact_sha256=None,
    baseline_prediction_result=None,
    baseline_history_rows=None,
    baseline_model_name=None,
    baseline_source_artifact_sha256=None,
    actual_draws=None,
    candidate_regime_id="__KEEP__",
    candidate_strategy_name="__KEEP__",
    baseline_regime_id="__KEEP__",
    baseline_strategy_name="__KEEP__",
):
    return TopKLiveEvaluationRuntimeRequest(
        window=(
            base.window
            if window is None
            else window
        ),
        candidate_prediction_result=(
            base.candidate_prediction_result
            if candidate_prediction_result is None
            else candidate_prediction_result
        ),
        candidate_history_rows=(
            base.candidate_history_rows
            if candidate_history_rows is None
            else candidate_history_rows
        ),
        candidate_model_name=(
            base.candidate_model_name
            if candidate_model_name is None
            else candidate_model_name
        ),
        candidate_source_artifact_sha256=(
            base.candidate_source_artifact_sha256
            if candidate_source_artifact_sha256 is None
            else candidate_source_artifact_sha256
        ),
        baseline_prediction_result=(
            base.baseline_prediction_result
            if baseline_prediction_result is None
            else baseline_prediction_result
        ),
        baseline_history_rows=(
            base.baseline_history_rows
            if baseline_history_rows is None
            else baseline_history_rows
        ),
        baseline_model_name=(
            base.baseline_model_name
            if baseline_model_name is None
            else baseline_model_name
        ),
        baseline_source_artifact_sha256=(
            base.baseline_source_artifact_sha256
            if baseline_source_artifact_sha256 is None
            else baseline_source_artifact_sha256
        ),
        actual_draws=(
            base.actual_draws
            if actual_draws is None
            else actual_draws
        ),
        candidate_regime_id=(
            base.candidate_regime_id
            if candidate_regime_id == "__KEEP__"
            else candidate_regime_id
        ),
        candidate_strategy_name=(
            base.candidate_strategy_name
            if candidate_strategy_name == "__KEEP__"
            else candidate_strategy_name
        ),
        baseline_regime_id=(
            base.baseline_regime_id
            if baseline_regime_id == "__KEEP__"
            else baseline_regime_id
        ),
        baseline_strategy_name=(
            base.baseline_strategy_name
            if baseline_strategy_name == "__KEEP__"
            else baseline_strategy_name
        ),
    )


def test_runtime_rejects_baseline_round_mismatch() -> None:
    fixture = _foundation()
    base = fixture._request()

    baseline = fixture._prediction_result(
        round_no=1234,
    )

    request = _clone_request(
        base,
        baseline_prediction_result=baseline,
    )

    with pytest.raises(ContractError):
        TopKLiveEvaluationRuntimeService().execute(
            request=request
        )


@pytest.mark.parametrize(
    "side",
    (
        "candidate",
        "baseline",
    ),
)
def test_runtime_rejects_reverse_history(
    side: str,
) -> None:
    fixture = _foundation()
    base = fixture._request()

    candidate_rows = (
        tuple(
            reversed(
                base.candidate_history_rows
            )
        )
        if side == "candidate"
        else base.candidate_history_rows
    )

    baseline_rows = (
        tuple(
            reversed(
                base.baseline_history_rows
            )
        )
        if side == "baseline"
        else base.baseline_history_rows
    )

    request = _clone_request(
        base,
        candidate_history_rows=candidate_rows,
        baseline_history_rows=baseline_rows,
    )

    with pytest.raises(ContractError):
        TopKLiveEvaluationRuntimeService().execute(
            request=request
        )


@pytest.mark.parametrize(
    "side",
    (
        "candidate",
        "baseline",
    ),
)
def test_runtime_rejects_duplicate_history_rounds(
    side: str,
) -> None:
    fixture = _foundation()
    base = fixture._request()

    candidate_rows = base.candidate_history_rows
    baseline_rows = base.baseline_history_rows

    if side == "candidate":
        candidate_rows = (
            candidate_rows[0],
            candidate_rows[1],
            candidate_rows[1],
        )

    else:
        baseline_rows = (
            baseline_rows[0],
            baseline_rows[1],
            baseline_rows[1],
        )

    with pytest.raises(ContractError):
        _clone_request(
            base,
            candidate_history_rows=candidate_rows,
            baseline_history_rows=baseline_rows,
        )


@pytest.mark.parametrize(
    "actual_round",
    (
        1232,
        1234,
    ),
)
def test_runtime_rejects_actual_draw_round_mismatch(
    actual_round: int,
) -> None:
    fixture = _foundation()
    base = fixture._request()

    actual_draws = (
        SimpleNamespace(
            round_no=actual_round,
            numbers=(
                2,
                8,
                16,
                23,
                31,
                44,
            ),
        ),
    )

    request = _clone_request(
        base,
        actual_draws=actual_draws,
    )

    with pytest.raises(ContractError):
        TopKLiveEvaluationRuntimeService().execute(
            request=request
        )


@pytest.mark.parametrize(
    (
        "start_round",
        "end_round",
    ),
    (
        (1232, 1232),
        (1234, 1234),
    ),
)
def test_runtime_rejects_window_outside_prediction_round(
    start_round: int,
    end_round: int,
) -> None:
    fixture = _foundation()
    base = fixture._request()

    window = EvaluationWindow(
        name="ag22-invalid-window",
        start_round=start_round,
        end_round=end_round,
    )

    request = _clone_request(
        base,
        window=window,
    )

    with pytest.raises(ContractError):
        TopKLiveEvaluationRuntimeService().execute(
            request=request
        )


def test_runtime_preserves_optional_none_provenance() -> None:
    fixture = _foundation()
    base = fixture._request()

    request = _clone_request(
        base,
        candidate_regime_id=None,
        candidate_strategy_name=None,
        baseline_regime_id=None,
        baseline_strategy_name=None,
    )

    result = (
        TopKLiveEvaluationRuntimeService()
        .execute(
            request=request
        )
    )

    assert result.source_pair.candidate.regime_id is None
    assert result.source_pair.candidate.strategy_name is None
    assert result.source_pair.baseline.regime_id is None
    assert result.source_pair.baseline.strategy_name is None


def test_runtime_does_not_mutate_request_inputs() -> None:
    fixture = _foundation()

    request = fixture._request()

    candidate = request.candidate_prediction_result
    baseline = request.baseline_prediction_result

    candidate_rows = request.candidate_history_rows
    baseline_rows = request.baseline_history_rows
    actual_draws = request.actual_draws

    candidate_generation = candidate.generation
    candidate_diversity = candidate.diversity

    baseline_generation = baseline.generation
    baseline_diversity = baseline.diversity

    TopKLiveEvaluationRuntimeService().execute(
        request=request
    )

    assert request.candidate_prediction_result is candidate
    assert request.baseline_prediction_result is baseline

    assert request.candidate_history_rows is candidate_rows
    assert request.baseline_history_rows is baseline_rows
    assert request.actual_draws is actual_draws

    assert candidate.generation is candidate_generation
    assert candidate.diversity is candidate_diversity

    assert baseline.generation is baseline_generation
    assert baseline.diversity is baseline_diversity


def test_runtime_result_identity_chain_is_consistent() -> None:
    fixture = _foundation()

    request = fixture._request()

    result = (
        TopKLiveEvaluationRuntimeService()
        .execute(
            request=request
        )
    )

    assert (
        result.source_pair.candidate.model_name
        == result.evaluation.candidate_binding.model_name
    )

    assert (
        result.source_pair.baseline.model_name
        == result.evaluation.baseline_binding.model_name
    )

    assert (
        result.source_pair.candidate.round_no
        == result.source_pair.baseline.round_no
    )

    assert (
        result.source_pair.candidate.round_no
        == request.candidate_prediction_result
        .generation
        .request
        .round_no
    )


def test_runtime_product_has_no_filesystem_or_process_dependency() -> None:
    source = Path(
        "lrp/evaluation/"
        "topk_live_evaluation_runtime.py"
    ).read_text(
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
    )

    assert not any(
        token in source
        for token in forbidden
    )


def test_runtime_product_has_no_runtime_nondeterminism_dependency() -> None:
    source = Path(
        "lrp/evaluation/"
        "topk_live_evaluation_runtime.py"
    ).read_text(
        encoding="utf-8-sig"
    )

    forbidden = (
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