from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any
import importlib
import inspect

import pytest

from lrp.contracts.exceptions import ContractError
from lrp.evaluation.contracts import EvaluationWindow
from lrp.evaluation.topk_replay_adapter import TopKReplayPrediction


PRODUCT_MODULE = (
    "lrp.operations."
    "durable_replay_evaluation_orchestrator"
)

PRODUCT_PATH = Path(
    "lrp/operations/"
    "durable_replay_evaluation_orchestrator.py"
)


def _product():
    return importlib.import_module(
        PRODUCT_MODULE
    )


def _spec(
    *,
    artifact_path: str | Path = "candidate.json",
    history_rounds: tuple[int, ...] = (1200, 1201),
    model_name: str = "model-A",
    regime_id: str | None = None,
    strategy_name: str | None = None,
):
    product = _product()

    return product.DurableReplayEvaluationSourceSpec(
        artifact_path=artifact_path,
        history_rounds=history_rounds,
        model_name=model_name,
        regime_id=regime_id,
        strategy_name=strategy_name,
    )


def _prediction(
    *,
    round_no: int = 1233,
    model_name: str = "model-A",
) -> TopKReplayPrediction:
    return TopKReplayPrediction(
        round_no=round_no,
        history_rounds=(1231, 1232),
        predictions=(
            (1, 7, 13, 24, 32, 41),
            (2, 8, 17, 25, 34, 42),
            (3, 9, 18, 26, 35, 43),
            (4, 10, 19, 27, 36, 44),
            (5, 11, 20, 28, 37, 45),
            (6, 12, 21, 29, 33, 40),
            (1, 14, 22, 30, 38, 45),
            (2, 15, 23, 31, 39, 44),
            (3, 16, 24, 32, 40, 43),
            (4, 17, 25, 33, 41, 42),
        ),
        model_name=model_name,
    )


def _window() -> EvaluationWindow:
    return EvaluationWindow(
        name="ak-window",
        start_round=1233,
        end_round=1234,
    )


def test_source_spec_is_frozen() -> None:
    spec = _spec()

    with pytest.raises(FrozenInstanceError):
        spec.model_name = "changed"


def test_source_spec_public_fields_are_exact() -> None:
    product = _product()

    fields = tuple(
        product
        .DurableReplayEvaluationSourceSpec
        .__dataclass_fields__
    )

    assert fields == (
        "artifact_path",
        "history_rounds",
        "model_name",
        "regime_id",
        "strategy_name",
    )


def test_orchestrator_is_parameterless() -> None:
    product = _product()

    assert (
        inspect.signature(
            product.DurableReplayEvaluationOrchestrator
        )
        == inspect.Signature()
    )


def test_evaluate_public_signature_is_exact() -> None:
    product = _product()

    signature = inspect.signature(
        product
        .DurableReplayEvaluationOrchestrator
        .evaluate
    )

    assert tuple(signature.parameters) == (
        "self",
        "window",
        "candidate_sources",
        "baseline_sources",
        "actual_draws",
    )


def test_evaluate_returns_service_result(
    monkeypatch,
) -> None:
    product = _product()
    expected = object()

    monkeypatch.setattr(
        product.DurableReplayOperationalConsumer,
        "load",
        lambda self, **kwargs: _prediction(),
    )

    monkeypatch.setattr(
        product.TopKReplayEvaluationService,
        "evaluate",
        lambda self, *, request: expected,
    )

    result = (
        product.DurableReplayEvaluationOrchestrator()
        .evaluate(
            window=_window(),
            candidate_sources=(_spec(),),
            baseline_sources=(
                _spec(
                    artifact_path="baseline.json",
                    model_name="model-B",
                ),
            ),
            actual_draws=(),
        )
    )

    assert result is expected


@pytest.mark.parametrize(
    "count",
    [1, 2, 3],
)
def test_evaluate_loads_each_candidate_source(
    monkeypatch,
    count: int,
) -> None:
    product = _product()
    calls = []

    def fake_load(self, **kwargs):
        calls.append(kwargs)
        return _prediction(
            round_no=1232 + len(calls),
            model_name="candidate",
        )

    monkeypatch.setattr(
        product.DurableReplayOperationalConsumer,
        "load",
        fake_load,
    )

    monkeypatch.setattr(
        product.TopKReplayEvaluationService,
        "evaluate",
        lambda self, *, request: request,
    )

    sources = tuple(
        _spec(
            artifact_path=f"candidate-{i}.json",
            model_name="candidate",
        )
        for i in range(count)
    )

    product.DurableReplayEvaluationOrchestrator().evaluate(
        window=_window(),
        candidate_sources=sources,
        baseline_sources=(),
        actual_draws=(),
    )

    assert len(calls) == count


@pytest.mark.parametrize(
    "count",
    [1, 2, 3],
)
def test_evaluate_loads_each_baseline_source(
    monkeypatch,
    count: int,
) -> None:
    product = _product()
    calls = []

    def fake_load(self, **kwargs):
        calls.append(kwargs)
        return _prediction(
            round_no=1232 + len(calls),
            model_name="baseline",
        )

    monkeypatch.setattr(
        product.DurableReplayOperationalConsumer,
        "load",
        fake_load,
    )

    monkeypatch.setattr(
        product.TopKReplayEvaluationService,
        "evaluate",
        lambda self, *, request: request,
    )

    sources = tuple(
        _spec(
            artifact_path=f"baseline-{i}.json",
            model_name="baseline",
        )
        for i in range(count)
    )

    product.DurableReplayEvaluationOrchestrator().evaluate(
        window=_window(),
        candidate_sources=(),
        baseline_sources=sources,
        actual_draws=(),
    )

    assert len(calls) == count


@pytest.mark.parametrize(
    "bad_item",
    [
        None,
        object(),
        {},
        [],
        (),
        "bad",
        1,
        True,
    ],
)
def test_evaluate_rejects_invalid_candidate_source_item(
    bad_item,
) -> None:
    product = _product()

    with pytest.raises(ContractError):
        product.DurableReplayEvaluationOrchestrator().evaluate(
            window=_window(),
            candidate_sources=(bad_item,),
            baseline_sources=(),
            actual_draws=(),
        )


@pytest.mark.parametrize(
    "bad_item",
    [
        None,
        object(),
        {},
        [],
        (),
        "bad",
        1,
        True,
    ],
)
def test_evaluate_rejects_invalid_baseline_source_item(
    bad_item,
) -> None:
    product = _product()

    with pytest.raises(ContractError):
        product.DurableReplayEvaluationOrchestrator().evaluate(
            window=_window(),
            candidate_sources=(),
            baseline_sources=(bad_item,),
            actual_draws=(),
        )


def test_product_has_no_direct_json_dependency() -> None:
    source = PRODUCT_PATH.read_text(
        encoding="utf-8-sig"
    )

    assert "import json" not in source
    assert "json.loads" not in source
    assert "json.load" not in source


def test_product_has_no_predictionresult_dependency() -> None:
    source = PRODUCT_PATH.read_text(
        encoding="utf-8-sig"
    )

    assert "PredictionResult" not in source


def test_product_has_no_filesystem_write_dependency() -> None:
    source = PRODUCT_PATH.read_text(
        encoding="utf-8-sig"
    )

    forbidden = (
        "write_text(",
        "write_bytes(",
        "mkdir(",
        "write_operation_artifact",
        "write_prediction_artifacts",
    )

    assert not any(
        token in source
        for token in forbidden
    )


def test_product_has_no_artifact_path_derivation() -> None:
    source = PRODUCT_PATH.read_text(
        encoding="utf-8-sig"
    )

    forbidden = (
        "evaluation_source.json",
        "prediction-evaluation-sources",
        "output_root",
    )

    assert not any(
        token in source
        for token in forbidden
    )


def test_product_has_no_cli_dependency() -> None:
    source = PRODUCT_PATH.read_text(
        encoding="utf-8-sig"
    )

    assert "lrp.cli" not in source