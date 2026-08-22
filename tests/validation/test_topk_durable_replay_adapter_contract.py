from __future__ import annotations

import ast
import importlib
import inspect
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lrp.contracts.exceptions import ContractError
from lrp.evaluation.topk_replay_adapter import TopKReplayPrediction
from lrp.pipelines.durable_prediction_evaluation_source import (
    DurablePredictionEvaluationSource,
)


PRODUCT_MODULE = "lrp.evaluation.topk_durable_replay_adapter"
PRODUCT_PATH = Path(
    "lrp/evaluation/topk_durable_replay_adapter.py"
)


def _product():
    return importlib.import_module(
        PRODUCT_MODULE
    )


def _kst():
    return timezone(
        timedelta(
            hours=9
        )
    )


def _source() -> DurablePredictionEvaluationSource:
    return DurablePredictionEvaluationSource(
        schema_version="1.0",
        round_no=1233,
        top_k=10,
        selected_sets=(
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
        generated_at_kst=datetime(
            2026,
            8,
            22,
            15,
            0,
            tzinfo=_kst(),
        ),
    )


def _adapt(
    *,
    source: DurablePredictionEvaluationSource | object | None = None,
    history_rounds: tuple[int, ...] = (1230, 1231, 1232),
    model_name: str = "candidate-v1",
    regime_id: str | None = "regime-A",
    strategy_name: str | None = "strategy-A",
):
    product = _product()
    adapter = product.TopKDurableReplayAdapter()

    if source is None:
        source = _source()

    return adapter.adapt(
        source=source,
        history_rounds=history_rounds,
        model_name=model_name,
        regime_id=regime_id,
        strategy_name=strategy_name,
    )


def test_adapter_is_parameterless() -> None:
    product = _product()

    assert inspect.signature(
        product.TopKDurableReplayAdapter
    ) == inspect.Signature()


def test_adapt_public_signature_is_exact() -> None:
    product = _product()

    signature = inspect.signature(
        product.TopKDurableReplayAdapter.adapt
    )

    parameters = tuple(
        signature.parameters.values()
    )

    assert tuple(
        item.name
        for item in parameters
    ) == (
        "self",
        "source",
        "history_rounds",
        "model_name",
        "regime_id",
        "strategy_name",
    )

    assert parameters[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD

    for item in parameters[1:]:
        assert item.kind is inspect.Parameter.KEYWORD_ONLY

    assert parameters[1].annotation == "DurablePredictionEvaluationSource"
    assert parameters[2].annotation == "tuple[int, ...]"
    assert parameters[3].annotation == "str"
    assert parameters[4].annotation == "str | None"
    assert parameters[5].annotation == "str | None"

    assert parameters[4].default is None
    assert parameters[5].default is None

    assert signature.return_annotation == "TopKReplayPrediction"


def test_adapt_returns_topk_replay_prediction() -> None:
    result = _adapt()

    assert isinstance(
        result,
        TopKReplayPrediction,
    )


def test_adapt_preserves_round_no() -> None:
    source = _source()

    result = _adapt(
        source=source,
    )

    assert result.round_no == source.round_no


def test_adapt_projects_selected_sets_to_predictions() -> None:
    source = _source()

    result = _adapt(
        source=source,
    )

    assert result.predictions == source.selected_sets


def test_adapt_preserves_history_rounds() -> None:
    history_rounds = (
        1229,
        1230,
        1231,
        1232,
    )

    result = _adapt(
        history_rounds=history_rounds,
    )

    assert result.history_rounds == history_rounds


def test_adapt_preserves_model_name() -> None:
    result = _adapt(
        model_name="model-X",
    )

    assert result.model_name == "model-X"


def test_adapt_preserves_optional_regime_id() -> None:
    result = _adapt(
        regime_id="regime-X",
    )

    assert result.regime_id == "regime-X"


def test_adapt_preserves_optional_strategy_name() -> None:
    result = _adapt(
        strategy_name="strategy-X",
    )

    assert result.strategy_name == "strategy-X"


def test_adapt_preserves_none_provenance() -> None:
    result = _adapt(
        regime_id=None,
        strategy_name=None,
    )

    assert result.regime_id is None
    assert result.strategy_name is None


@pytest.mark.parametrize(
    "invalid_source",
    (
        object(),
        {},
        [],
        (),
        "source",
        1,
        True,
    ),
)
def test_adapt_rejects_invalid_source_type(
    invalid_source: object,
) -> None:
    with pytest.raises(
        ContractError,
    ):
        _adapt(
            source=invalid_source,
        )


def test_adapt_delegates_invalid_history_to_replay_prediction() -> None:
    with pytest.raises(
        ContractError,
    ):
        _adapt(
            history_rounds=(
                1232,
                1231,
                1230,
            ),
        )


def test_adapt_delegates_invalid_model_name_to_replay_prediction() -> None:
    with pytest.raises(
        ContractError,
    ):
        _adapt(
            model_name="",
        )


def test_adapt_delegates_invalid_regime_id_to_replay_prediction() -> None:
    with pytest.raises(
        ContractError,
    ):
        _adapt(
            regime_id="",
        )


def test_adapt_delegates_invalid_strategy_name_to_replay_prediction() -> None:
    with pytest.raises(
        ContractError,
    ):
        _adapt(
            strategy_name="",
        )


def test_adapt_does_not_mutate_source() -> None:
    source = _source()

    before = (
        source.schema_version,
        source.round_no,
        source.top_k,
        source.selected_sets,
        source.generated_at_kst,
    )

    _adapt(
        source=source,
    )

    after = (
        source.schema_version,
        source.round_no,
        source.top_k,
        source.selected_sets,
        source.generated_at_kst,
    )

    assert after == before


def test_adapt_does_not_mutate_history_rounds() -> None:
    history_rounds = (
        1230,
        1231,
        1232,
    )

    before = history_rounds

    _adapt(
        history_rounds=history_rounds,
    )

    assert history_rounds == before


def test_adapt_is_semantically_deterministic() -> None:
    source = _source()

    first = _adapt(
        source=source,
    )

    second = _adapt(
        source=source,
    )

    assert first == second


def test_product_has_no_filesystem_dependency() -> None:
    product = _product()

    source = inspect.getsource(
        product
    )

    tree = ast.parse(
        source
    )

    imports = []

    for node in ast.walk(
        tree
    ):
        if isinstance(
            node,
            ast.Import,
        ):
            imports.extend(
                item.name
                for item in node.names
            )

        elif isinstance(
            node,
            ast.ImportFrom,
        ):
            if node.module:
                imports.append(
                    node.module
                )

    forbidden = {
        "pathlib",
        "os",
    }

    assert not (
        forbidden
        & set(imports)
    )

    forbidden_tokens = (
        "Path(",
        "open(",
        "read_text",
        "read_bytes",
        "write_text",
        "write_bytes",
        "evaluation_source.json",
    )

    assert not any(
        token in source
        for token in forbidden_tokens
    )


def test_product_has_no_predictionresult_dependency() -> None:
    product = _product()

    source = inspect.getsource(
        product
    )

    assert "PredictionResult" not in source
    assert "prediction_to_dict" not in source
    assert "prediction_to_json" not in source


def test_product_has_no_replay_execution_dependency() -> None:
    product = _product()

    source = inspect.getsource(
        product
    )

    forbidden_tokens = (
        "TopKReplayEvaluationService",
        "TopKReplayEvaluationRequest",
        "TopKReplayRow",
        "WalkForwardEvaluation",
        "actual_draw",
        "actual_draws",
        "write_operation_artifact",
        "write_prediction_artifacts",
    )

    assert not any(
        token in source
        for token in forbidden_tokens
    )