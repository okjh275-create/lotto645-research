from __future__ import annotations

import ast
import inspect
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lrp.contracts.exceptions import ContractError
from lrp.evaluation.topk_durable_replay_adapter import (
    TopKDurableReplayAdapter,
)
from lrp.evaluation.topk_replay_adapter import (
    TopKReplayPrediction,
)
from lrp.pipelines.durable_prediction_evaluation_source import (
    DurablePredictionEvaluationSource,
)


PRODUCT_PATH = Path(
    "lrp/evaluation/topk_durable_replay_adapter.py"
)


_POOL = (
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
    (5, 18, 26, 34, 39, 45),
    (6, 19, 27, 35, 40, 44),
)


def _kst():
    return timezone(
        timedelta(
            hours=9
        )
    )


def _source(
    top_k: int = 10,
) -> DurablePredictionEvaluationSource:
    return DurablePredictionEvaluationSource(
        schema_version="1.0",
        round_no=1233,
        top_k=top_k,
        selected_sets=_POOL[:top_k],
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
    source: DurablePredictionEvaluationSource | None = None,
    history_rounds: tuple[int, ...] = (1230, 1231, 1232),
    model_name: str = "model-A",
    regime_id: str | None = None,
    strategy_name: str | None = None,
):
    if source is None:
        source = _source()

    return TopKDurableReplayAdapter().adapt(
        source=source,
        history_rounds=history_rounds,
        model_name=model_name,
        regime_id=regime_id,
        strategy_name=strategy_name,
    )


@pytest.mark.parametrize(
    "top_k",
    (
        1,
        3,
        5,
        9,
    ),
)
def test_replay_rejects_ah_valid_source_below_minimum(
    top_k: int,
) -> None:
    source = _source(
        top_k
    )

    with pytest.raises(
        ContractError,
        match="at least ten sets",
    ):
        _adapt(
            source=source,
        )


def test_replay_accepts_exact_minimum_ten_sets() -> None:
    source = _source(
        10
    )

    result = _adapt(
        source=source,
    )

    assert isinstance(
        result,
        TopKReplayPrediction,
    )

    assert len(
        result.predictions
    ) == 10

    assert result.predictions == source.selected_sets


@pytest.mark.parametrize(
    "top_k",
    (
        11,
        12,
    ),
)
def test_replay_preserves_sources_above_minimum(
    top_k: int,
) -> None:
    source = _source(
        top_k
    )

    result = _adapt(
        source=source,
    )

    assert len(
        result.predictions
    ) == top_k

    assert result.predictions == source.selected_sets


@pytest.mark.parametrize(
    (
        "history_rounds",
        "message",
    ),
    (
        (
            (),
            "history_rounds must be non-empty",
        ),
        (
            (1232, 1231, 1230),
            "history_rounds must be strictly ascending",
        ),
        (
            (1230, 1231, 1231),
            "history_rounds must not contain duplicates",
        ),
        (
            (1231, 1232, 1233),
            "history rounds must be strictly before prediction round",
        ),
        (
            (1231, 1232, 1234),
            "history rounds must be strictly before prediction round",
        ),
        (
            (1230, True, 1232),
            "history round must be a positive integer",
        ),
    ),
)
def test_replay_history_failures_propagate_from_lower_layer(
    history_rounds: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(
        ContractError,
        match=message,
    ):
        _adapt(
            history_rounds=history_rounds,
        )


@pytest.mark.parametrize(
    "model_name",
    (
        "",
        " ",
        None,
        1,
        True,
    ),
)
def test_replay_model_identity_failures_propagate_from_lower_layer(
    model_name: object,
) -> None:
    with pytest.raises(
        ContractError,
        match="model_name must be a non-empty string",
    ):
        _adapt(
            model_name=model_name,
        )


@pytest.mark.parametrize(
    (
        "field",
        "value",
        "message",
    ),
    (
        (
            "regime_id",
            "",
            "regime_id must be non-empty when provided",
        ),
        (
            "regime_id",
            " ",
            "regime_id must be non-empty when provided",
        ),
        (
            "regime_id",
            1,
            "regime_id must be str or None",
        ),
        (
            "regime_id",
            True,
            "regime_id must be str or None",
        ),
        (
            "strategy_name",
            "",
            "strategy_name must be non-empty when provided",
        ),
        (
            "strategy_name",
            " ",
            "strategy_name must be non-empty when provided",
        ),
        (
            "strategy_name",
            1,
            "strategy_name must be str or None",
        ),
        (
            "strategy_name",
            True,
            "strategy_name must be str or None",
        ),
    ),
)
def test_replay_optional_provenance_failures_propagate_from_lower_layer(
    field: str,
    value: object,
    message: str,
) -> None:
    kwargs = {
        "source": _source(),
        "history_rounds": (
            1230,
            1231,
            1232,
        ),
        "model_name": "model-A",
        "regime_id": None,
        "strategy_name": None,
    }

    kwargs[field] = value

    with pytest.raises(
        ContractError,
        match=message,
    ):
        TopKDurableReplayAdapter().adapt(
            **kwargs
        )


def test_replay_optional_none_provenance_remains_none() -> None:
    result = _adapt(
        regime_id=None,
        strategy_name=None,
    )

    assert result.regime_id is None
    assert result.strategy_name is None


def test_replay_canonicalization_is_lower_layer_owned() -> None:
    source = _source(
        10
    )

    result = _adapt(
        source=source,
    )

    assert result.predictions == source.selected_sets

    assert (
        result.predictions
        is not source.selected_sets
    )


def test_replay_repeated_execution_is_semantically_stable() -> None:
    source = _source()

    history = (
        1230,
        1231,
        1232,
    )

    adapter = TopKDurableReplayAdapter()

    results = tuple(
        adapter.adapt(
            source=source,
            history_rounds=history,
            model_name="model-A",
            regime_id="regime-A",
            strategy_name="strategy-A",
        )
        for _ in range(10)
    )

    assert len(
        set(
            repr(item)
            for item in results
        )
    ) == 1


def test_replay_adapter_does_not_mutate_inputs() -> None:
    source = _source()

    history = (
        1230,
        1231,
        1232,
    )

    source_before = repr(
        source
    )

    history_before = repr(
        history
    )

    _adapt(
        source=source,
        history_rounds=history,
    )

    assert repr(
        source
    ) == source_before

    assert repr(
        history
    ) == history_before


def test_replay_adapter_has_no_exception_normalization_layer() -> None:
    source = PRODUCT_PATH.read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(
        source
    )

    handlers = [
        node
        for node in ast.walk(tree)
        if isinstance(
            node,
            ast.ExceptHandler,
        )
    ]

    assert handlers == []


def test_replay_adapter_static_boundary_is_exact() -> None:
    source = PRODUCT_PATH.read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(
        source
    )

    imports = []

    for node in ast.walk(tree):

        if isinstance(
            node,
            ast.Import,
        ):
            imports.extend(
                alias.name
                for alias in node.names
            )

        elif isinstance(
            node,
            ast.ImportFrom,
        ):
            if node.module:
                imports.append(
                    node.module
                )

    allowed = {
        "__future__",
        "lrp.contracts.exceptions",
        "lrp.evaluation.topk_replay_adapter",
        "lrp.pipelines.durable_prediction_evaluation_source",
    }

    assert set(
        imports
    ) <= allowed

    forbidden_tokens = (
        "PredictionResult",
        "TopKReplayEvaluationService",
        "TopKReplayEvaluationRequest",
        "TopKReplayRow",
        "WalkForwardEvaluation",
        "write_operation_artifact",
        "write_prediction_artifacts",
        "evaluation_source.json",
        "pathlib",
        "random",
        "secrets",
    )

    assert not any(
        token in source
        for token in forbidden_tokens
    )


def test_replay_adapter_public_surface_remains_minimal() -> None:
    from lrp.evaluation import topk_durable_replay_adapter as product

    public_classes = [
        name
        for name, value in vars(
            product
        ).items()
        if (
            inspect.isclass(
                value
            )
            and value.__module__
            == product.__name__
            and not name.startswith("_")
        )
    ]

    public_functions = [
        name
        for name, value in vars(
            product
        ).items()
        if (
            inspect.isfunction(
                value
            )
            and value.__module__
            == product.__name__
            and not name.startswith("_")
        )
    ]

    assert public_classes == [
        "TopKDurableReplayAdapter"
    ]

    assert public_functions == []