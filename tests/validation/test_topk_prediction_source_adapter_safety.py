from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone

from lrp.pipelines.models import (
    PredictionGenerationResult,
    PredictionRequest,
    PredictionResult,
)
from lrp.evaluation.topk_prediction_source_adapter import (
    TopKPredictionSourceAdapter,
    TopKPredictionSourceRecord,
)


PRODUCT = Path(
    "lrp/evaluation/topk_prediction_source_adapter.py"
)


def _result() -> PredictionResult:
    request = PredictionRequest(
        round_no=1200,
        seed=20260821,
        long_gap_numbers=frozenset({45}),
        top_k=10,
        practical_k=5,
    )

    generation = PredictionGenerationResult(
        request=request,
        windows=(10, 20, 50),
        probabilities={1: 1.0},
        statistics_contract=object(),
        number_signals={},
        candidates=(),
        statistics_version="test-statistics",
        candidate_version="test-candidate",
    )

    return PredictionResult(
        generation=generation,
        scored_candidates=(),
        ranking=object(),
        diversity=SimpleNamespace(
            selected=tuple(
                SimpleNamespace(
                    numbers=(
                        start + 5,
                        start + 4,
                        start + 3,
                        start + 2,
                        start + 1,
                        start,
                    ),
                )
                for start in (
                    1,
                    5,
                    9,
                    13,
                    17,
                    21,
                    25,
                    29,
                    33,
                    37,
                )
            )
        ),
        practical=object(),
        generated_at_kst=datetime.now(
            timezone.utc
        ),
    )


def _source() -> TopKPredictionSourceRecord:
    return TopKPredictionSourceRecord(
        prediction_result=_result(),
        model_name="candidate",
        history_rounds=(
            1197,
            1198,
            1199,
        ),
        regime_id=None,
        strategy_name=None,
    )


def _tree() -> ast.Module:
    return ast.parse(
        PRODUCT.read_text(
            encoding="utf-8-sig"
        )
    )


def _imports() -> tuple[str, ...]:
    values: list[str] = []

    for node in ast.walk(
        _tree()
    ):
        if isinstance(
            node,
            ast.Import,
        ):
            values.extend(
                alias.name
                for alias in node.names
            )

        elif isinstance(
            node,
            ast.ImportFrom,
        ):
            module = (
                node.module
                or ""
            )

            values.append(
                module
            )

            values.extend(
                (
                    module
                    + "."
                    + alias.name
                ).strip(
                    "."
                )
                for alias in node.names
            )

    return tuple(
        values
    )


def _call_names() -> tuple[str, ...]:
    values: list[str] = []

    for node in ast.walk(
        _tree()
    ):
        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        if isinstance(
            node.func,
            ast.Name,
        ):
            values.append(
                node.func.id
            )

        elif isinstance(
            node.func,
            ast.Attribute,
        ):
            values.append(
                node.func.attr
            )

    return tuple(
        values
    )


def test_adapter_is_read_only() -> None:
    forbidden = {
        "open",
        "write",
        "write_text",
        "write_bytes",
        "unlink",
        "mkdir",
        "replace",
        "rename",
        "remove",
        "rmdir",
        "touch",
        "dump",
        "dumps",
    }

    calls = set(
        _call_names()
    )

    assert not (
        calls
        & forbidden
    )


def test_adapter_does_not_mutate_prediction_result() -> None:
    source = _source()

    before_history = source.history_rounds

    before_selected = tuple(
        tuple(
            item.numbers
        )
        for item in source.prediction_result.diversity.selected
    )

    result = TopKPredictionSourceAdapter().adapt(
        source=source
    )

    after_history = source.history_rounds

    after_selected = tuple(
        tuple(
            item.numbers
        )
        for item in source.prediction_result.diversity.selected
    )

    assert before_history == after_history

    assert before_selected == after_selected

    assert result.history_rounds == before_history


def test_adapter_does_not_import_serializer_private_helpers() -> None:
    imports = _imports()

    assert not any(
        item.startswith(
            "lrp.pipelines.serializer"
        )
        for item in imports
    )


def test_adapter_has_no_pipeline_execution_dependency() -> None:
    imports = _imports()

    forbidden = (
        "lrp.pipelines.prediction",
        "PredictionPipeline",
    )

    assert not any(
        item in imports
        for item in forbidden
    )


def test_adapter_has_no_draw_io_dependency() -> None:
    imports = _imports()

    assert not any(
        item == "lrp.io.draws"
        or item.startswith(
            "lrp.io.draws."
        )
        for item in imports
    )


def test_adapter_has_no_evaluation_execution_dependency() -> None:
    imports = _imports()

    forbidden_tokens = (
        "topk_replay_evaluation",
        "topk_walkforward",
    )

    assert not any(
        any(
            token in item
            for token in forbidden_tokens
        )
        for item in imports
    )
