from __future__ import annotations

import ast
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from lrp.contracts.exceptions import ContractError
from lrp.evaluation.topk_prediction_source_adapter import (
    TopKPredictionSourceAdapter,
    TopKPredictionSourceRecord,
)
from lrp.pipelines.models import (
    PredictionGenerationResult,
    PredictionRequest,
    PredictionResult,
)


PRODUCT = Path(
    "lrp/evaluation/topk_prediction_source_adapter.py"
)


def _candidate(
    numbers: object,
) -> object:
    return SimpleNamespace(
        numbers=numbers,
    )


def _ten_candidates() -> tuple[object, ...]:
    return tuple(
        _candidate(
            (
                start + 5,
                start,
                start + 3,
                start + 2,
                start + 4,
                start + 1,
            )
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


def _prediction_result(
    *,
    selected: object | None = None,
    diversity: object | None = None,
    top_k: int = 10,
) -> PredictionResult:
    request = PredictionRequest(
        round_no=1200,
        seed=20260821,
        top_k=top_k,
        practical_k=min(
            5,
            top_k,
        ),
        long_gap_numbers=frozenset(
            {
                45,
            }
        ),
    )

    generation = PredictionGenerationResult(
        request=request,
        windows=(
            10,
            20,
            50,
        ),
        probabilities={
            1:
                1.0,
        },
        statistics_contract=object(),
        number_signals={},
        candidates=(),
        statistics_version="test-statistics",
        candidate_version="test-candidate",
    )

    if diversity is None:
        if selected is None:
            selected = _ten_candidates()

        diversity = SimpleNamespace(
            selected=selected,
        )

    return PredictionResult(
        generation=generation,
        scored_candidates=(),
        ranking=object(),
        diversity=diversity,
        practical=object(),
        generated_at_kst=datetime.now(
            timezone.utc
        ),
    )


def _source(
    *,
    prediction_result: PredictionResult | None = None,
    history_rounds: tuple[int, ...] = (
        1197,
        1198,
        1199,
    ),
    regime_id: str | None = "regime-a",
    strategy_name: str | None = "strategy-a",
) -> TopKPredictionSourceRecord:
    return TopKPredictionSourceRecord(
        prediction_result=(
            _prediction_result()
            if prediction_result is None
            else prediction_result
        ),
        model_name="candidate",
        history_rounds=history_rounds,
        regime_id=regime_id,
        strategy_name=strategy_name,
    )


def _adapt(
    source: TopKPredictionSourceRecord | None = None,
):
    return TopKPredictionSourceAdapter().adapt(
        source=(
            _source()
            if source is None
            else source
        )
    )


def test_adapter_rejects_wrong_source_type() -> None:
    with pytest.raises(
        ContractError
    ):
        TopKPredictionSourceAdapter().adapt(
            source=object()  # type: ignore[arg-type]
        )


def test_adapter_rejects_non_iterable_selected_output() -> None:
    result = _prediction_result(
        diversity=SimpleNamespace(
            selected=123,
        )
    )

    with pytest.raises(
        ContractError
    ):
        _adapt(
            _source(
                prediction_result=result,
            )
        )


def test_adapter_rejects_string_selected_output() -> None:
    result = _prediction_result(
        diversity=SimpleNamespace(
            selected="not-selected-items",
        )
    )

    with pytest.raises(
        ContractError
    ):
        _adapt(
            _source(
                prediction_result=result,
            )
        )


def test_adapter_accepts_mapping_candidate_numbers() -> None:
    selected = tuple(
        {
            "numbers":
                (
                    start + 5,
                    start,
                    start + 3,
                    start + 2,
                    start + 4,
                    start + 1,
                )
        }
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

    result = _prediction_result(
        selected=selected,
    )

    replay = _adapt(
        _source(
            prediction_result=result,
        )
    )

    assert len(
        replay.predictions
    ) == 10

    assert replay.predictions[0] == (
        1,
        2,
        3,
        4,
        5,
        6,
    )


def test_adapter_rejects_mapping_candidate_without_numbers() -> None:
    selected = list(
        _ten_candidates()
    )

    selected[0] = {
        "other":
            (
                1,
                2,
                3,
                4,
                5,
                6,
            )
    }

    result = _prediction_result(
        selected=tuple(
            selected
        )
    )

    with pytest.raises(
        ContractError
    ):
        _adapt(
            _source(
                prediction_result=result,
            )
        )


def test_adapter_rejects_mapping_candidate_with_none_numbers() -> None:
    selected = list(
        _ten_candidates()
    )

    selected[0] = {
        "numbers":
            None,
    }

    result = _prediction_result(
        selected=tuple(
            selected
        )
    )

    with pytest.raises(
        ContractError
    ):
        _adapt(
            _source(
                prediction_result=result,
            )
        )


def test_adapter_rejects_candidate_without_numbers_attribute() -> None:
    selected = list(
        _ten_candidates()
    )

    selected[0] = object()

    result = _prediction_result(
        selected=tuple(
            selected
        )
    )

    with pytest.raises(
        ContractError
    ):
        _adapt(
            _source(
                prediction_result=result,
            )
        )


def test_adapter_rejects_top_k_below_replay_minimum() -> None:
    result = _prediction_result(
        top_k=9,
        selected=_ten_candidates()[:9],
    )

    with pytest.raises(
        ContractError
    ):
        _adapt(
            _source(
                prediction_result=result,
            )
        )


def test_repeated_adaptation_is_semantically_stable() -> None:
    source = _source(
        regime_id=None,
        strategy_name=None,
    )

    first = _adapt(
        source
    )

    second = _adapt(
        source
    )

    assert first == second

    assert repr(
        first
    ) == repr(
        second
    )


def test_prediction_order_is_deterministic() -> None:
    selected = tuple(
        _candidate(
            (
                start + 5,
                start + 4,
                start + 3,
                start + 2,
                start + 1,
                start,
            )
        )
        for start in (
            37,
            33,
            29,
            25,
            21,
            17,
            13,
            9,
            5,
            1,
        )
    )

    source = _source(
        prediction_result=_prediction_result(
            selected=selected,
        )
    )

    result = _adapt(
        source
    )

    assert result.predictions[0] == (
        37,
        38,
        39,
        40,
        41,
        42,
    )

    assert result.predictions[-1] == (
        1,
        2,
        3,
        4,
        5,
        6,
    )


def test_number_normalization_is_deterministic() -> None:
    source = _source()

    first = _adapt(
        source
    )

    second = _adapt(
        source
    )

    assert first.predictions == second.predictions

    for row in first.predictions:
        assert row == tuple(
            sorted(
                row
            )
        )


def test_source_history_tuple_is_not_mutated() -> None:
    history = (
        1194,
        1195,
        1196,
        1197,
        1198,
        1199,
    )

    source = _source(
        history_rounds=history,
    )

    before = source.history_rounds

    _adapt(
        source
    )

    assert source.history_rounds == before
    assert source.history_rounds == history


def test_selected_candidate_numbers_are_not_mutated() -> None:
    selected = _ten_candidates()

    before = tuple(
        tuple(
            item.numbers
        )
        for item in selected
    )

    result = _prediction_result(
        selected=selected,
    )

    source = _source(
        prediction_result=result,
    )

    _adapt(
        source
    )

    after = tuple(
        tuple(
            item.numbers
        )
        for item in selected
    )

    assert after == before


def test_optional_none_provenance_is_stable() -> None:
    source = _source(
        regime_id=None,
        strategy_name=None,
    )

    first = _adapt(
        source
    )

    second = _adapt(
        source
    )

    assert first.regime_id is None
    assert first.strategy_name is None

    assert second.regime_id is None
    assert second.strategy_name is None


def test_adapter_has_no_randomness_dependency() -> None:
    tree = ast.parse(
        PRODUCT.read_text(
            encoding="utf-8-sig"
        )
    )

    imports: list[str] = []

    for node in ast.walk(
        tree
    ):
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
            imports.append(
                node.module
                or ""
            )

    assert not any(
        value == "random"
        or value.startswith(
            "random."
        )
        or value == "secrets"
        or value.startswith(
            "secrets."
        )
        for value in imports
    )


def test_adapter_has_no_filesystem_dependency() -> None:
    tree = ast.parse(
        PRODUCT.read_text(
            encoding="utf-8-sig"
        )
    )

    imports: list[str] = []

    calls: list[str] = []

    for node in ast.walk(
        tree
    ):
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
            imports.append(
                node.module
                or ""
            )

        elif isinstance(
            node,
            ast.Call,
        ):
            if isinstance(
                node.func,
                ast.Name,
            ):
                calls.append(
                    node.func.id
                )

            elif isinstance(
                node.func,
                ast.Attribute,
            ):
                calls.append(
                    node.func.attr
                )

    forbidden_imports = (
        "pathlib",
        "os",
        "shutil",
        "tempfile",
    )

    assert not any(
        value == token
        or value.startswith(
            token + "."
        )
        for value in imports
        for token in forbidden_imports
    )

    forbidden_calls = {
        "open",
        "read_text",
        "read_bytes",
        "write_text",
        "write_bytes",
        "unlink",
        "mkdir",
        "rename",
        "replace",
    }

    assert not (
        set(
            calls
        )
        & forbidden_calls
    )
