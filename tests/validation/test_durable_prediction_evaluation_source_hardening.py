from __future__ import annotations

import ast
import hashlib
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lrp.contracts.exceptions import ContractError
from lrp.pipelines.durable_prediction_evaluation_source import (
    DurablePredictionEvaluationSource,
    source_from_dict,
    source_from_json,
    source_to_dict,
    source_to_json,
)


KST = timezone(
    timedelta(hours=9)
)


def _source(
    **changes,
) -> DurablePredictionEvaluationSource:
    values = {
        "schema_version": "1.0",
        "round_no": 1233,
        "top_k": 1,
        "selected_sets": (
            (1, 7, 13, 24, 32, 41),
        ),
        "generated_at_kst": datetime(
            2026,
            8,
            21,
            17,
            0,
            tzinfo=KST,
        ),
    }

    values.update(
        changes
    )

    return DurablePredictionEvaluationSource(
        **values
    )


@pytest.mark.parametrize(
    "value",
    (
        None,
        object(),
        {},
        [],
    ),
)
def test_source_to_dict_rejects_invalid_source_type(
    value,
) -> None:
    with pytest.raises(
        ContractError
    ):
        source_to_dict(
            value
        )


@pytest.mark.parametrize(
    "value",
    (
        None,
        object(),
        {},
        [],
    ),
)
def test_source_to_json_rejects_invalid_source_type(
    value,
) -> None:
    with pytest.raises(
        ContractError
    ):
        source_to_json(
            value
        )


@pytest.mark.parametrize(
    "value",
    (
        None,
        [],
        (),
        "text",
        1,
        True,
    ),
)
def test_source_from_dict_rejects_non_dict_type(
    value,
) -> None:
    with pytest.raises(
        ContractError
    ):
        source_from_dict(
            value
        )


@pytest.mark.parametrize(
    "value",
    (
        None,
        object(),
        "abcdef",
        {
            "a": (
                1,
                7,
                13,
                24,
                32,
                41,
            ),
        },
    ),
)
def test_source_from_dict_rejects_malformed_selected_sets(
    value,
) -> None:
    payload = source_to_dict(
        _source()
    )

    payload[
        "selected_sets"
    ] = value

    with pytest.raises(
        ContractError
    ):
        source_from_dict(
            payload
        )


@pytest.mark.parametrize(
    "value",
    (
        None,
        b"{}",
        {},
        [],
        1,
        True,
    ),
)
def test_source_from_json_rejects_non_string_type(
    value,
) -> None:
    with pytest.raises(
        ContractError
    ):
        source_from_json(
            value
        )


def test_source_canonicalizes_reverse_number_order() -> None:
    source = _source(
        selected_sets=(
            (
                41,
                32,
                24,
                13,
                7,
                1,
            ),
        )
    )

    assert source.selected_sets == (
        (
            1,
            7,
            13,
            24,
            32,
            41,
        ),
    )


def test_source_accepts_any_plus_nine_offset_timezone() -> None:
    custom_kst = timezone(
        timedelta(hours=9),
        "CUSTOM",
    )

    source = _source(
        generated_at_kst=datetime(
            2026,
            8,
            21,
            17,
            0,
            tzinfo=custom_kst,
        )
    )

    assert (
        source.generated_at_kst.utcoffset()
        == timedelta(hours=9)
    )


def test_source_constructor_does_not_mutate_input() -> None:
    selected_sets = [
        [
            41,
            1,
            32,
            7,
            24,
            13,
        ],
    ]

    before = deepcopy(
        selected_sets
    )

    _source(
        selected_sets=selected_sets
    )

    assert (
        selected_sets
        == before
    )


def test_source_to_dict_returns_isolated_nested_copy() -> None:
    source = _source()

    payload = source_to_dict(
        source
    )

    payload[
        "selected_sets"
    ][0][0] = 45

    assert source.selected_sets == (
        (
            1,
            7,
            13,
            24,
            32,
            41,
        ),
    )


def test_source_json_is_byte_stable_across_repeated_calls() -> None:
    source = _source()

    values = tuple(
        source_to_json(
            source
        )
        for _ in range(10)
    )

    assert len(
        set(values)
    ) == 1

    hashes = {
        hashlib.sha256(
            value.encode(
                "utf-8"
            )
        ).hexdigest()
        for value in values
    }

    assert len(
        hashes
    ) == 1


def test_source_product_exception_boundary_is_narrow() -> None:
    path = Path(
        "lrp/pipelines/"
        "durable_prediction_evaluation_source.py"
    )

    source = path.read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(
        source
    )

    for node in ast.walk(
        tree
    ):
        if not isinstance(
            node,
            ast.ExceptHandler,
        ):
            continue

        assert node.type is not None

        if isinstance(
            node.type,
            ast.Name,
        ):
            assert node.type.id not in {
                "Exception",
                "BaseException",
            }

        elif isinstance(
            node.type,
            ast.Tuple,
        ):
            names = {
                item.id
                for item in node.type.elts
                if isinstance(
                    item,
                    ast.Name,
                )
            }

            assert "Exception" not in names
            assert "BaseException" not in names


def test_source_product_has_no_runtime_or_persistence_dependency() -> None:
    path = Path(
        "lrp/pipelines/"
        "durable_prediction_evaluation_source.py"
    )

    source = path.read_text(
        encoding="utf-8-sig"
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
                alias.name
                for alias in node.names
            )

        elif isinstance(
            node,
            ast.ImportFrom,
        ) and node.module:
            imports.append(
                node.module
            )

    forbidden_roots = {
        "os",
        "pathlib",
        "subprocess",
        "sqlite3",
        "random",
        "secrets",
        "uuid",
        "time",
        "requests",
        "httpx",
    }

    assert not any(
        name.split(
            ".",
            1,
        )[0] in forbidden_roots
        for name in imports
    )

    forbidden_tokens = (
        "PredictionResult",
        "write_prediction_artifacts",
        "write_operation_artifact",
        "datetime.now",
        "datetime.utcnow",
        "time.time",
    )

    assert not any(
        token in source
        for token in forbidden_tokens
    )