from __future__ import annotations

import ast
import importlib
import inspect
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lrp.contracts.exceptions import ContractError


KST = timezone(timedelta(hours=9))


def _product():
    return importlib.import_module(
        "lrp.pipelines.durable_prediction_evaluation_source"
    )


def _source(
    **changes,
):
    product = _product()

    values = {
        "schema_version": "1.0",
        "round_no": 1233,
        "top_k": 3,
        "selected_sets": (
            (1, 7, 13, 24, 32, 41),
            (2, 8, 17, 25, 34, 42),
            (3, 9, 18, 26, 35, 43),
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

    values.update(changes)

    return product.DurablePredictionEvaluationSource(
        **values
    )


def test_source_public_signature_is_exact() -> None:
    product = _product()

    signature = inspect.signature(
        product.DurablePredictionEvaluationSource
    )

    assert tuple(
        signature.parameters
    ) == (
        "schema_version",
        "round_no",
        "top_k",
        "selected_sets",
        "generated_at_kst",
    )


def test_source_is_immutable() -> None:
    source = _source()

    with pytest.raises(
        FrozenInstanceError
    ):
        source.round_no = 9999


def test_source_normalizes_selected_sets_to_tuples() -> None:
    source = _source(
        selected_sets=[
            [41, 1, 32, 7, 24, 13],
            [42, 2, 34, 8, 25, 17],
            [43, 3, 35, 9, 26, 18],
        ]
    )

    assert source.selected_sets == (
        (1, 7, 13, 24, 32, 41),
        (2, 8, 17, 25, 34, 42),
        (3, 9, 18, 26, 35, 43),
    )


def test_source_rejects_unsupported_schema_version() -> None:
    with pytest.raises(
        ContractError
    ):
        _source(
            schema_version="2.0"
        )


def test_source_rejects_invalid_round() -> None:
    for value in (
        0,
        -1,
        True,
        1.5,
        "1233",
    ):
        with pytest.raises(
            ContractError
        ):
            _source(
                round_no=value
            )


def test_source_rejects_invalid_top_k() -> None:
    for value in (
        0,
        -1,
        True,
        1.5,
        "3",
    ):
        with pytest.raises(
            ContractError
        ):
            _source(
                top_k=value
            )


def test_source_rejects_set_count_mismatch() -> None:
    with pytest.raises(
        ContractError
    ):
        _source(
            top_k=3,
            selected_sets=(
                (1, 7, 13, 24, 32, 41),
                (2, 8, 17, 25, 34, 42),
            ),
        )


def test_source_rejects_invalid_set_shape() -> None:
    invalid_sets = (
        (1, 7, 13, 24, 32),
        (1, 7, 13, 24, 32, 41, 44),
        (1, 7, 13, 24, 32, 46),
        (1, 7, 13, 24, 32, True),
        (1, 7, 13, 24, 32, 32),
    )

    for numbers in invalid_sets:
        with pytest.raises(
            ContractError
        ):
            _source(
                top_k=1,
                selected_sets=(
                    numbers,
                ),
            )


def test_source_rejects_duplicate_selected_sets() -> None:
    numbers = (
        1,
        7,
        13,
        24,
        32,
        41,
    )

    with pytest.raises(
        ContractError
    ):
        _source(
            top_k=2,
            selected_sets=(
                numbers,
                numbers,
            ),
        )


def test_source_rejects_naive_generated_at_kst() -> None:
    with pytest.raises(
        ContractError
    ):
        _source(
            generated_at_kst=datetime(
                2026,
                8,
                21,
                17,
                0,
            )
        )


def test_source_rejects_non_kst_offset_datetime() -> None:
    with pytest.raises(
        ContractError
    ):
        _source(
            generated_at_kst=datetime(
                2026,
                8,
                21,
                8,
                0,
                tzinfo=timezone.utc,
            )
        )


def test_source_accepts_kst_offset_datetime() -> None:
    source = _source()

    assert (
        source.generated_at_kst.utcoffset()
        == timedelta(hours=9)
    )


def test_source_dict_round_trip_is_exact() -> None:
    product = _product()

    source = _source()

    payload = product.source_to_dict(
        source
    )

    restored = product.source_from_dict(
        payload
    )

    assert restored == source


def test_source_json_round_trip_is_exact() -> None:
    product = _product()

    source = _source()

    payload = product.source_to_json(
        source
    )

    restored = product.source_from_json(
        payload
    )

    assert restored == source


def test_source_json_is_deterministic() -> None:
    product = _product()

    source = _source()

    values = tuple(
        product.source_to_json(
            source
        )
        for _ in range(5)
    )

    assert len(
        set(values)
    ) == 1


def test_source_from_dict_rejects_unknown_key() -> None:
    product = _product()

    payload = product.source_to_dict(
        _source()
    )

    payload[
        "unexpected"
    ] = True

    with pytest.raises(
        ContractError
    ):
        product.source_from_dict(
            payload
        )


def test_source_from_dict_rejects_missing_key() -> None:
    product = _product()

    payload = product.source_to_dict(
        _source()
    )

    payload.pop(
        "top_k"
    )

    with pytest.raises(
        ContractError
    ):
        product.source_from_dict(
            payload
        )


def test_source_from_json_rejects_malformed_payload() -> None:
    product = _product()

    for payload in (
        "",
        "[]",
        "null",
        "123",
        "{",
    ):
        with pytest.raises(
            ContractError
        ):
            product.source_from_json(
                payload
            )


def test_source_product_has_no_filesystem_dependency() -> None:
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
        ):
            if node.module:
                imports.append(
                    node.module
                )

    forbidden = {
        "os",
        "pathlib",
        "subprocess",
        "sqlite3",
    }

    assert not (
        forbidden
        & {
            name.split(
                ".",
                1,
            )[0]
            for name in imports
        }
    )


def test_source_product_has_no_predictionresult_dependency() -> None:
    path = Path(
        "lrp/pipelines/"
        "durable_prediction_evaluation_source.py"
    )

    source = path.read_text(
        encoding="utf-8-sig"
    )

    assert (
        "PredictionResult"
        not in source
    )


def test_source_product_does_not_modify_prediction_serializer() -> None:
    product_path = Path(
        "lrp/pipelines/"
        "durable_prediction_evaluation_source.py"
    )

    source = product_path.read_text(
        encoding="utf-8-sig"
    )

    assert (
        "prediction_to_dict"
        not in source
    )

    assert (
        "prediction_to_json"
        not in source
    )