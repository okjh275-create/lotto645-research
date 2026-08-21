from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from lrp.contracts.exceptions import ContractError


MODULE = (
    "lrp.evaluation."
    "topk_live_evaluation_source_snapshot"
)


def _product():
    module = __import__(
        MODULE,
        fromlist=["*"],
    )
    return module


def _valid_snapshot_kwargs() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "round_no": 1233,
        "top_k": 3,
        "selected_sets": (
            (1, 7, 13, 24, 32, 41),
            (3, 9, 18, 27, 35, 44),
            (5, 11, 20, 28, 37, 45),
        ),
        "model_name": "candidate-v1",
        "history_rounds": (
            1228,
            1229,
            1230,
            1231,
            1232,
        ),
        "regime_id": "regime-a",
        "strategy_name": "strategy-a",
        "generated_at_kst": (
            datetime.fromisoformat(
                "2026-08-21T17:00:00+09:00"
            )
        ),
        "source_artifact_sha256": "a" * 64,
    }


def _snapshot(**changes):
    product = _product()
    values = _valid_snapshot_kwargs()
    values.update(changes)

    return product.TopKLiveEvaluationSourceSnapshot(
        **values
    )


# ================================================================
# MODELS
# ================================================================


def test_snapshot_public_signature_is_exact() -> None:
    import inspect

    product = _product()

    assert tuple(
        inspect.signature(
            product.TopKLiveEvaluationSourceSnapshot
        ).parameters
    ) == (
        "schema_version",
        "round_no",
        "top_k",
        "selected_sets",
        "model_name",
        "history_rounds",
        "regime_id",
        "strategy_name",
        "generated_at_kst",
        "source_artifact_sha256",
    )


def test_source_pair_public_signature_is_exact() -> None:
    import inspect

    product = _product()

    assert tuple(
        inspect.signature(
            product.TopKLiveEvaluationSourcePair
        ).parameters
    ) == (
        "candidate",
        "baseline",
    )


def test_snapshot_normalizes_collections_to_tuples() -> None:
    snapshot = _snapshot(
        selected_sets=[
            [1, 7, 13, 24, 32, 41],
            [3, 9, 18, 27, 35, 44],
            [5, 11, 20, 28, 37, 45],
        ],
        history_rounds=[
            1228,
            1229,
            1230,
            1231,
            1232,
        ],
    )

    assert isinstance(
        snapshot.selected_sets,
        tuple,
    )

    assert all(
        isinstance(row, tuple)
        for row in snapshot.selected_sets
    )

    assert isinstance(
        snapshot.history_rounds,
        tuple,
    )


def test_snapshot_is_immutable() -> None:
    snapshot = _snapshot()

    with pytest.raises(
        (AttributeError, TypeError),
    ):
        snapshot.model_name = "mutated"


# ================================================================
# SERIALIZATION
# ================================================================


def test_snapshot_dict_round_trip_is_exact() -> None:
    product = _product()

    original = _snapshot()

    payload = product.snapshot_to_dict(
        original
    )

    restored = product.snapshot_from_dict(
        payload
    )

    assert restored == original


def test_snapshot_json_round_trip_is_exact() -> None:
    product = _product()

    original = _snapshot()

    encoded = product.snapshot_to_json(
        original
    )

    restored = product.snapshot_from_json(
        encoded
    )

    assert restored == original


def test_snapshot_json_is_deterministic() -> None:
    product = _product()

    snapshot = _snapshot()

    assert (
        product.snapshot_to_json(snapshot)
        == product.snapshot_to_json(snapshot)
    )


# ================================================================
# SNAPSHOT SAFETY
# ================================================================


@pytest.mark.parametrize(
    "round_no",
    [
        True,
        0,
        -1,
        "1233",
    ],
)
def test_snapshot_rejects_invalid_round(
    round_no,
) -> None:
    with pytest.raises(ContractError):
        _snapshot(
            round_no=round_no,
        )


@pytest.mark.parametrize(
    "top_k",
    [
        True,
        0,
        -1,
        "3",
    ],
)
def test_snapshot_rejects_invalid_top_k(
    top_k,
) -> None:
    with pytest.raises(ContractError):
        _snapshot(
            top_k=top_k,
        )


@pytest.mark.parametrize(
    "model_name",
    [
        "",
        " ",
        None,
    ],
)
def test_snapshot_rejects_invalid_model_name(
    model_name,
) -> None:
    with pytest.raises(ContractError):
        _snapshot(
            model_name=model_name,
        )


def test_snapshot_rejects_set_count_mismatch() -> None:
    with pytest.raises(ContractError):
        _snapshot(
            top_k=3,
            selected_sets=(
                (1, 7, 13, 24, 32, 41),
                (3, 9, 18, 27, 35, 44),
            ),
        )


def test_snapshot_rejects_duplicate_selected_sets() -> None:
    row = (
        1,
        7,
        13,
        24,
        32,
        41,
    )

    with pytest.raises(ContractError):
        _snapshot(
            selected_sets=(
                row,
                row,
                (
                    5,
                    11,
                    20,
                    28,
                    37,
                    45,
                ),
            ),
        )


def test_snapshot_rejects_target_round_in_history() -> None:
    with pytest.raises(ContractError):
        _snapshot(
            history_rounds=(
                1231,
                1232,
                1233,
            ),
        )


def test_snapshot_rejects_future_round_in_history() -> None:
    with pytest.raises(ContractError):
        _snapshot(
            history_rounds=(
                1231,
                1232,
                1234,
            ),
        )


# ================================================================
# SOURCE PAIR SAFETY
# ================================================================


def test_source_pair_accepts_same_round_distinct_models() -> None:
    product = _product()

    candidate = _snapshot(
        model_name="candidate-v1",
    )

    baseline = _snapshot(
        model_name="baseline-v1",
    )

    pair = product.TopKLiveEvaluationSourcePair(
        candidate=candidate,
        baseline=baseline,
    )

    assert pair.candidate is candidate
    assert pair.baseline is baseline


def test_source_pair_rejects_round_mismatch() -> None:
    product = _product()

    candidate = _snapshot(
        round_no=1233,
        model_name="candidate-v1",
    )

    baseline = _snapshot(
        round_no=1232,
        model_name="baseline-v1",
        history_rounds=(
            1227,
            1228,
            1229,
            1230,
            1231,
        ),
    )

    with pytest.raises(ContractError):
        product.TopKLiveEvaluationSourcePair(
            candidate=candidate,
            baseline=baseline,
        )


def test_source_pair_rejects_same_model_identity() -> None:
    product = _product()

    candidate = _snapshot(
        model_name="same-model",
    )

    baseline = _snapshot(
        model_name="same-model",
    )

    with pytest.raises(ContractError):
        product.TopKLiveEvaluationSourcePair(
            candidate=candidate,
            baseline=baseline,
        )


# ================================================================
# ARCHITECTURE / SCOPE
# ================================================================


def test_snapshot_product_has_no_filesystem_dependency() -> None:
    product = _product()

    path = Path(product.__file__)
    source = path.read_text(
        encoding="utf-8-sig"
    )

    forbidden = (
        "open(",
        "write_text(",
        "write_bytes(",
        "read_text(",
        "read_bytes(",
        "Path(",
        "os.",
        "subprocess",
    )

    assert not any(
        token in source
        for token in forbidden
    )


def test_snapshot_product_does_not_import_round_completion() -> None:
    product = _product()

    path = Path(product.__file__)
    source = path.read_text(
        encoding="utf-8-sig"
    )

    assert (
        "round_completion"
        not in source
    )


def test_snapshot_product_does_not_modify_prediction_serializer() -> None:
    serializer = Path(
        "lrp/pipelines/serializer.py"
    ).read_text(
        encoding="utf-8-sig"
    )

    assert (
        "TopKLiveEvaluationSourceSnapshot"
        not in serializer
    )
