from __future__ import annotations

from datetime import datetime

import pytest

from lrp.contracts.exceptions import ContractError
from lrp.evaluation.topk_live_evaluation_source_snapshot import (
    TopKLiveEvaluationSourceSnapshot,
    snapshot_from_dict,
    snapshot_to_dict,
)


def _snapshot(**changes):
    values = {
        "schema_version": "1.0",
        "round_no": 1233,
        "top_k": 2,
        "selected_sets": (
            (1, 7, 13, 24, 32, 41),
            (3, 9, 18, 27, 35, 44),
        ),
        "model_name": "candidate-v1",
        "history_rounds": (
            1230,
            1231,
            1232,
        ),
        "regime_id": None,
        "strategy_name": None,
        "generated_at_kst": (
            datetime.fromisoformat(
                "2026-08-21T17:00:00+09:00"
            )
        ),
        "source_artifact_sha256": "a" * 64,
    }

    values.update(changes)

    return TopKLiveEvaluationSourceSnapshot(
        **values
    )


# ================================================================
# SCHEMA VERSION
# ================================================================


@pytest.mark.parametrize(
    "schema_version",
    (
        "1",
        "v1",
        "999.999",
        "1.0.0",
    ),
)
def test_snapshot_rejects_unsupported_schema_version(
    schema_version,
) -> None:
    with pytest.raises(ContractError):
        _snapshot(
            schema_version=schema_version,
        )


def test_snapshot_accepts_supported_schema_version() -> None:
    snapshot = _snapshot(
        schema_version="1.0",
    )

    assert snapshot.schema_version == "1.0"


# ================================================================
# SHA-256 IDENTITY
# ================================================================


@pytest.mark.parametrize(
    "value",
    (
        "abc",
        "a" * 63,
        "a" * 65,
        "g" * 64,
        "A" * 64,
    ),
)
def test_snapshot_rejects_noncanonical_sha256(
    value,
) -> None:
    with pytest.raises(ContractError):
        _snapshot(
            source_artifact_sha256=value,
        )


def test_snapshot_accepts_lowercase_sha256() -> None:
    value = (
        "0123456789abcdef"
        "0123456789abcdef"
        "0123456789abcdef"
        "0123456789abcdef"
    )

    snapshot = _snapshot(
        source_artifact_sha256=value,
    )

    assert (
        snapshot.source_artifact_sha256
        == value
    )


# ================================================================
# KST TIMESTAMP
# ================================================================


def test_snapshot_rejects_naive_generated_at_kst() -> None:
    with pytest.raises(ContractError):
        _snapshot(
            generated_at_kst=datetime(
                2026,
                8,
                21,
                17,
                0,
                0,
            ),
        )


def test_snapshot_rejects_non_kst_offset_datetime() -> None:
    with pytest.raises(ContractError):
        _snapshot(
            generated_at_kst=(
                datetime.fromisoformat(
                    "2026-08-21T08:00:00+00:00"
                )
            ),
        )


def test_snapshot_accepts_kst_offset_datetime() -> None:
    value = datetime.fromisoformat(
        "2026-08-21T17:00:00+09:00"
    )

    snapshot = _snapshot(
        generated_at_kst=value,
    )

    assert (
        snapshot.generated_at_kst
        == value
    )


# ================================================================
# HISTORY CANONICALIZATION
# ================================================================


def test_snapshot_rejects_empty_history() -> None:
    with pytest.raises(ContractError):
        _snapshot(
            history_rounds=(),
        )


def test_snapshot_rejects_reverse_history() -> None:
    with pytest.raises(ContractError):
        _snapshot(
            history_rounds=(
                1232,
                1231,
                1230,
            ),
        )


def test_snapshot_accepts_strictly_increasing_history() -> None:
    history = (
        1229,
        1230,
        1231,
        1232,
    )

    snapshot = _snapshot(
        history_rounds=history,
    )

    assert (
        snapshot.history_rounds
        == history
    )


# ================================================================
# SELECTED SET CANONICALIZATION
# ================================================================


def test_snapshot_sorts_numbers_inside_each_set() -> None:
    snapshot = _snapshot(
        selected_sets=(
            (
                41,
                1,
                32,
                7,
                24,
                13,
            ),
            (
                44,
                35,
                27,
                18,
                9,
                3,
            ),
        ),
    )

    assert snapshot.selected_sets == (
        (
            1,
            7,
            13,
            24,
            32,
            41,
        ),
        (
            3,
            9,
            18,
            27,
            35,
            44,
        ),
    )


def test_snapshot_preserves_topk_set_order() -> None:
    snapshot = _snapshot(
        selected_sets=(
            (
                44,
                35,
                27,
                18,
                9,
                3,
            ),
            (
                41,
                1,
                32,
                7,
                24,
                13,
            ),
        ),
    )

    assert snapshot.selected_sets == (
        (
            3,
            9,
            18,
            27,
            35,
            44,
        ),
        (
            1,
            7,
            13,
            24,
            32,
            41,
        ),
    )


# ================================================================
# STRICT DICT SCHEMA
# ================================================================


def test_snapshot_from_dict_rejects_unknown_key() -> None:
    payload = snapshot_to_dict(
        _snapshot()
    )

    payload["unexpected"] = 123

    with pytest.raises(ContractError):
        snapshot_from_dict(
            payload
        )


def test_snapshot_from_dict_accepts_exact_schema() -> None:
    original = _snapshot()

    payload = snapshot_to_dict(
        original
    )

    restored = snapshot_from_dict(
        payload
    )

    assert restored == original
