from __future__ import annotations

from datetime import datetime

import pytest

from lrp.contracts.exceptions import ContractError
from lrp.io.draws import HistoryRow
from lrp.pipelines.models import (
    PredictionGenerationResult,
    PredictionRequest,
    PredictionResult,
)


MODULE = (
    "lrp.evaluation."
    "topk_live_evaluation_snapshot_factory"
)


class _SelectedItem:
    def __init__(
        self,
        numbers: tuple[int, ...],
    ) -> None:
        self.numbers = numbers


class _Diversity:
    def __init__(
        self,
        selected: tuple[_SelectedItem, ...],
    ) -> None:
        self.selected = selected


def _product():
    return __import__(
        MODULE,
        fromlist=["*"],
    )


def _prediction_result(
    *,
    round_no: int = 1233,
    top_k: int = 2,
) -> PredictionResult:

    request = PredictionRequest(
        round_no=round_no,
        seed=20260821,
        top_k=top_k,
        practical_k=1,
        long_gap_numbers=frozenset({45}),
    )

    generation = PredictionGenerationResult(
        request=request,
        windows=(10, 20, 50),
        probabilities={},
        statistics_contract=object(),
        number_signals={},
        candidates=(),
        statistics_version="stats-v1",
        candidate_version="candidate-v1",
    )

    diversity = _Diversity(
        selected=(
            _SelectedItem(
                (41, 1, 32, 7, 24, 13),
            ),
            _SelectedItem(
                (44, 35, 27, 18, 9, 3),
            ),
        )[:top_k],
    )

    return PredictionResult(
        generation=generation,
        scored_candidates=(),
        ranking=object(),
        diversity=diversity,
        practical=object(),
        generated_at_kst=(
            datetime.fromisoformat(
                "2026-08-21T17:00:00+09:00"
            )
        ),
    )


def _history_rows() -> tuple[HistoryRow, ...]:
    return (
        HistoryRow(
            round_no=1230,
            numbers=(1, 2, 3, 4, 5, 6),
            bonus=7,
        ),
        HistoryRow(
            round_no=1231,
            numbers=(8, 9, 10, 11, 12, 13),
            bonus=14,
        ),
        HistoryRow(
            round_no=1232,
            numbers=(15, 16, 17, 18, 19, 20),
            bonus=21,
        ),
    )


def _request(**changes):
    product = _product()

    values = {
        "prediction_result": _prediction_result(),
        "history_rows": _history_rows(),
        "model_name": "candidate-v1",
        "source_artifact_sha256": "a" * 64,
        "regime_id": "regime-a",
        "strategy_name": "strategy-a",
    }

    values.update(changes)

    return product.TopKLiveEvaluationSnapshotBuildRequest(
        **values
    )


# ================================================================
# PUBLIC MODELS
# ================================================================


def test_build_request_public_signature_is_exact() -> None:
    import inspect

    product = _product()

    assert tuple(
        inspect.signature(
            product.TopKLiveEvaluationSnapshotBuildRequest
        ).parameters
    ) == (
        "prediction_result",
        "history_rows",
        "model_name",
        "source_artifact_sha256",
        "regime_id",
        "strategy_name",
    )


def test_factory_is_parameterless() -> None:
    import inspect

    product = _product()

    assert tuple(
        inspect.signature(
            product.TopKLiveEvaluationSnapshotFactory
        ).parameters
    ) == ()


def test_factory_build_signature_is_exact() -> None:
    import inspect

    product = _product()

    signature = inspect.signature(
        product.TopKLiveEvaluationSnapshotFactory.build
    )

    assert tuple(
        signature.parameters
    ) == (
        "self",
        "request",
    )

    assert (
        signature.parameters["request"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )


# ================================================================
# PROJECTION SEMANTICS
# ================================================================


def test_factory_projects_prediction_result_to_snapshot() -> None:
    product = _product()

    result = (
        product.TopKLiveEvaluationSnapshotFactory()
        .build(
            request=_request()
        )
    )

    assert result.schema_version == "1.0"
    assert result.round_no == 1233
    assert result.top_k == 2

    assert result.selected_sets == (
        (1, 7, 13, 24, 32, 41),
        (3, 9, 18, 27, 35, 44),
    )

    assert result.model_name == "candidate-v1"

    assert result.history_rounds == (
        1230,
        1231,
        1232,
    )

    assert result.regime_id == "regime-a"
    assert result.strategy_name == "strategy-a"

    assert result.generated_at_kst == (
        datetime.fromisoformat(
            "2026-08-21T17:00:00+09:00"
        )
    )

    assert result.source_artifact_sha256 == (
        "a" * 64
    )


def test_factory_preserves_optional_none_provenance() -> None:
    product = _product()

    result = (
        product.TopKLiveEvaluationSnapshotFactory()
        .build(
            request=_request(
                regime_id=None,
                strategy_name=None,
            )
        )
    )

    assert result.regime_id is None
    assert result.strategy_name is None


def test_factory_is_semantically_deterministic() -> None:
    product = _product()

    request = _request()

    first = (
        product.TopKLiveEvaluationSnapshotFactory()
        .build(
            request=request
        )
    )

    second = (
        product.TopKLiveEvaluationSnapshotFactory()
        .build(
            request=request
        )
    )

    assert first == second


# ================================================================
# REQUEST SAFETY
# ================================================================


def test_request_rejects_invalid_prediction_result() -> None:
    with pytest.raises(ContractError):
        _request(
            prediction_result=object(),
        )


def test_request_rejects_empty_history() -> None:
    with pytest.raises(ContractError):
        _request(
            history_rows=(),
        )


def test_request_rejects_non_historyrow_item() -> None:
    with pytest.raises(ContractError):
        _request(
            history_rows=(
                object(),
            ),
        )


def test_request_rejects_target_round_in_history() -> None:
    rows = (
        *_history_rows(),
        HistoryRow(
            round_no=1233,
            numbers=(21, 22, 23, 24, 25, 26),
            bonus=27,
        ),
    )

    with pytest.raises(ContractError):
        _request(
            history_rows=rows,
        )


def test_request_rejects_future_round_in_history() -> None:
    rows = (
        *_history_rows(),
        HistoryRow(
            round_no=1234,
            numbers=(21, 22, 23, 24, 25, 26),
            bonus=27,
        ),
    )

    with pytest.raises(ContractError):
        _request(
            history_rows=rows,
        )


def test_request_rejects_duplicate_history_round() -> None:
    rows = (
        _history_rows()[0],
        _history_rows()[1],
        _history_rows()[1],
    )

    with pytest.raises(ContractError):
        _request(
            history_rows=rows,
        )


@pytest.mark.parametrize(
    "model_name",
    (
        "",
        " ",
        None,
    ),
)
def test_request_rejects_invalid_model_name(
    model_name,
) -> None:
    with pytest.raises(ContractError):
        _request(
            model_name=model_name,
        )


@pytest.mark.parametrize(
    "value",
    (
        "",
        "abc",
        "A" * 64,
        "g" * 64,
        "a" * 63,
        "a" * 65,
    ),
)
def test_request_rejects_invalid_source_artifact_sha256(
    value,
) -> None:
    with pytest.raises(ContractError):
        _request(
            source_artifact_sha256=value,
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "regime_id",
        "strategy_name",
    ),
)
def test_request_rejects_blank_optional_provenance(
    field_name,
) -> None:
    with pytest.raises(ContractError):
        _request(
            **{
                field_name: " ",
            }
        )


# ================================================================
# BUILD-TIME SOURCE SAFETY
# ================================================================


def test_factory_rejects_diversity_count_mismatch() -> None:
    product = _product()

    prediction = _prediction_result(
        top_k=2,
    )

    prediction.diversity.selected = (
        _SelectedItem(
            (1, 7, 13, 24, 32, 41),
        ),
    )

    with pytest.raises(ContractError):
        (
            product.TopKLiveEvaluationSnapshotFactory()
            .build(
                request=_request(
                    prediction_result=prediction,
                )
            )
        )


def test_factory_rejects_missing_selected_set_source() -> None:
    product = _product()

    valid = _prediction_result()

    prediction = PredictionResult(
        generation=valid.generation,
        scored_candidates=valid.scored_candidates,
        ranking=valid.ranking,
        diversity=object(),
        practical=valid.practical,
        generated_at_kst=valid.generated_at_kst,
        ensemble=valid.ensemble,
    )

    with pytest.raises(ContractError):
        (
            product.TopKLiveEvaluationSnapshotFactory()
            .build(
                request=_request(
                    prediction_result=prediction,
                )
            )
        )


# ================================================================
# ARCHITECTURE
# ================================================================


def test_factory_has_no_filesystem_dependency() -> None:
    from pathlib import Path

    product = _product()

    source = Path(
        product.__file__
    ).read_text(
        encoding="utf-8-sig"
    )

    forbidden = (
        "open(",
        "Path(",
        "read_text(",
        "read_bytes(",
        "write_text(",
        "write_bytes(",
        "sha256_file(",
        "write_prediction_artifacts(",
        "subprocess",
    )

    assert not any(
        token in source
        for token in forbidden
    )


def test_factory_does_not_modify_snapshot_core_contract() -> None:
    from pathlib import Path

    source = Path(
        "lrp/evaluation/"
        "topk_live_evaluation_source_snapshot.py"
    ).read_text(
        encoding="utf-8-sig"
    )

    assert (
        "TopKLiveEvaluationSnapshotBuildRequest"
        not in source
    )


def test_factory_does_not_import_round_completion() -> None:
    from pathlib import Path

    product = _product()

    source = Path(
        product.__file__
    ).read_text(
        encoding="utf-8-sig"
    )

    assert (
        "round_completion"
        not in source
    )
