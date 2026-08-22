from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import pytest

from lrp.contracts.exceptions import ContractError
from lrp.evaluation.topk_replay_adapter import TopKReplayPrediction
from lrp.pipelines.durable_prediction_evaluation_source import (
    DurablePredictionEvaluationSource,
    source_to_json,
)


PRODUCT_MODULE = "lrp.operations.durable_replay_consumer"

ROOT = Path(__file__).resolve().parents[2]
PRODUCT_PATH = ROOT / "lrp" / "operations" / "durable_replay_consumer.py"


def _product():
    return importlib.import_module(
        PRODUCT_MODULE
    )


def _selected_sets() -> tuple[tuple[int, ...], ...]:
    return (
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
    )


def _source() -> DurablePredictionEvaluationSource:
    from datetime import datetime, timedelta, timezone

    return DurablePredictionEvaluationSource(
        schema_version="1.0",
        round_no=1233,
        top_k=10,
        selected_sets=_selected_sets(),
        generated_at_kst=datetime(
            2026,
            8,
            21,
            17,
            0,
            tzinfo=timezone(
                timedelta(
                    hours=9
                )
            ),
        ),
    )


def _write_source(
    tmp_path: Path,
) -> Path:
    path = tmp_path / "explicit-source.json"

    path.write_text(
        source_to_json(
            _source()
        ),
        encoding="utf-8",
    )

    return path


def _load(
    tmp_path: Path,
    *,
    artifact_path: str | Path | None = None,
    history_rounds: tuple[int, ...] = (
        1230,
        1231,
        1232,
    ),
    model_name: str = "model-A",
    regime_id: str | None = None,
    strategy_name: str | None = None,
):
    product = _product()

    consumer = (
        product.DurableReplayOperationalConsumer()
    )

    resolved_path = (
        artifact_path
        if artifact_path is not None
        else _write_source(
            tmp_path
        )
    )

    return consumer.load(
        artifact_path=resolved_path,
        history_rounds=history_rounds,
        model_name=model_name,
        regime_id=regime_id,
        strategy_name=strategy_name,
    )


def test_consumer_is_parameterless() -> None:
    product = _product()

    signature = inspect.signature(
        product.DurableReplayOperationalConsumer
    )

    assert signature == inspect.Signature()


def test_load_public_signature_is_exact() -> None:
    product = _product()

    signature = inspect.signature(
        product.DurableReplayOperationalConsumer.load
    )

    parameters = tuple(
        signature.parameters.values()
    )

    assert tuple(
        item.name
        for item in parameters
    ) == (
        "self",
        "artifact_path",
        "history_rounds",
        "model_name",
        "regime_id",
        "strategy_name",
    )

    assert (
        parameters[0].kind
        is inspect.Parameter.POSITIONAL_OR_KEYWORD
    )

    for item in parameters[1:]:
        assert (
            item.kind
            is inspect.Parameter.KEYWORD_ONLY
        )

    assert parameters[4].default is None
    assert parameters[5].default is None

    assert (
        str(
            signature.return_annotation
        )
        == "TopKReplayPrediction"
    )


def test_load_returns_topk_replay_prediction(
    tmp_path: Path,
) -> None:
    result = _load(
        tmp_path
    )

    assert isinstance(
        result,
        TopKReplayPrediction,
    )


def test_load_reads_explicit_artifact_path(
    tmp_path: Path,
) -> None:
    path = _write_source(
        tmp_path
    )

    result = _load(
        tmp_path,
        artifact_path=path,
    )

    assert result.round_no == 1233


def test_load_decodes_with_source_from_json() -> None:
    source = PRODUCT_PATH.read_text(
        encoding="utf-8-sig"
    )

    assert "source_from_json" in source
    assert "json.loads" not in source
    assert "json.load" not in source


def test_load_delegates_to_durable_replay_adapter() -> None:
    source = PRODUCT_PATH.read_text(
        encoding="utf-8-sig"
    )

    assert "TopKDurableReplayAdapter" in source
    assert ".adapt(" in source


def test_load_preserves_round_no(
    tmp_path: Path,
) -> None:
    result = _load(
        tmp_path
    )

    assert result.round_no == 1233


def test_load_projects_selected_sets_to_predictions(
    tmp_path: Path,
) -> None:
    result = _load(
        tmp_path
    )

    assert (
        result.predictions
        == _selected_sets()
    )


def test_load_preserves_history_rounds(
    tmp_path: Path,
) -> None:
    history = (
        1228,
        1229,
        1230,
        1231,
        1232,
    )

    result = _load(
        tmp_path,
        history_rounds=history,
    )

    assert result.history_rounds == history


def test_load_preserves_model_name(
    tmp_path: Path,
) -> None:
    result = _load(
        tmp_path,
        model_name="model-X",
    )

    assert result.model_name == "model-X"


def test_load_preserves_optional_regime_id(
    tmp_path: Path,
) -> None:
    result = _load(
        tmp_path,
        regime_id="regime-A",
    )

    assert result.regime_id == "regime-A"


def test_load_preserves_optional_strategy_name(
    tmp_path: Path,
) -> None:
    result = _load(
        tmp_path,
        strategy_name="strategy-A",
    )

    assert result.strategy_name == "strategy-A"


def test_load_preserves_none_provenance(
    tmp_path: Path,
) -> None:
    result = _load(
        tmp_path
    )

    assert result.regime_id is None
    assert result.strategy_name is None


@pytest.mark.parametrize(
    "value",
    (
        None,
        1,
        True,
        object(),
        [],
        {},
        (),
    ),
)
def test_load_rejects_invalid_artifact_path_type(
    tmp_path: Path,
    value,
) -> None:
    product = _product()

    consumer = (
        product.DurableReplayOperationalConsumer()
    )

    with pytest.raises(
        ContractError
    ):
        consumer.load(
            artifact_path=value,
            history_rounds=(
                1230,
                1231,
                1232,
            ),
            model_name="model-A",
        )


def test_load_propagates_missing_file_failure(
    tmp_path: Path,
) -> None:
    missing = (
        tmp_path
        / "missing-source.json"
    )

    with pytest.raises(
        FileNotFoundError
    ):
        _load(
            tmp_path,
            artifact_path=missing,
        )


def test_load_propagates_codec_contract_error(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "malformed.json"
    )

    path.write_text(
        "{",
        encoding="utf-8",
    )

    with pytest.raises(
        ContractError
    ):
        _load(
            tmp_path,
            artifact_path=path,
        )


def test_load_propagates_adapter_contract_error(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ContractError
    ):
        _load(
            tmp_path,
            history_rounds=(),
        )


def test_load_does_not_mutate_history_rounds(
    tmp_path: Path,
) -> None:
    history = (
        1230,
        1231,
        1232,
    )

    before = tuple(
        history
    )

    _load(
        tmp_path,
        history_rounds=history,
    )

    assert history == before


def test_load_is_semantically_deterministic(
    tmp_path: Path,
) -> None:
    path = _write_source(
        tmp_path
    )

    first = _load(
        tmp_path,
        artifact_path=path,
    )

    second = _load(
        tmp_path,
        artifact_path=path,
    )

    assert first == second


def test_product_has_no_filesystem_write_dependency() -> None:
    source = PRODUCT_PATH.read_text(
        encoding="utf-8-sig"
    )

    forbidden = (
        "write_text(",
        "write_bytes(",
        "mkdir(",
        "atomic_write",
        "write_operation_artifact",
        "write_prediction_artifacts",
        "append_operation_log",
    )

    assert all(
        token not in source
        for token in forbidden
    )


def test_product_has_no_replay_execution_dependency() -> None:
    source = PRODUCT_PATH.read_text(
        encoding="utf-8-sig"
    )

    forbidden = (
        "TopKReplayEvaluationService",
        "TopKReplayEvaluationRequest",
        "EvaluationWindow",
        ".evaluate(",
    )

    assert all(
        token not in source
        for token in forbidden
    )


def test_product_has_no_predictionresult_dependency() -> None:
    source = PRODUCT_PATH.read_text(
        encoding="utf-8-sig"
    )

    assert "PredictionResult" not in source


def test_product_does_not_derive_operation_artifact_path() -> None:
    source = PRODUCT_PATH.read_text(
        encoding="utf-8-sig"
    )

    forbidden = (
        "prediction-evaluation-sources",
        "evaluation_source.json",
        "round_",
        "output_root",
    )

    assert all(
        token not in source
        for token in forbidden
    )