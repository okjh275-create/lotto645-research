from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import ast

import pytest

from lrp.contracts.exceptions import ContractError
from lrp.operations.durable_replay_consumer import (
    DurableReplayOperationalConsumer,
)
from lrp.pipelines.durable_prediction_evaluation_source import (
    DurablePredictionEvaluationSource,
    source_to_json,
)


PRODUCT_PATH = (
    Path(__file__).resolve().parents[2]
    / "lrp"
    / "operations"
    / "durable_replay_consumer.py"
)


def _sets() -> tuple[tuple[int, ...], ...]:
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


def _payload() -> str:
    source = DurablePredictionEvaluationSource(
        schema_version="1.0",
        round_no=1233,
        top_k=10,
        selected_sets=_sets(),
        generated_at_kst=datetime(
            2026,
            8,
            21,
            17,
            0,
            tzinfo=timezone(
                timedelta(hours=9)
            ),
        ),
    )

    return source_to_json(source)


def _write(
    tmp_path: Path,
    *,
    text: str | None = None,
) -> Path:
    path = tmp_path / "source.json"

    path.write_text(
        _payload() if text is None else text,
        encoding="utf-8",
    )

    return path


def _load(
    artifact_path: str | Path,
):
    return DurableReplayOperationalConsumer().load(
        artifact_path=artifact_path,
        history_rounds=(1230, 1231, 1232),
        model_name="model-A",
    )


def test_str_and_path_inputs_are_semantically_equivalent(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path)

    from_path = _load(path)
    from_str = _load(str(path))

    assert from_path == from_str


def test_directory_path_failure_propagates(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "directory"
    directory.mkdir()

    with pytest.raises(
        (PermissionError, IsADirectoryError)
    ):
        _load(directory)


@pytest.mark.parametrize(
    "payload",
    (
        "",
        " ",
        "\n",
        "\t",
        " \n\t ",
    ),
)
def test_empty_or_whitespace_source_fails_closed(
    tmp_path: Path,
    payload: str,
) -> None:
    path = _write(
        tmp_path,
        text=payload,
    )

    with pytest.raises(
        ContractError
    ):
        _load(path)


def test_invalid_utf8_propagates_unicode_decode_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid.json"

    path.write_bytes(
        b"\xff\xfe\xfa"
    )

    with pytest.raises(
        UnicodeDecodeError
    ):
        _load(path)


@pytest.mark.parametrize(
    "payload",
    (
        "{",
        "[",
        '{"schema_version":',
    ),
)
def test_malformed_json_fails_closed(
    tmp_path: Path,
    payload: str,
) -> None:
    path = _write(
        tmp_path,
        text=payload,
    )

    with pytest.raises(
        ContractError
    ):
        _load(path)


@pytest.mark.parametrize(
    "payload",
    (
        "[]",
        "null",
        "1",
        '"text"',
        "true",
    ),
)
def test_non_object_json_fails_closed(
    tmp_path: Path,
    payload: str,
) -> None:
    path = _write(
        tmp_path,
        text=payload,
    )

    with pytest.raises(
        ContractError
    ):
        _load(path)


def test_utf8_bom_source_is_rejected_by_current_contract(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bom-source.json"

    path.write_bytes(
        b"\xef\xbb\xbf"
        + _payload().encode("utf-8")
    )

    with pytest.raises(
        ContractError
    ):
        _load(path)


def test_symlink_reads_target_when_platform_supports_it(
    tmp_path: Path,
) -> None:
    target = _write(tmp_path)

    link = tmp_path / "source-link.json"

    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip(
            "symlink creation unavailable on this platform"
        )

    result = _load(link)

    assert result.round_no == 1233
    assert result.predictions == _sets()


def test_consumer_performs_exactly_one_read_text_call() -> None:
    source = PRODUCT_PATH.read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(source)

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "read_text"
    ]

    assert len(calls) == 1

    encodings = [
        ast.literal_eval(keyword.value)
        for keyword in calls[0].keywords
        if keyword.arg == "encoding"
    ]

    assert encodings == ["utf-8"]


def test_consumer_has_no_exception_normalization_layer() -> None:
    source = PRODUCT_PATH.read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(source)

    handlers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler)
    ]

    assert handlers == []


def test_consumer_has_exact_single_owned_raise() -> None:
    source = PRODUCT_PATH.read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(source)

    raises = [
        ast.unparse(node.exc)
        for node in ast.walk(tree)
        if isinstance(node, ast.Raise)
        and node.exc is not None
    ]

    assert len(raises) == 1
    assert (
        "artifact_path must be str or Path"
        in raises[0]
    )


def test_consumer_has_no_path_derivation_contract() -> None:
    source = PRODUCT_PATH.read_text(
        encoding="utf-8-sig"
    )

    forbidden = (
        "evaluation_source.json",
        "prediction-evaluation-sources",
        "output_root",
        "artifact_type",
        "round_",
    )

    assert all(
        token not in source
        for token in forbidden
    )


def test_consumer_has_no_filesystem_write_contract() -> None:
    source = PRODUCT_PATH.read_text(
        encoding="utf-8-sig"
    )

    forbidden = (
        "write_text(",
        "write_bytes(",
        "mkdir(",
        "unlink(",
        "rename(",
        "replace(",
        "write_operation_artifact",
        "write_prediction_artifacts",
    )

    assert all(
        token not in source
        for token in forbidden
    )


def test_consumer_has_no_replay_execution_contract() -> None:
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


def test_consumer_has_no_direct_json_decoder() -> None:
    source = PRODUCT_PATH.read_text(
        encoding="utf-8-sig"
    )

    assert "json.loads" not in source
    assert "json.load" not in source


def test_consumer_delegates_exactly_once_to_source_codec() -> None:
    source = PRODUCT_PATH.read_text(
        encoding="utf-8-sig"
    )

    assert (
        source.count(
            "source_from_json("
        )
        == 1
    )


def test_consumer_delegates_exactly_once_to_durable_adapter() -> None:
    source = PRODUCT_PATH.read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(source)

    adapter_builds = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        == "TopKDurableReplayAdapter"
    ]

    adapt_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "adapt"
    ]

    assert len(adapter_builds) == 1
    assert len(adapt_calls) == 1


def test_repeated_load_is_semantically_stable(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path)

    results = tuple(
        _load(path)
        for _ in range(5)
    )

    assert len(set(results)) == 1


def test_source_file_is_not_mutated(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path)

    before = path.read_bytes()

    _load(path)

    after = path.read_bytes()

    assert before == after


def test_product_static_dependency_boundary_is_exact() -> None:
    source = PRODUCT_PATH.read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(source)

    imports: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(
                alias.name
                for alias in node.names
            )

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)

    allowed = {
        "__future__",
        "pathlib",
        "lrp.contracts.exceptions",
        "lrp.evaluation.topk_durable_replay_adapter",
        "lrp.evaluation.topk_replay_adapter",
        "lrp.pipelines.durable_prediction_evaluation_source",
    }

    assert set(imports) == allowed


def test_product_public_surface_remains_minimal() -> None:
    source = PRODUCT_PATH.read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(source)

    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    ]

    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    ]

    assert len(classes) == 1
    assert (
        classes[0].name
        == "DurableReplayOperationalConsumer"
    )

    assert functions == []

    public_methods = [
        node.name
        for node in classes[0].body
        if isinstance(node, ast.FunctionDef)
        and not node.name.startswith("_")
    ]

    assert public_methods == ["load"]