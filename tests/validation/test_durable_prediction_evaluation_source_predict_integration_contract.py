from __future__ import annotations

import ast
from pathlib import Path


PREDICT_PATH = Path(
    "lrp/cli/predict.py"
)

CORE_PATH = Path(
    "lrp/pipelines/"
    "durable_prediction_evaluation_source.py"
)

PREDICTION_WRITER_PATH = Path(
    "lrp/io/artifacts.py"
)

OPERATION_WRITER_PATH = Path(
    "lrp/operations/runtime.py"
)

WEEKLY_PATH = Path(
    "lrp/cli/weekly.py"
)


def _source(
    path: Path,
) -> str:
    return path.read_text(
        encoding="utf-8-sig"
    )


def _tree(
    path: Path,
) -> ast.Module:
    return ast.parse(
        _source(path)
    )


def _run_predict() -> ast.FunctionDef:
    tree = _tree(
        PREDICT_PATH
    )

    return next(
        node
        for node in tree.body
        if isinstance(
            node,
            ast.FunctionDef,
        )
        and node.name
        == "run_predict"
    )


def _call_name(
    node: ast.Call,
) -> str:
    return ast.unparse(
        node.func
    )


def _calls(
    function: ast.FunctionDef,
) -> tuple[ast.Call, ...]:
    return tuple(
        node
        for node in ast.walk(
            function
        )
        if isinstance(
            node,
            ast.Call,
        )
    )


def test_predict_imports_durable_source_contract() -> None:
    source = _source(
        PREDICT_PATH
    )

    assert (
        "DurablePredictionEvaluationSource"
        in source
    )

    assert (
        "source_to_dict"
        in source
    )


def test_predict_imports_operation_artifact_writer() -> None:
    source = _source(
        PREDICT_PATH
    )

    assert (
        "write_operation_artifact"
        in source
    )


def test_run_predict_constructs_durable_source() -> None:
    function = _run_predict()

    names = tuple(
        _call_name(node)
        for node in _calls(
            function
        )
    )

    assert (
        "DurablePredictionEvaluationSource"
        in names
    )


def test_run_predict_projects_durable_source_to_dict() -> None:
    function = _run_predict()

    names = tuple(
        _call_name(node)
        for node in _calls(
            function
        )
    )

    assert (
        "source_to_dict"
        in names
    )


def test_run_predict_persists_durable_source_with_operation_writer() -> None:
    function = _run_predict()

    names = tuple(
        _call_name(node)
        for node in _calls(
            function
        )
    )

    assert (
        "write_operation_artifact"
        in names
    )


def test_run_predict_durable_source_uses_schema_version_1_0() -> None:
    function = _run_predict()

    constructors = tuple(
        node
        for node in _calls(
            function
        )
        if _call_name(node)
        == "DurablePredictionEvaluationSource"
    )

    assert len(
        constructors
    ) == 1

    constructor = constructors[0]

    keywords = {
        item.arg: item.value
        for item in constructor.keywords
        if item.arg is not None
    }

    assert (
        ast.literal_eval(
            keywords[
                "schema_version"
            ]
        )
        == "1.0"
    )


def test_run_predict_durable_source_uses_prediction_round() -> None:
    source = ast.unparse(
        _run_predict()
    )

    assert (
        "round_no="
        in source
    )

    assert (
        "arguments.round_no"
        in source
        or "request.round_no"
        in source
    )


def test_run_predict_durable_source_uses_request_top_k() -> None:
    source = ast.unparse(
        _run_predict()
    )

    assert (
        "top_k="
        in source
    )

    assert (
        "request.top_k"
        in source
        or "arguments.top_k"
        in source
    )


def test_run_predict_durable_source_uses_prediction_generated_at_kst() -> None:
    source = ast.unparse(
        _run_predict()
    )

    assert (
        "generated_at_kst="
        in source
    )

    assert (
        "result.generated_at_kst"
        in source
    )


def test_run_predict_durable_source_uses_serialized_selected_sets() -> None:
    source = ast.unparse(
        _run_predict()
    )

    assert (
        "selected_sets="
        in source
    )

    assert (
        "payload"
        in source
    )

    assert (
        "'sets'"
        in source
        or '"sets"'
        in source
    )

    assert (
        "'numbers'"
        in source
        or '"numbers"'
        in source
    )


def test_operation_artifact_identity_is_exact() -> None:
    function = _run_predict()

    calls = tuple(
        node
        for node in _calls(
            function
        )
        if _call_name(node)
        == "write_operation_artifact"
    )

    assert len(
        calls
    ) == 1

    call = calls[0]

    keywords = {
        item.arg: item.value
        for item in call.keywords
        if item.arg is not None
    }

    assert (
        ast.literal_eval(
            keywords[
                "artifact_type"
            ]
        )
        == "prediction-evaluation-sources"
    )

    assert (
        ast.literal_eval(
            keywords[
                "filename"
            ]
        )
        == "evaluation_source.json"
    )


def test_operation_artifact_uses_prediction_output_root() -> None:
    function = _run_predict()

    call = next(
        node
        for node in _calls(
            function
        )
        if _call_name(node)
        == "write_operation_artifact"
    )

    keywords = {
        item.arg: item.value
        for item in call.keywords
        if item.arg is not None
    }

    assert (
        ast.unparse(
            keywords[
                "output_root"
            ]
        )
        == "arguments.output"
    )


def test_operation_artifact_uses_prediction_round() -> None:
    function = _run_predict()

    call = next(
        node
        for node in _calls(
            function
        )
        if _call_name(node)
        == "write_operation_artifact"
    )

    keywords = {
        item.arg: item.value
        for item in call.keywords
        if item.arg is not None
    }

    round_expression = ast.unparse(
        keywords[
            "round_no"
        ]
    )

    assert round_expression in {
        "arguments.round_no",
        "request.round_no",
    }


def test_prediction_artifact_is_written_before_durable_source_artifact() -> None:
    function = _run_predict()

    ordered_calls = tuple(
        (
            node.lineno,
            _call_name(node),
        )
        for node in _calls(
            function
        )
    )

    prediction_write_line = min(
        line
        for line, name in ordered_calls
        if name
        == "write_prediction_artifacts"
    )

    durable_build_line = min(
        line
        for line, name in ordered_calls
        if name
        == "DurablePredictionEvaluationSource"
    )

    durable_write_line = min(
        line
        for line, name in ordered_calls
        if name
        == "write_operation_artifact"
    )

    assert (
        prediction_write_line
        < durable_build_line
        < durable_write_line
    )


def test_run_predict_returns_evaluation_source_artifact() -> None:
    function = _run_predict()

    returns = tuple(
        node
        for node in ast.walk(
            function
        )
        if isinstance(
            node,
            ast.Return,
        )
        and isinstance(
            node.value,
            ast.Dict,
        )
    )

    assert len(
        returns
    ) == 1

    keys = tuple(
        ast.literal_eval(key)
        for key in returns[0].value.keys
        if key is not None
        and isinstance(
            key,
            ast.Constant,
        )
    )

    assert (
        "evaluation_source_artifact"
        in keys
    )


def test_prediction_writer_contract_remains_unchanged() -> None:
    source = _source(
        PREDICTION_WRITER_PATH
    )

    assert (
        "evaluation_source.json"
        not in source
    )

    assert (
        "prediction-evaluation-sources"
        not in source
    )


def test_weekly_has_no_direct_durable_source_integration() -> None:
    source = _source(
        WEEKLY_PATH
    )

    assert (
        "DurablePredictionEvaluationSource"
        not in source
    )

    assert (
        "evaluation_source.json"
        not in source
    )

    assert (
        "prediction-evaluation-sources"
        not in source
    )


def test_weekly_continues_to_delegate_to_run_predict() -> None:
    tree = _tree(
        WEEKLY_PATH
    )

    calls = tuple(
        _call_name(node)
        for node in ast.walk(
            tree
        )
        if isinstance(
            node,
            ast.Call,
        )
    )

    assert calls.count(
        "run_predict"
    ) == 1


def test_durable_source_core_remains_filesystem_independent() -> None:
    source = _source(
        CORE_PATH
    )

    forbidden = (
        "write_operation_artifact",
        "write_prediction_artifacts",
        "Path(",
        ".write_text(",
        ".write_bytes(",
        "open(",
    )

    assert not any(
        token in source
        for token in forbidden
    )


def test_operation_writer_remains_generic() -> None:
    source = _source(
        OPERATION_WRITER_PATH
    )

    assert (
        "prediction-evaluation-sources"
        not in source
    )

    assert (
        "evaluation_source.json"
        not in source
    )


def test_integration_does_not_modify_prediction_json_payload_contract() -> None:
    source = _source(
        PREDICT_PATH
    )

    function = _run_predict()

    function_source = ast.unparse(
        function
    )

    assert (
        "prediction_to_dict(result)"
        in function_source
    )

    assert (
        "write_prediction_artifacts("
        in function_source
    )

    assert (
        "artifact_payload"
        in function_source
    )

    # Durable evaluation source must be persisted independently,
    # not inserted into prediction.json.
    assert (
        'artifact_payload["evaluation_source"]'
        not in source
    )

    assert (
        "artifact_payload['evaluation_source']"
        not in source
    )