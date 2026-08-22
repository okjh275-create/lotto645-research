from __future__ import annotations

import ast
from pathlib import Path

import pytest


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

WEEKLY_PATH = Path(
    "lrp/cli/weekly.py"
)


def _source(path: Path) -> str:
    return path.read_text(
        encoding="utf-8-sig"
    )


def _tree(path: Path) -> ast.Module:
    return ast.parse(
        _source(path)
    )


def _run_predict() -> ast.FunctionDef:
    tree = _tree(
        PREDICT_PATH
    )

    matches = [
        node
        for node in tree.body
        if isinstance(
            node,
            ast.FunctionDef,
        )
        and node.name == "run_predict"
    ]

    assert len(matches) == 1

    return matches[0]


def _call_name(
    node: ast.Call,
) -> str | None:
    try:
        return ast.unparse(
            node.func
        )
    except Exception:
        return None


def _calls_named(
    root: ast.AST,
    name: str,
) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(root)
        if isinstance(
            node,
            ast.Call,
        )
        and _call_name(node) == name
    ]


def _one_call(
    root: ast.AST,
    name: str,
) -> ast.Call:
    matches = _calls_named(
        root,
        name,
    )

    assert len(matches) == 1

    return matches[0]


def _is_name(
    node: ast.AST,
    value: str,
) -> bool:
    return (
        isinstance(
            node,
            ast.Name,
        )
        and node.id == value
    )


def test_primary_prediction_artifact_write_precedes_durable_source_write() -> None:
    run_predict = _run_predict()

    prediction_write = _one_call(
        run_predict,
        "write_prediction_artifacts",
    )

    durable_build = _one_call(
        run_predict,
        "DurablePredictionEvaluationSource",
    )

    operation_write = _one_call(
        run_predict,
        "write_operation_artifact",
    )

    assert (
        prediction_write.lineno
        < durable_build.lineno
        < operation_write.lineno
    )


def test_operation_writer_payload_is_canonical_source_projection() -> None:
    run_predict = _run_predict()

    operation_write = _one_call(
        run_predict,
        "write_operation_artifact",
    )

    assert len(
        operation_write.args
    ) == 1

    payload = (
        operation_write.args[0]
    )

    assert isinstance(
        payload,
        ast.Call,
    )

    assert _call_name(
        payload
    ) == "source_to_dict"

    assert len(
        payload.args
    ) == 1

    assert _is_name(
        payload.args[0],
        "durable_evaluation_source",
    )


def test_integration_failure_paths_are_not_swallowed() -> None:
    run_predict = _run_predict()

    relevant = {
        "write_prediction_artifacts",
        "DurablePredictionEvaluationSource",
        "write_operation_artifact",
    }

    for try_node in [
        node
        for node in ast.walk(
            run_predict
        )
        if isinstance(
            node,
            ast.Try,
        )
    ]:
        contained = {
            _call_name(node)
            for node in ast.walk(
                try_node
            )
            if isinstance(
                node,
                ast.Call,
            )
        }

        assert not (
            relevant
            & contained
        )


def test_evaluation_source_artifact_return_preserves_writer_identity() -> None:
    run_predict = _run_predict()

    returns = [
        node
        for node in ast.walk(
            run_predict
        )
        if isinstance(
            node,
            ast.Return,
        )
        and isinstance(
            node.value,
            ast.Dict,
        )
    ]

    assert len(returns) == 1

    result = returns[0].value

    values = {}

    for key, value in zip(
        result.keys,
        result.values,
    ):
        if isinstance(
            key,
            ast.Constant,
        ):
            values[
                key.value
            ] = value

    assert (
        "evaluation_source_artifact"
        in values
    )

    assert _is_name(
        values[
            "evaluation_source_artifact"
        ],
        "evaluation_source_artifact",
    )


def test_weekly_delegates_prediction_generation_exactly_once() -> None:
    tree = _tree(
        WEEKLY_PATH
    )

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(
            node,
            ast.Call,
        )
        and _call_name(node)
        == "run_predict"
    ]

    assert len(calls) == 1


def test_weekly_does_not_build_durable_evaluation_source_directly() -> None:
    tree = _tree(
        WEEKLY_PATH
    )

    assert not _calls_named(
        tree,
        "DurablePredictionEvaluationSource",
    )

    assert not _calls_named(
        tree,
        "source_to_dict",
    )


def test_weekly_does_not_own_prediction_evaluation_source_artifact_identity() -> None:
    tree = _tree(
        WEEKLY_PATH
    )

    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(
            node,
            ast.Constant,
        )
        and isinstance(
            node.value,
            str,
        )
    }

    assert (
        "prediction-evaluation-sources"
        not in literals
    )

    assert (
        "evaluation_source.json"
        not in literals
    )


def test_prediction_writer_remains_unaware_of_durable_evaluation_source() -> None:
    source = _source(
        PREDICTION_WRITER_PATH
    )

    assert (
        "prediction-evaluation-sources"
        not in source
    )

    assert (
        "evaluation_source.json"
        not in source
    )

    assert (
        "DurablePredictionEvaluationSource"
        not in source
    )


def test_integration_hardening_keeps_durable_source_core_filesystem_independent() -> None:
    source = _source(
        CORE_PATH
    )

    forbidden = (
        "write_operation_artifact",
        "write_prediction_artifacts",
        ".write_text(",
        ".write_bytes(",
        "Path(",
    )

    violations = [
        token
        for token in forbidden
        if token in source
    ]

    assert violations == []


@pytest.mark.parametrize(
    "token",
    [
        "datetime.now",
        "datetime.utcnow",
        "random.",
        "secrets.",
        "uuid.",
        "time.time",
    ],
)
def test_durable_source_core_has_no_runtime_nondeterminism_dependency(
    token: str,
) -> None:
    source = _source(
        CORE_PATH
    )

    assert token not in source


def test_predict_integration_uses_existing_operation_writer_only() -> None:
    source = _source(
        PREDICT_PATH
    )

    assert (
        "write_operation_artifact"
        in source
    )

    assert (
        "prediction-evaluation-sources"
        in source
    )

    assert (
        "evaluation_source.json"
        in source
    )

    assert (
        "Repository("
        not in source
    )

    assert (
        "EvaluationSourceRepository"
        not in source
    )


def test_partial_write_semantics_are_structurally_explicit() -> None:
    run_predict = _run_predict()

    prediction_write = _one_call(
        run_predict,
        "write_prediction_artifacts",
    )

    operation_write = _one_call(
        run_predict,
        "write_operation_artifact",
    )

    assert (
        prediction_write.lineno
        < operation_write.lineno
    )

    relevant_try_nodes = []

    for try_node in [
        node
        for node in ast.walk(
            run_predict
        )
        if isinstance(
            node,
            ast.Try,
        )
    ]:
        names = {
            _call_name(node)
            for node in ast.walk(
                try_node
            )
            if isinstance(
                node,
                ast.Call,
            )
        }

        if (
            "write_prediction_artifacts"
            in names
            or "write_operation_artifact"
            in names
        ):
            relevant_try_nodes.append(
                try_node
            )

    assert relevant_try_nodes == []
