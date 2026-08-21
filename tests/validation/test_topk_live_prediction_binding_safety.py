from __future__ import annotations

import ast
import inspect
from pathlib import Path

import lrp.evaluation.topk_live_prediction_binding as product
from lrp.evaluation.topk_live_prediction_binding import (
    TopKLivePredictionBinder,
)


def _tree() -> ast.AST:
    path = Path(
        inspect.getsourcefile(
            TopKLivePredictionBinder
        )
    )

    return ast.parse(
        path.read_text(
            encoding="utf-8-sig"
        )
    )


def _imports() -> tuple[str, ...]:
    values: list[str] = []

    for node in ast.walk(
        _tree()
    ):
        if isinstance(
            node,
            ast.Import,
        ):
            values.extend(
                alias.name
                for alias in node.names
            )

        elif isinstance(
            node,
            ast.ImportFrom,
        ):
            values.append(
                node.module
                or ""
            )

    return tuple(
        values
    )


def _calls() -> set[str]:
    result: set[str] = set()

    for node in ast.walk(
        _tree()
    ):
        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        if isinstance(
            node.func,
            ast.Name,
        ):
            result.add(
                node.func.id
            )

        elif isinstance(
            node.func,
            ast.Attribute,
        ):
            result.add(
                node.func.attr
            )

    return result


def test_product_is_read_only() -> None:
    forbidden = {
        "open",
        "write",
        "write_text",
        "write_bytes",
        "unlink",
        "mkdir",
        "rmdir",
        "rename",
        "replace",
        "remove",
        "touch",
    }

    assert (
        _calls()
        & forbidden
    ) == set()


def test_product_has_no_forbidden_dependencies() -> None:
    forbidden = (
        "lrp.pipelines.prediction",
        "lrp.cli",
        "lrp.production",
        "lrp.evaluation.topk_replay_adapter",
        "lrp.evaluation.topk_replay_evaluation",
        "lrp.evaluation.topk_walkforward",
        "tools",
        "random",
        "secrets",
    )

    imports = _imports()

    violations = [
        name
        for name in imports
        if any(
            name == token
            or name.startswith(
                token + "."
            )
            for token in forbidden
        )
    ]

    assert violations == []


def test_product_does_not_execute_prediction_pipeline() -> None:
    assert "PredictionPipeline" not in vars(
        product
    )

    assert "run_predict" not in _calls()
    assert "run" not in {
        call
        for call in _calls()
        if call == "run_predict"
    }


def test_product_does_not_execute_walkforward_or_e2e_evaluation() -> None:
    imports = _imports()

    assert not any(
        name.startswith(
            "lrp.evaluation.topk_walkforward"
        )
        for name in imports
    )

    assert not any(
        name.startswith(
            "lrp.evaluation.topk_replay_evaluation"
        )
        for name in imports
    )

    assert "evaluate" not in _calls()
