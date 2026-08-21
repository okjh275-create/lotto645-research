from __future__ import annotations

import ast
from pathlib import Path

import lrp.evaluation as evaluation_package

from lrp.evaluation.topk_live_evaluation_orchestrator import (
    TopKLiveEvaluationOrchestrator,
    TopKLiveEvaluationRequest,
    TopKLiveEvaluationResult,
)


PRODUCT = Path(
    "lrp/evaluation/topk_live_evaluation_orchestrator.py"
)


def _tree() -> ast.Module:
    return ast.parse(
        PRODUCT.read_text(
            encoding="utf-8-sig"
        )
    )


def test_product_public_surface_remains_module_local() -> None:
    names = (
        "TopKLiveEvaluationRequest",
        "TopKLiveEvaluationResult",
        "TopKLiveEvaluationOrchestrator",
    )

    assert all(
        value is not None
        for value in (
            TopKLiveEvaluationRequest,
            TopKLiveEvaluationResult,
            TopKLiveEvaluationOrchestrator,
        )
    )

    assert not any(
        hasattr(
            evaluation_package,
            name,
        )
        for name in names
    )


def test_product_has_no_forbidden_operational_dependencies() -> None:
    imports: list[str] = []

    for node in ast.walk(
        _tree()
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
            imports.append(
                node.module or ""
            )

    forbidden = (
        "lrp.cli",
        "lrp.production",
        "tools",
        "sqlite3",
        "pathlib",
    )

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


def test_product_has_no_mutating_io_calls() -> None:
    calls: set[str] = set()

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
            calls.add(
                node.func.id
            )

        elif isinstance(
            node.func,
            ast.Attribute,
        ):
            calls.add(
                node.func.attr
            )

    forbidden = {
        "open",
        "write",
        "write_text",
        "write_bytes",
        "unlink",
        "mkdir",
        "rmdir",
        "rename",
        "remove",
        "replace",
        "touch",
    }

    assert sorted(
        calls & forbidden
    ) == []


def test_product_does_not_own_prediction_or_history_io() -> None:
    text = PRODUCT.read_text(
        encoding="utf-8-sig"
    )

    forbidden_signals = (
        "PredictionPipeline(",
        "history_until_round(",
        "load_history",
        "sqlite3",
    )

    assert not any(
        signal in text
        for signal in forbidden_signals
    )