from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from lrp.evaluation import EvaluationWindow
from lrp.evaluation.topk_replay_adapter import (
    TopKReplayPrediction,
)
from lrp.evaluation.topk_replay_evaluation import (
    TopKReplayEvaluationRequest,
    TopKReplayEvaluationService,
)


@dataclass(frozen=True)
class Draw:
    round_no: int
    numbers: tuple[int, ...]


def _predictions() -> tuple[
    tuple[int, ...],
    ...
]:
    return (
        (1, 2, 3, 4, 5, 6),
        (7, 8, 9, 10, 11, 12),
        (13, 14, 15, 16, 17, 18),
        (19, 20, 21, 22, 23, 24),
        (25, 26, 27, 28, 29, 30),
        (31, 32, 33, 34, 35, 36),
        (37, 38, 39, 40, 41, 42),
        (1, 2, 3, 43, 44, 45),
        (10, 11, 12, 13, 14, 15),
        (20, 21, 22, 23, 24, 25),
    )


def _prediction(
    *,
    model_name: str,
) -> TopKReplayPrediction:
    return TopKReplayPrediction(
        round_no=1200,
        history_rounds=(
            1197,
            1198,
            1199,
        ),
        predictions=_predictions(),
        model_name=model_name,
        regime_id="R1",
        strategy_name="S1",
    )


def _request() -> TopKReplayEvaluationRequest:
    return TopKReplayEvaluationRequest(
        window=EvaluationWindow(
            name="ac04-safety",
            start_round=1200,
            end_round=1200,
        ),
        candidate_predictions=(
            _prediction(
                model_name="candidate"
            ),
        ),
        baseline_predictions=(
            _prediction(
                model_name="baseline"
            ),
        ),
        actual_draws=(
            Draw(
                round_no=1200,
                numbers=(
                    1, 2, 3, 4, 5, 6
                ),
            ),
        ),
    )


def test_service_is_read_only(
    tmp_path: Path,
) -> None:
    before = {
        path.relative_to(
            tmp_path
        ).as_posix():
            path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    TopKReplayEvaluationService().evaluate(
        request=_request()
    )

    after = {
        path.relative_to(
            tmp_path
        ).as_posix():
            path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    assert after == before


def test_service_does_not_mutate_inputs() -> None:
    request = _request()

    candidate_before = tuple(
        request.candidate_predictions
    )

    baseline_before = tuple(
        request.baseline_predictions
    )

    draws_before = tuple(
        request.actual_draws
    )

    TopKReplayEvaluationService().evaluate(
        request=request
    )

    assert request.candidate_predictions == candidate_before
    assert request.baseline_predictions == baseline_before
    assert request.actual_draws == draws_before


def test_product_has_no_persistence_dependency() -> None:
    path = Path(
        "lrp/evaluation/topk_replay_evaluation.py"
    )

    tree = ast.parse(
        path.read_text(
            encoding="utf-8-sig"
        )
    )

    forbidden_calls = {
        "write_text",
        "write_bytes",
        "unlink",
        "mkdir",
        "rename",
        "replace",
        "remove",
    }

    calls = set()

    for node in ast.walk(tree):
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

    assert calls.isdisjoint(
        forbidden_calls
    )


def test_product_has_no_production_dependency() -> None:
    path = Path(
        "lrp/evaluation/topk_replay_evaluation.py"
    )

    tree = ast.parse(
        path.read_text(
            encoding="utf-8-sig"
        )
    )

    imports = []

    for node in ast.walk(tree):
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
                node.module
                or ""
            )

    assert not any(
        value == "lrp.production"
        or value.startswith(
            "lrp.production."
        )
        for value in imports
    )


def test_product_has_no_cli_dependency() -> None:
    path = Path(
        "lrp/evaluation/topk_replay_evaluation.py"
    )

    tree = ast.parse(
        path.read_text(
            encoding="utf-8-sig"
        )
    )

    imports = []

    for node in ast.walk(tree):
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
                node.module
                or ""
            )

    assert not any(
        value == "lrp.cli"
        or value.startswith(
            "lrp.cli."
        )
        for value in imports
    )


def test_product_remains_module_local() -> None:
    import lrp.evaluation as package

    assert not hasattr(
        package,
        "TopKReplayEvaluationRequest",
    )

    assert not hasattr(
        package,
        "TopKReplayEvaluationResult",
    )

    assert not hasattr(
        package,
        "TopKReplayEvaluationService",
    )


def test_repeated_evaluation_is_deterministic() -> None:
    service = TopKReplayEvaluationService()

    request = _request()

    first = service.evaluate(
        request=request
    )

    second = service.evaluate(
        request=request
    )

    assert first == second
    assert repr(first) == repr(second)
