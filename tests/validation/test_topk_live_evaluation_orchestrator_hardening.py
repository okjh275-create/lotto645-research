from __future__ import annotations

import ast
import copy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from lrp.contracts.exceptions import ContractError
from lrp.io.draws import HistoryRow
from lrp.pipelines.models import (
    PredictionGenerationResult,
    PredictionRequest,
    PredictionResult,
)
from lrp.evaluation.contracts import EvaluationWindow
from lrp.evaluation.topk_live_evaluation_orchestrator import (
    TopKLiveEvaluationOrchestrator,
    TopKLiveEvaluationRequest,
)


PRODUCT = Path(
    "lrp/evaluation/"
    "topk_live_evaluation_orchestrator.py"
)


def _prediction_result(
    round_no: object = 1200,
) -> PredictionResult:
    request = PredictionRequest(
        round_no=round_no,  # type: ignore[arg-type]
        seed=20260821,
        long_gap_numbers=frozenset({1}),
    )

    generation = PredictionGenerationResult(
        request=request,
        windows=(10, 20, 50),
        probabilities={1: 1.0},
        statistics_contract=object(),
        number_signals={},
        candidates=(),
        statistics_version="af06-test-statistics",
        candidate_version="af06-test-candidate",
    )

    selected_numbers = (
        (1, 2, 3, 4, 5, 6),
        (7, 8, 9, 10, 11, 12),
        (13, 14, 15, 16, 17, 18),
        (19, 20, 21, 22, 23, 24),
        (25, 26, 27, 28, 29, 30),
        (31, 32, 33, 34, 35, 36),
        (37, 38, 39, 40, 41, 42),
        (1, 8, 15, 22, 29, 36),
        (2, 9, 16, 23, 30, 37),
        (3, 10, 17, 24, 31, 38),
    )

    diversity = SimpleNamespace(
        selected=tuple(
            SimpleNamespace(
                numbers=numbers
            )
            for numbers in selected_numbers
        )
    )

    return PredictionResult(
        generation=generation,
        scored_candidates=(),
        ranking=(),
        diversity=diversity,
        practical=(),
        generated_at_kst=datetime(
            2026,
            8,
            21,
            0,
            0,
            0,
            tzinfo=timezone.utc,
        ),
    )


def _history() -> tuple[HistoryRow, ...]:
    return (
        HistoryRow(
            round_no=1197,
            numbers=(1, 2, 3, 4, 5, 6),
        ),
        HistoryRow(
            round_no=1198,
            numbers=(7, 8, 9, 10, 11, 12),
        ),
        HistoryRow(
            round_no=1199,
            numbers=(13, 14, 15, 16, 17, 18),
        ),
    )


def _draw(
    round_no: int = 1200,
) -> SimpleNamespace:
    return SimpleNamespace(
        round_no=round_no,
        numbers=(1, 2, 3, 4, 5, 6),
    )


def _request(
    **overrides: object,
) -> TopKLiveEvaluationRequest:
    values: dict[str, object] = {
        "window": EvaluationWindow(
            name="af06",
            start_round=1200,
            end_round=1200,
        ),
        "candidate_prediction_result":
            _prediction_result(),
        "candidate_history_rows":
            _history(),
        "candidate_model_name":
            "candidate",
        "baseline_prediction_result":
            _prediction_result(),
        "baseline_history_rows":
            _history(),
        "baseline_model_name":
            "baseline",
        "actual_draws":
            (_draw(),),
        "candidate_regime_id":
            "candidate-regime",
        "candidate_strategy_name":
            "candidate-strategy",
        "baseline_regime_id":
            "baseline-regime",
        "baseline_strategy_name":
            "baseline-strategy",
    }

    values.update(overrides)

    return TopKLiveEvaluationRequest(
        **values  # type: ignore[arg-type]
    )


def _evaluate(
    request: TopKLiveEvaluationRequest,
):
    return TopKLiveEvaluationOrchestrator().evaluate(
        request=request
    )


def test_orchestrator_repeated_evaluation_is_semantically_stable() -> None:
    orchestrator = TopKLiveEvaluationOrchestrator()
    request = _request()

    first = orchestrator.evaluate(
        request=request
    )
    second = orchestrator.evaluate(
        request=request
    )

    assert first == second
    assert repr(first) == repr(second)


def test_evaluation_does_not_mutate_history_inputs() -> None:
    candidate_history = _history()
    baseline_history = tuple(
        reversed(_history())
    )

    candidate_before = copy.deepcopy(
        candidate_history
    )
    baseline_before = copy.deepcopy(
        baseline_history
    )

    _evaluate(
        _request(
            candidate_history_rows=candidate_history,
            baseline_history_rows=baseline_history,
        )
    )

    assert candidate_history == candidate_before
    assert baseline_history == baseline_before


def test_evaluation_does_not_mutate_prediction_results() -> None:
    candidate = _prediction_result()
    baseline = _prediction_result()

    candidate_generation = candidate.generation
    candidate_diversity = candidate.diversity
    baseline_generation = baseline.generation
    baseline_diversity = baseline.diversity

    _evaluate(
        _request(
            candidate_prediction_result=candidate,
            baseline_prediction_result=baseline,
        )
    )

    assert candidate.generation is candidate_generation
    assert candidate.diversity is candidate_diversity
    assert baseline.generation is baseline_generation
    assert baseline.diversity is baseline_diversity


def test_candidate_and_baseline_identity_chain_is_stable() -> None:
    first = _evaluate(_request())
    second = _evaluate(_request())

    assert (
        first.candidate_binding.model_name
        == second.candidate_binding.model_name
        == "candidate"
    )
    assert (
        first.baseline_binding.model_name
        == second.baseline_binding.model_name
        == "baseline"
    )

    assert (
        first.candidate_replay_prediction.model_name
        == second.candidate_replay_prediction.model_name
        == "candidate"
    )
    assert (
        first.baseline_replay_prediction.model_name
        == second.baseline_replay_prediction.model_name
        == "baseline"
    )


def test_orchestrator_optional_none_provenance_is_stable() -> None:
    request = _request(
        candidate_regime_id=None,
        candidate_strategy_name=None,
        baseline_regime_id=None,
        baseline_strategy_name=None,
    )

    first = _evaluate(request)
    second = _evaluate(request)

    assert first.candidate_binding.source.regime_id is None
    assert first.candidate_binding.source.strategy_name is None
    assert first.baseline_binding.source.regime_id is None
    assert first.baseline_binding.source.strategy_name is None

    assert (
        first.candidate_replay_prediction.regime_id
        is None
    )
    assert (
        first.candidate_replay_prediction.strategy_name
        is None
    )
    assert (
        first.baseline_replay_prediction.regime_id
        is None
    )
    assert (
        first.baseline_replay_prediction.strategy_name
        is None
    )

    assert first == second


def test_rejects_boolean_candidate_prediction_round() -> None:
    with pytest.raises(ContractError):
        _evaluate(
            _request(
                candidate_prediction_result=
                    _prediction_result(True)
            )
        )


def test_rejects_non_integer_candidate_prediction_round() -> None:
    with pytest.raises(ContractError):
        _evaluate(
            _request(
                candidate_prediction_result=
                    _prediction_result("1200")
            )
        )


def test_rejects_boolean_baseline_prediction_round() -> None:
    with pytest.raises(ContractError):
        _evaluate(
            _request(
                baseline_prediction_result=
                    _prediction_result(True)
            )
        )


def test_rejects_non_integer_baseline_prediction_round() -> None:
    with pytest.raises(ContractError):
        _evaluate(
            _request(
                baseline_prediction_result=
                    _prediction_result("1200")
            )
        )


def test_actual_draw_order_is_semantically_stable() -> None:
    window = EvaluationWindow(
        name="af06-draw-order",
        start_round=1200,
        end_round=1201,
    )

    # The live orchestrator owns one prediction round.
    # Extra draws are downstream evaluation context.
    forward = (
        _draw(1200),
        _draw(1201),
    )
    reverse = tuple(
        reversed(forward)
    )

    candidate = _prediction_result()
    baseline = _prediction_result()
    candidate_history = _history()
    baseline_history = _history()

    first = _evaluate(
        _request(
            window=window,
            candidate_prediction_result=candidate,
            baseline_prediction_result=baseline,
            candidate_history_rows=candidate_history,
            baseline_history_rows=baseline_history,
            actual_draws=forward,
        )
    )
    second = _evaluate(
        _request(
            window=window,
            candidate_prediction_result=candidate,
            baseline_prediction_result=baseline,
            candidate_history_rows=candidate_history,
            baseline_history_rows=baseline_history,
            actual_draws=reverse,
        )
    )

    assert first == second


def test_orchestrator_product_has_no_runtime_nondeterminism_dependencies() -> None:
    tree = ast.parse(
        PRODUCT.read_text(
            encoding="utf-8-sig"
        )
    )

    imports: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(
                alias.name
                for alias in node.names
            )

        elif isinstance(node, ast.ImportFrom):
            imports.append(
                node.module or ""
            )

    forbidden = (
        "random",
        "secrets",
        "uuid",
        "time",
        "datetime",
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


def test_product_has_no_filesystem_or_process_dependency() -> None:
    tree = ast.parse(
        PRODUCT.read_text(
            encoding="utf-8-sig"
        )
    )

    imports: list[str] = []
    calls: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(
                alias.name
                for alias in node.names
            )

        elif isinstance(node, ast.ImportFrom):
            imports.append(
                node.module or ""
            )

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.append(
                    node.func.id
                )

            elif isinstance(
                node.func,
                ast.Attribute,
            ):
                calls.append(
                    node.func.attr
                )

    forbidden_imports = (
        "pathlib",
        "os",
        "shutil",
        "tempfile",
        "subprocess",
    )

    assert not any(
        value == token
        or value.startswith(
            token + "."
        )
        for value in imports
        for token in forbidden_imports
    )

    forbidden_calls = {
        "open",
        "read_text",
        "read_bytes",
        "write_text",
        "write_bytes",
        "unlink",
        "mkdir",
        "rename",
        "replace",
    }

    assert not (
        set(calls)
        & forbidden_calls
    )
