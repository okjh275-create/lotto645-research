from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import importlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


_PRODUCT_MODULE = "lrp.cli.durable_replay_evaluation"


def _product():
    return importlib.import_module(
        _PRODUCT_MODULE
    )


def _argv() -> list[str]:
    return [
        "--history",
        "history.json",
        "--window-name",
        "window-001",
        "--start-round",
        "1230",
        "--end-round",
        "1231",
        "--candidate",
        "candidate-2.json|1231|candidate-A|regime-2|strategy-2",
        "--candidate",
        "candidate-1.json|1230|candidate-A|regime-1|strategy-1",
        "--baseline",
        "baseline-2.json|1231|baseline-A|regime-B2|strategy-B2",
        "--baseline",
        "baseline-1.json|1230|baseline-A|regime-B1|strategy-B1",
    ]


def _result() -> Any:
    return SimpleNamespace(
        evaluation={
            "z": 3,
            "a": 1,
            "nested": {
                "y": 2,
                "x": 1,
            },
        },
        candidate_model_name="candidate-A",
        baseline_model_name="baseline-A",
        round_count=2,
    )


@pytest.mark.parametrize(
    "value",
    [
        "candidate.json|+1230|candidate-A",
        "candidate.json|-1|candidate-A",
        "candidate.json| 1230|candidate-A",
        "candidate.json|1230 |candidate-A",
    ],
)
def test_signed_or_whitespace_round_syntax_is_rejected(
    value: str,
) -> None:
    product = _product()

    with pytest.raises(
        argparse.ArgumentTypeError
    ):
        product._parse_source(
            value
        )


def test_leading_zero_round_is_accepted_and_canonicalized() -> None:
    product = _product()

    source = product._parse_source(
        "candidate.json|001230|candidate-A"
    )

    assert source.round_no == 1230


def test_empty_artifact_path_is_currently_forwarded() -> None:
    product = _product()

    source = product._parse_source(
        "|1230|candidate-A"
    )

    assert source.artifact_path == ""
    assert source.round_no == 1230
    assert source.model_name == "candidate-A"


def test_empty_model_name_is_currently_forwarded() -> None:
    product = _product()

    source = product._parse_source(
        "candidate.json|1230|"
    )

    assert source.artifact_path == "candidate.json"
    assert source.round_no == 1230
    assert source.model_name == ""


def test_strategy_only_optional_context_is_preserved() -> None:
    product = _product()

    source = product._parse_source(
        "candidate.json|1230|candidate-A||strategy-A"
    )

    assert source.regime_id is None
    assert source.strategy_name == "strategy-A"


def test_parser_required_field_surface_is_exact() -> None:
    product = _product()
    parser = product._parser()

    actions = tuple(
        action
        for action in parser._actions
        if action.dest != "help"
    )

    assert tuple(
        action.dest
        for action in actions
    ) == (
        "history",
        "window_name",
        "start_round",
        "end_round",
        "candidate",
        "baseline",
        "artifact_root",
        "candidate_selector",
        "baseline_selector",
        "output",
    )

    output_action = next(
        action
        for action in actions
        if action.dest == "output"
    )

    assert output_action.required is False
    assert output_action.default is None


def test_parser_candidate_and_baseline_are_append_actions() -> None:
    product = _product()

    parser = product._parser()

    actions = {
        action.dest: action
        for action in parser._actions
        if action.dest != "help"
    }

    assert type(
        actions["candidate"]
    ).__name__ == "_AppendAction"

    assert type(
        actions["baseline"]
    ).__name__ == "_AppendAction"


def test_candidate_and_baseline_order_are_preserved_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()

    calls: list[Any] = []

    def fake_execute(
        self: Any,
        *,
        request: Any,
    ) -> Any:
        calls.append(
            request
        )
        return _result()

    monkeypatch.setattr(
        product.DurableReplayExecutionService,
        "execute",
        fake_execute,
    )

    product.main(
        _argv()
    )

    assert len(calls) == 1

    request = calls[0]

    assert tuple(
        source.artifact_path
        for source in request.candidate_sources
    ) == (
        "candidate-2.json",
        "candidate-1.json",
    )

    assert tuple(
        source.artifact_path
        for source in request.baseline_sources
    ) == (
        "baseline-2.json",
        "baseline-1.json",
    )


def test_execution_request_projection_is_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()

    calls: list[Any] = []

    def fake_execute(
        self: Any,
        *,
        request: Any,
    ) -> Any:
        calls.append(
            request
        )
        return _result()

    monkeypatch.setattr(
        product.DurableReplayExecutionService,
        "execute",
        fake_execute,
    )

    product.main(
        _argv()
    )

    request = calls[0]

    assert isinstance(
        request,
        product.DurableReplayExecutionRequest,
    )

    assert request.history_path == "history.json"
    assert request.window_name == "window-001"
    assert request.start_round == 1230
    assert request.end_round == 1231
    assert len(request.candidate_sources) == 2
    assert len(request.baseline_sources) == 2


def test_execution_service_is_invoked_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()

    count = 0

    def fake_execute(
        self: Any,
        *,
        request: Any,
    ) -> Any:
        nonlocal count
        count += 1
        return _result()

    monkeypatch.setattr(
        product.DurableReplayExecutionService,
        "execute",
        fake_execute,
    )

    assert product.main(
        _argv()
    ) == 0

    assert count == 1


def test_service_failure_identity_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()

    sentinel = RuntimeError(
        "AM hardening sentinel"
    )

    def fail(
        self: Any,
        *,
        request: Any,
    ) -> Any:
        raise sentinel

    monkeypatch.setattr(
        product.DurableReplayExecutionService,
        "execute",
        fail,
    )

    with pytest.raises(
        RuntimeError
    ) as exc_info:
        product.main(
            _argv()
        )

    assert exc_info.value is sentinel


def test_repeated_json_rendering_is_semantically_stable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    product = _product()

    monkeypatch.setattr(
        product.DurableReplayExecutionService,
        "execute",
        lambda self, *, request: _result(),
    )

    outputs = []

    for _ in range(5):
        assert product.main(
            _argv()
        ) == 0

        outputs.append(
            capsys.readouterr().out
        )

    assert len(
        set(outputs)
    ) == 1

    payload = json.loads(
        outputs[0]
    )

    assert tuple(
        payload
    ) == (
        "baseline_model_name",
        "candidate_model_name",
        "evaluation",
        "round_count",
        "status",
    )


@dataclass(frozen=True)
class _Sample:
    value: int
    items: tuple[int, ...]


@pytest.mark.parametrize(
    (
        "value",
        "expected",
    ),
    [
        (
            (1, 2, 3),
            [1, 2, 3],
        ),
        (
            [1, 2, 3],
            [1, 2, 3],
        ),
        (
            {
                "b": 2,
                "a": 1,
            },
            {
                "b": 2,
                "a": 1,
            },
        ),
        (
            _Sample(
                value=7,
                items=(1, 2),
            ),
            {
                "value": 7,
                "items": [1, 2],
            },
        ),
        (
            SimpleNamespace(
                alpha=1,
                beta=(2, 3),
            ),
            {
                "alpha": 1,
                "beta": [2, 3],
            },
        ),
        (
            Path(
                "artifact.json"
            ),
            "artifact.json",
        ),
    ],
)
def test_json_compatibility_projection_is_frozen(
    value: object,
    expected: object,
) -> None:
    product = _product()

    assert product._json_compatible(
        value
    ) == expected


def test_result_projection_surface_is_exact() -> None:
    product = _product()

    payload = product._result_to_dict(
        _result()
    )

    assert tuple(
        payload
    ) == (
        "status",
        "candidate_model_name",
        "baseline_model_name",
        "round_count",
        "evaluation",
    )

    assert payload["status"] == "PASS"
    assert payload["candidate_model_name"] == "candidate-A"
    assert payload["baseline_model_name"] == "baseline-A"
    assert payload["round_count"] == 2


def test_root_dispatcher_handler_identity_is_exact() -> None:
    import lrp.cli as root

    product = _product()

    assert len(
        root._COMMANDS
    ) == 16

    assert (
        root._COMMANDS[
            "durable-replay-evaluation"
        ]
        is product.main
    )

    assert (
        "model-evaluation"
        in root._COMMANDS
    )


def test_product_has_no_exception_normalization_layer() -> None:
    product = _product()

    source = Path(
        product.__file__
    ).read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(
        source
    )

    assert not any(
        isinstance(
            node,
            ast.ExceptHandler,
        )
        for node in ast.walk(tree)
    )


def test_product_has_exact_eight_owned_raise_sites() -> None:
    product = _product()

    source = Path(
        product.__file__
    ).read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(source)

    raises = tuple(
        ast.unparse(node.exc)
        for node in ast.walk(tree)
        if isinstance(
            node,
            ast.Raise,
        )
        and node.exc is not None
    )

    assert raises == (
        "argparse.ArgumentTypeError('source descriptor must contain 3 to 5 fields')",
        "argparse.ArgumentTypeError('source round_no must be an integer')",
        "argparse.ArgumentTypeError('selector descriptor must contain 2 to 5 fields')",
        "argparse.ArgumentTypeError('selector round_no must be an integer')",
        "argparse.ArgumentTypeError('artifact_key must not be empty')",
        "argparse.ArgumentTypeError('artifact_key must be at most 128 characters')",
        "argparse.ArgumentTypeError('artifact_key contains invalid characters')",
        "argparse.ArgumentTypeError('artifact_key must not be dot path')",
    )


def test_product_static_dependency_boundary_is_exact() -> None:
    product = _product()

    source = Path(
        product.__file__
    ).read_text(
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

    assert set(imports) == {
        "__future__",
        "argparse",
        "dataclasses",
        "json",
        "re",
        "typing",
        "lrp.operations",
        "lrp.operations.durable_replay_execution",
        "lrp.operations.durable_replay_artifact_discovery",
        "lrp.operations.durable_replay_composition",
    }


def test_product_structural_call_contract_is_exact() -> None:
    product = _product()

    source = Path(
        product.__file__
    ).read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(source)

    calls = tuple(
        ast.unparse(node.func)
        for node in ast.walk(tree)
        if isinstance(
            node,
            ast.Call,
        )
    )

    assert sum(
        call == "DurableReplayExecutionRequest"
        for call in calls
    ) == 1

    assert sum(
        call == "DurableReplayExecutionService"
        for call in calls
    ) == 1

    assert sum(
        call == "DurableReplayCompositionRequest"
        for call in calls
    ) == 1

    assert sum(
        call == "DurableReplayCompositionService"
        for call in calls
    ) == 1

    assert sum(
        call.endswith(".execute")
        for call in calls
    ) == 2

    assert sum(
        call == "_result_to_dict"
        for call in calls
    ) == 1

    assert sum(
        call == "json.dumps"
        for call in calls
    ) == 1


def test_product_has_no_lower_layer_ownership_leak() -> None:
    product = _product()

    source = Path(
        product.__file__
    ).read_text(
        encoding="utf-8-sig"
    )

    forbidden = (
        "subprocess",
        "tools.validation",
        "lrp.io.draws",
        "load_history",
        "history_until_round",
        "source_from_json",
        "source_from_dict",
        "durable_prediction_evaluation_source",
        "evaluation_source.json",
        "prediction-evaluation-sources",
        ".glob(",
        ".rglob(",
        ".iterdir(",
        "write_text",
        "write_bytes",
        ".mkdir(",
        "write_prediction_artifacts",
    )

    assert all(
        token not in source
        for token in forbidden
    )

    assert source.count(
        "write_operation_artifact"
    ) == 2


def test_product_public_surface_remains_minimal() -> None:
    product = _product()

    source = Path(
        product.__file__
    ).read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(source)

    classes = tuple(
        node.name
        for node in tree.body
        if isinstance(
            node,
            ast.ClassDef,
        )
    )

    functions = tuple(
        node.name
        for node in tree.body
        if isinstance(
            node,
            ast.FunctionDef,
        )
    )

    assert classes == ()

    assert functions == (
        "_parse_source",
        "_parse_selector",
        "_parser",
        "_json_compatible",
        "_result_to_dict",
        "main",
    )
