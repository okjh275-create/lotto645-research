from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest


def _product():
    return importlib.import_module(
        "lrp.cli.durable_replay_evaluation"
    )


def _shared_args() -> list[str]:
    return [
        "--history",
        "history.json",
        "--window-name",
        "window-hardening",
        "--start-round",
        "1000",
        "--end-round",
        "1300",
    ]


def test_hardening_public_surface_is_exact() -> None:
    product = _product()

    source = Path(product.__file__).read_text(
        encoding="utf-8-sig"
    )
    tree = ast.parse(source)

    classes = tuple(
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    )

    functions = tuple(
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
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

    assert str(inspect.signature(product.main)) == (
        "(argv: 'Sequence[str] | None' = None) -> 'int'"
    )


def test_hardening_parser_surface_is_exact() -> None:
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
    )

    assert {
        action.dest: action.required
        for action in actions
    } == {
        "history": True,
        "window_name": True,
        "start_round": True,
        "end_round": True,
        "candidate": False,
        "baseline": False,
        "artifact_root": False,
        "candidate_selector": False,
        "baseline_selector": False,
    }


def test_hardening_source_parser_valid_matrix() -> None:
    product = _product()

    matrix = (
        (
            "a.json|1|model-a",
            ("a.json", 1, "model-a", None, None),
        ),
        (
            "a.json|1|model-a|regime-a",
            ("a.json", 1, "model-a", "regime-a", None),
        ),
        (
            "a.json|1|model-a|regime-a|strategy-a",
            (
                "a.json",
                1,
                "model-a",
                "regime-a",
                "strategy-a",
            ),
        ),
    )

    for value, expected in matrix:
        result = product._parse_source(value)

        assert (
            str(result.artifact_path),
            result.round_no,
            result.model_name,
            result.regime_id,
            result.strategy_name,
        ) == expected


@pytest.mark.parametrize(
    "value",
    (
        "",
        "a",
        "a|b",
        "a|x|model",
        "a|1|b|c|d|e",
    ),
)
def test_hardening_source_parser_rejects_invalid_descriptors(
    value: str,
) -> None:
    product = _product()

    with pytest.raises(argparse.ArgumentTypeError):
        product._parse_source(value)


def test_hardening_selector_parser_valid_matrix() -> None:
    product = _product()

    matrix = (
        (
            "1|model-a",
            (1, "model-a", None, None),
        ),
        (
            "1|model-a|regime-a",
            (1, "model-a", "regime-a", None),
        ),
        (
            "1|model-a|regime-a|strategy-a",
            (
                1,
                "model-a",
                "regime-a",
                "strategy-a",
            ),
        ),
        (
            "7|model||strategy",
            (7, "model", None, "strategy"),
        ),
    )

    for value, expected in matrix:
        result = product._parse_selector(value)

        assert (
            result.round_no,
            result.model_name,
            result.regime_id,
            result.strategy_name,
        ) == expected


@pytest.mark.parametrize(
    "value",
    (
        "",
        "1",
        "+1|model",
        "-1|model",
        "abc|model",
        "1|model|regime|strategy|extra",
    ),
)
def test_hardening_selector_parser_rejects_invalid_descriptors(
    value: str,
) -> None:
    product = _product()

    with pytest.raises(argparse.ArgumentTypeError):
        product._parse_selector(value)


def test_hardening_explicit_mode_invokes_only_execution_service(
    monkeypatch,
) -> None:
    product = _product()

    execution_requests = []
    execution_builds = 0
    composition_builds = 0
    sentinel = object()

    class ExecutionService:
        def __init__(self) -> None:
            nonlocal execution_builds
            execution_builds += 1

        def execute(self, *, request):
            execution_requests.append(request)
            return sentinel

    class ForbiddenCompositionService:
        def __init__(self) -> None:
            nonlocal composition_builds
            composition_builds += 1
            raise AssertionError(
                "explicit mode constructed composition service"
            )

    monkeypatch.setattr(
        product,
        "DurableReplayExecutionService",
        ExecutionService,
    )

    monkeypatch.setattr(
        product,
        "DurableReplayCompositionService",
        ForbiddenCompositionService,
    )

    monkeypatch.setattr(
        product,
        "_result_to_dict",
        lambda result: {
            "identity": result is sentinel
        },
    )

    stdout = io.StringIO()

    with redirect_stdout(stdout):
        exit_code = product.main(
            _shared_args()
            + [
                "--candidate",
                "candidate.json|1300|candidate-model",
                "--baseline",
                "baseline.json|1200|baseline-model",
            ]
        )

    assert exit_code == 0
    assert execution_builds == 1
    assert composition_builds == 0
    assert len(execution_requests) == 1

    request = execution_requests[0]

    assert tuple(
        source.round_no
        for source in request.candidate_sources
    ) == (1300,)

    assert tuple(
        source.round_no
        for source in request.baseline_sources
    ) == (1200,)

    assert json.loads(stdout.getvalue()) == {
        "identity": True
    }


def test_hardening_selector_mode_invokes_only_composition_service(
    monkeypatch,
) -> None:
    product = _product()

    composition_requests = []
    composition_builds = 0
    execution_builds = 0
    sentinel = object()

    class CompositionService:
        def __init__(self) -> None:
            nonlocal composition_builds
            composition_builds += 1

        def execute(self, *, request):
            composition_requests.append(request)
            return sentinel

    class ForbiddenExecutionService:
        def __init__(self) -> None:
            nonlocal execution_builds
            execution_builds += 1
            raise AssertionError(
                "selector mode constructed execution service"
            )

    monkeypatch.setattr(
        product,
        "DurableReplayCompositionService",
        CompositionService,
    )

    monkeypatch.setattr(
        product,
        "DurableReplayExecutionService",
        ForbiddenExecutionService,
    )

    monkeypatch.setattr(
        product,
        "_result_to_dict",
        lambda result: {
            "identity": result is sentinel
        },
    )

    stdout = io.StringIO()

    with redirect_stdout(stdout):
        exit_code = product.main(
            _shared_args()
            + [
                "--artifact-root",
                "artifact-root",
                "--candidate-selector",
                "1300|candidate-a|regime-a|strategy-a",
                "--candidate-selector",
                "1200|candidate-b",
                "--baseline-selector",
                "1100|baseline-a",
            ]
        )

    assert exit_code == 0
    assert composition_builds == 1
    assert execution_builds == 0
    assert len(composition_requests) == 1

    request = composition_requests[0]

    assert str(request.artifact_root) == "artifact-root"

    assert tuple(
        selector.round_no
        for selector in request.candidate_selectors
    ) == (
        1300,
        1200,
    )

    assert tuple(
        selector.round_no
        for selector in request.baseline_selectors
    ) == (1100,)

    assert json.loads(stdout.getvalue()) == {
        "identity": True
    }


@pytest.mark.parametrize(
    "mode_args",
    (
        (),
        (
            "--artifact-root",
            "root",
        ),
        (
            "--candidate-selector",
            "1300|model-a",
        ),
        (
            "--candidate",
            "a.json|1300|model-a",
            "--artifact-root",
            "root",
        ),
        (
            "--candidate",
            "a.json|1300|model-a",
            "--candidate-selector",
            "1300|model-a",
        ),
        (
            "--candidate",
            "a.json|1300|model-a",
            "--artifact-root",
            "root",
            "--candidate-selector",
            "1300|model-a",
        ),
    ),
)
def test_hardening_invalid_mode_combinations_fail_closed(
    mode_args: tuple[str, ...],
) -> None:
    product = _product()
    stderr = io.StringIO()

    with redirect_stderr(stderr):
        with pytest.raises(SystemExit):
            product.main(
                _shared_args()
                + list(mode_args)
            )


def test_hardening_candidate_only_explicit_shape_is_preserved(
    monkeypatch,
) -> None:
    product = _product()
    requests = []

    class Service:
        def execute(self, *, request):
            requests.append(request)
            return object()

    monkeypatch.setattr(
        product,
        "DurableReplayExecutionService",
        Service,
    )

    monkeypatch.setattr(
        product,
        "_result_to_dict",
        lambda result: {},
    )

    with redirect_stdout(io.StringIO()):
        assert product.main(
            _shared_args()
            + [
                "--candidate",
                "candidate.json|1300|candidate-model",
            ]
        ) == 0

    request = requests[0]

    assert len(request.candidate_sources) == 1
    assert request.baseline_sources == ()


def test_hardening_baseline_only_explicit_shape_is_preserved(
    monkeypatch,
) -> None:
    product = _product()
    requests = []

    class Service:
        def execute(self, *, request):
            requests.append(request)
            return object()

    monkeypatch.setattr(
        product,
        "DurableReplayExecutionService",
        Service,
    )

    monkeypatch.setattr(
        product,
        "_result_to_dict",
        lambda result: {},
    )

    with redirect_stdout(io.StringIO()):
        assert product.main(
            _shared_args()
            + [
                "--baseline",
                "baseline.json|1200|baseline-model",
            ]
        ) == 0

    request = requests[0]

    assert request.candidate_sources == ()
    assert len(request.baseline_sources) == 1


def test_hardening_candidate_selector_order_is_preserved(
    monkeypatch,
) -> None:
    product = _product()
    requests = []

    class Service:
        def execute(self, *, request):
            requests.append(request)
            return object()

    monkeypatch.setattr(
        product,
        "DurableReplayCompositionService",
        Service,
    )

    monkeypatch.setattr(
        product,
        "_result_to_dict",
        lambda result: {},
    )

    with redirect_stdout(io.StringIO()):
        assert product.main(
            _shared_args()
            + [
                "--artifact-root",
                "root",
                "--candidate-selector",
                "1300|model-c",
                "--candidate-selector",
                "1001|model-a",
                "--candidate-selector",
                "1250|model-b",
            ]
        ) == 0

    assert tuple(
        selector.round_no
        for selector in requests[0].candidate_selectors
    ) == (
        1300,
        1001,
        1250,
    )


def test_hardening_baseline_selector_order_is_preserved(
    monkeypatch,
) -> None:
    product = _product()
    requests = []

    class Service:
        def execute(self, *, request):
            requests.append(request)
            return object()

    monkeypatch.setattr(
        product,
        "DurableReplayCompositionService",
        Service,
    )

    monkeypatch.setattr(
        product,
        "_result_to_dict",
        lambda result: {},
    )

    with redirect_stdout(io.StringIO()):
        assert product.main(
            _shared_args()
            + [
                "--artifact-root",
                "root",
                "--baseline-selector",
                "1250|model-c",
                "--baseline-selector",
                "1000|model-a",
                "--baseline-selector",
                "1100|model-b",
            ]
        ) == 0

    assert tuple(
        selector.round_no
        for selector in requests[0].baseline_selectors
    ) == (
        1250,
        1000,
        1100,
    )


def test_hardening_execution_failure_identity_is_preserved(
    monkeypatch,
) -> None:
    product = _product()
    failure = RuntimeError(
        "execution-failure"
    )

    class Service:
        def execute(self, *, request):
            raise failure

    monkeypatch.setattr(
        product,
        "DurableReplayExecutionService",
        Service,
    )

    with pytest.raises(RuntimeError) as caught:
        product.main(
            _shared_args()
            + [
                "--candidate",
                "candidate.json|1300|model",
            ]
        )

    assert caught.value is failure


def test_hardening_composition_failure_identity_is_preserved(
    monkeypatch,
) -> None:
    product = _product()
    failure = RuntimeError(
        "composition-failure"
    )

    class Service:
        def execute(self, *, request):
            raise failure

    monkeypatch.setattr(
        product,
        "DurableReplayCompositionService",
        Service,
    )

    with pytest.raises(RuntimeError) as caught:
        product.main(
            _shared_args()
            + [
                "--artifact-root",
                "root",
                "--candidate-selector",
                "1300|model",
            ]
        )

    assert caught.value is failure


def test_hardening_single_result_serialization_path_is_exact() -> None:
    product = _product()

    source = Path(product.__file__).read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(source)

    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "main"
    )

    calls = tuple(
        ast.unparse(node.func)
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
    )

    assert calls.count(
        "_result_to_dict"
    ) == 1

    assert calls.count(
        "json.dumps"
    ) == 1

    assert calls.count(
        "print"
    ) == 1


def test_hardening_exact_dependency_boundary() -> None:
    product = _product()

    source = Path(product.__file__).read_text(
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
        "typing",
        "lrp.operations.durable_replay_execution",
        "lrp.operations.durable_replay_artifact_discovery",
        "lrp.operations.durable_replay_composition",
    }


def test_hardening_exact_owned_raise_boundary() -> None:
    product = _product()

    source = Path(product.__file__).read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(source)

    raises = tuple(
        ast.unparse(node.exc)
        for node in ast.walk(tree)
        if isinstance(node, ast.Raise)
        and node.exc is not None
    )

    assert raises == (
        (
            "argparse.ArgumentTypeError("
            "'source descriptor must contain 3 to 5 fields'"
            ")"
        ),
        (
            "argparse.ArgumentTypeError("
            "'source round_no must be an integer'"
            ")"
        ),
        (
            "argparse.ArgumentTypeError("
            "'selector descriptor must contain 2 to 4 fields'"
            ")"
        ),
        (
            "argparse.ArgumentTypeError("
            "'selector round_no must be an integer'"
            ")"
        ),
    )


def test_hardening_no_exception_normalization_layer() -> None:
    product = _product()

    source = Path(product.__file__).read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(source)

    assert not any(
        isinstance(
            node,
            ast.ExceptHandler,
        )
        for node in ast.walk(tree)
    )


def test_hardening_no_filesystem_artifact_io() -> None:
    product = _product()

    source = Path(product.__file__).read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(source)

    calls = {
        ast.unparse(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }

    forbidden = {
        "open",
        "Path.open",
        "Path.read_text",
        "Path.read_bytes",
        "Path.write_text",
        "Path.write_bytes",
        "Path.glob",
        "Path.rglob",
        "Path.iterdir",
    }

    assert not (
        calls
        & forbidden
    )


def test_hardening_structural_call_contract_is_exact() -> None:
    product = _product()

    source = Path(product.__file__).read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(source)

    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "main"
    )

    calls = tuple(
        ast.unparse(node.func)
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
    )

    assert calls.count(
        "DurableReplayExecutionRequest"
    ) == 1

    assert calls.count(
        "DurableReplayExecutionService"
    ) == 1

    assert calls.count(
        "DurableReplayCompositionRequest"
    ) == 1

    assert calls.count(
        "DurableReplayCompositionService"
    ) == 1

    assert sum(
        call.endswith(".execute")
        for call in calls
    ) == 2


def test_hardening_product_is_free_of_quality_markers() -> None:
    product = _product()

    source = Path(product.__file__).read_text(
        encoding="utf-8-sig"
    )

    for marker in (
        "TODO",
        "FIXME",
        "XXX",
        "NotImplemented",
        "placeholder",
        "temporary",
        "workaround",
        "hack",
    ):
        assert marker not in source
