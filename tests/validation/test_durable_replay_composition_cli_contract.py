from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import io
import json
import sys
from contextlib import redirect_stdout
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


MODULE_NAME = "lrp.cli.durable_replay_evaluation"


def _module() -> ModuleType:
    return importlib.import_module(MODULE_NAME)


def _product_source() -> str:
    module = _module()
    return Path(module.__file__).read_text(
        encoding="utf-8-sig"
    )


def _parser():
    module = _module()
    return module._parser()


def _option_action(option: str):
    parser = _parser()

    for action in parser._actions:
        if option in action.option_strings:
            return action

    raise AssertionError(
        f"missing parser option: {option}"
    )


def _invoke_main(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
) -> str:
    module = _module()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "durable-replay-evaluation",
            *argv,
        ],
    )

    stdout = io.StringIO()

    with redirect_stdout(stdout):
        module.main()

    return stdout.getvalue()


def test_parse_selector_exists() -> None:
    module = _module()

    assert hasattr(
        module,
        "_parse_selector",
    )


def test_parse_selector_public_signature_is_exact() -> None:
    module = _module()

    parser = module._parse_selector

    assert str(
        inspect.signature(parser)
    ) == (
        "(value: 'str') -> "
        "'DurableReplayArtifactSelector'"
    )


@pytest.mark.parametrize(
    (
        "value",
        "expected",
    ),
    [
        (
            "1|model-a",
            (
                1,
                "model-a",
                None,
                None,
            ),
        ),
        (
            "1|model-a|regime-a",
            (
                1,
                "model-a",
                "regime-a",
                None,
            ),
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
            "0007|model-b",
            (
                7,
                "model-b",
                None,
                None,
            ),
        ),
        (
            "7|",
            (
                7,
                "",
                None,
                None,
            ),
        ),
        (
            "7|model||strategy-only",
            (
                7,
                "model",
                None,
                "strategy-only",
            ),
        ),
    ],
)
def test_parse_selector_valid_matrix(
    value: str,
    expected: tuple[
        int,
        str,
        str | None,
        str | None,
    ],
) -> None:
    module = _module()

    result = module._parse_selector(
        value
    )

    observed = (
        result.round_no,
        result.model_name,
        result.regime_id,
        result.strategy_name,
    )

    assert observed == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        "1",
        "1|model|regime|strategy|extra",
    ],
)
def test_parse_selector_rejects_invalid_field_count(
    value: str,
) -> None:
    module = _module()

    with pytest.raises(
        argparse.ArgumentTypeError,
        match=(
            "selector descriptor must contain "
            "2 to 4 fields"
        ),
    ):
        module._parse_selector(
            value
        )


@pytest.mark.parametrize(
    "value",
    [
        "+1|model",
        "-1|model",
        " 1|model",
        "1 |model",
        "1.0|model",
        "abc|model",
    ],
)
def test_parse_selector_rejects_invalid_round_syntax(
    value: str,
) -> None:
    module = _module()

    with pytest.raises(
        argparse.ArgumentTypeError,
        match=(
            "selector round_no must be an integer"
        ),
    ):
        module._parse_selector(
            value
        )


def test_parser_has_artifact_root_option() -> None:
    action = _option_action(
        "--artifact-root"
    )

    assert action.required is False


def test_parser_has_candidate_selector_option() -> None:
    action = _option_action(
        "--candidate-selector"
    )

    assert action.required is False
    assert action.type is _module()._parse_selector


def test_parser_has_baseline_selector_option() -> None:
    action = _option_action(
        "--baseline-selector"
    )

    assert action.required is False
    assert action.type is _module()._parse_selector


def test_candidate_source_is_no_longer_argparse_required() -> None:
    action = _option_action(
        "--candidate"
    )

    assert action.required is False


def test_baseline_source_is_no_longer_argparse_required() -> None:
    action = _option_action(
        "--baseline"
    )

    assert action.required is False


def test_existing_source_parser_contract_is_preserved() -> None:
    module = _module()

    result = module._parse_source(
        (
            "path.json|0007|model"
            "|regime|strategy"
        )
    )

    assert str(
        result.artifact_path
    ) == "path.json"

    assert result.round_no == 7
    assert result.model_name == "model"
    assert result.regime_id == "regime"
    assert result.strategy_name == "strategy"


def test_existing_source_parser_empty_context_is_preserved() -> None:
    module = _module()

    result = module._parse_source(
        "path.json|7|model||"
    )

    assert result.regime_id is None
    assert result.strategy_name is None


def test_parser_selector_order_is_preserved() -> None:
    parser = _parser()

    args = parser.parse_args(
        [
            "--history",
            "history.json",
            "--window-name",
            "window-a",
            "--start-round",
            "1000",
            "--end-round",
            "1200",
            "--artifact-root",
            "artifacts-root",
            "--candidate-selector",
            "1200|model-c",
            "--candidate-selector",
            "1100|model-b",
            "--candidate-selector",
            "1000|model-a",
            "--baseline-selector",
            "900|base-a",
            "--baseline-selector",
            "950|base-b",
        ]
    )

    assert tuple(
        item.round_no
        for item in args.candidate_selector
    ) == (
        1200,
        1100,
        1000,
    )

    assert tuple(
        item.round_no
        for item in args.baseline_selector
    ) == (
        900,
        950,
    )


def test_selector_mode_builds_exact_composition_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()

    captured: list[Any] = []

    sentinel = object()

    class FakeCompositionService:
        def execute(
            self,
            *,
            request,
        ):
            captured.append(request)
            return sentinel

    monkeypatch.setattr(
        module,
        "DurableReplayCompositionService",
        FakeCompositionService,
    )

    monkeypatch.setattr(
        module,
        "_result_to_dict",
        lambda result: {
            "identity": result is sentinel,
        },
    )

    stdout = _invoke_main(
        monkeypatch,
        [
            "--history",
            "history.json",
            "--window-name",
            "window-a",
            "--start-round",
            "1000",
            "--end-round",
            "1200",
            "--artifact-root",
            "artifact-root",
            "--candidate-selector",
            "1200|candidate-a|regime-a|strategy-a",
            "--candidate-selector",
            "1100|candidate-b",
            "--baseline-selector",
            "900|baseline-a",
        ],
    )

    assert len(captured) == 1

    request = captured[0]

    assert isinstance(
        request,
        module.DurableReplayCompositionRequest,
    )

    assert str(
        request.artifact_root
    ) == "artifact-root"

    assert str(
        request.history_path
    ) == "history.json"

    assert request.window_name == "window-a"
    assert request.start_round == 1000
    assert request.end_round == 1200

    assert tuple(
        (
            item.round_no,
            item.model_name,
            item.regime_id,
            item.strategy_name,
        )
        for item in request.candidate_selectors
    ) == (
        (
            1200,
            "candidate-a",
            "regime-a",
            "strategy-a",
        ),
        (
            1100,
            "candidate-b",
            None,
            None,
        ),
    )

    assert tuple(
        (
            item.round_no,
            item.model_name,
            item.regime_id,
            item.strategy_name,
        )
        for item in request.baseline_selectors
    ) == (
        (
            900,
            "baseline-a",
            None,
            None,
        ),
    )

    payload = json.loads(stdout)

    assert payload == {
        "identity": True,
    }


def test_selector_mode_constructs_composition_service_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()

    builds = 0

    sentinel = object()

    class FakeCompositionService:
        def __init__(
            self,
        ) -> None:
            nonlocal builds
            builds += 1

        def execute(
            self,
            *,
            request,
        ):
            return sentinel

    monkeypatch.setattr(
        module,
        "DurableReplayCompositionService",
        FakeCompositionService,
    )

    monkeypatch.setattr(
        module,
        "_result_to_dict",
        lambda result: {
            "ok": result is sentinel,
        },
    )

    _invoke_main(
        monkeypatch,
        [
            "--history",
            "history.json",
            "--window-name",
            "window-a",
            "--start-round",
            "1",
            "--end-round",
            "10",
            "--artifact-root",
            "root",
            "--candidate-selector",
            "10|model",
        ],
    )

    assert builds == 1


def test_selector_mode_invokes_composition_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()

    executes = 0

    sentinel = object()

    class FakeCompositionService:
        def execute(
            self,
            *,
            request,
        ):
            nonlocal executes
            executes += 1
            return sentinel

    monkeypatch.setattr(
        module,
        "DurableReplayCompositionService",
        FakeCompositionService,
    )

    monkeypatch.setattr(
        module,
        "_result_to_dict",
        lambda result: {
            "ok": result is sentinel,
        },
    )

    _invoke_main(
        monkeypatch,
        [
            "--history",
            "history.json",
            "--window-name",
            "window-a",
            "--start-round",
            "1",
            "--end-round",
            "10",
            "--artifact-root",
            "root",
            "--candidate-selector",
            "10|model",
        ],
    )

    assert executes == 1


def test_selector_mode_does_not_directly_construct_execution_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()

    sentinel = object()

    class ForbiddenExecutionService:
        def __init__(
            self,
        ) -> None:
            raise AssertionError(
                "selector mode must not construct AL service"
            )

    class FakeCompositionService:
        def execute(
            self,
            *,
            request,
        ):
            return sentinel

    monkeypatch.setattr(
        module,
        "DurableReplayExecutionService",
        ForbiddenExecutionService,
    )

    monkeypatch.setattr(
        module,
        "DurableReplayCompositionService",
        FakeCompositionService,
    )

    monkeypatch.setattr(
        module,
        "_result_to_dict",
        lambda result: {
            "ok": result is sentinel,
        },
    )

    _invoke_main(
        monkeypatch,
        [
            "--history",
            "history.json",
            "--window-name",
            "window-a",
            "--start-round",
            "1",
            "--end-round",
            "10",
            "--artifact-root",
            "root",
            "--candidate-selector",
            "10|model",
        ],
    )


def test_explicit_mode_still_invokes_execution_service_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()

    executes = 0
    sentinel = object()

    class FakeExecutionService:
        def execute(
            self,
            *,
            request,
        ):
            nonlocal executes
            executes += 1
            return sentinel

    monkeypatch.setattr(
        module,
        "DurableReplayExecutionService",
        FakeExecutionService,
    )

    monkeypatch.setattr(
        module,
        "_result_to_dict",
        lambda result: {
            "ok": result is sentinel,
        },
    )

    _invoke_main(
        monkeypatch,
        [
            "--history",
            "history.json",
            "--window-name",
            "window-a",
            "--start-round",
            "1",
            "--end-round",
            "10",
            "--candidate",
            "candidate.json|10|model-a",
        ],
    )

    assert executes == 1


def test_explicit_mode_does_not_construct_composition_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()

    sentinel = object()

    class FakeExecutionService:
        def execute(
            self,
            *,
            request,
        ):
            return sentinel

    class ForbiddenCompositionService:
        def __init__(
            self,
        ) -> None:
            raise AssertionError(
                "explicit mode must not construct AO service"
            )

    monkeypatch.setattr(
        module,
        "DurableReplayExecutionService",
        FakeExecutionService,
    )

    monkeypatch.setattr(
        module,
        "DurableReplayCompositionService",
        ForbiddenCompositionService,
    )

    monkeypatch.setattr(
        module,
        "_result_to_dict",
        lambda result: {
            "ok": result is sentinel,
        },
    )

    _invoke_main(
        monkeypatch,
        [
            "--history",
            "history.json",
            "--window-name",
            "window-a",
            "--start-round",
            "1",
            "--end-round",
            "10",
            "--candidate",
            "candidate.json|10|model-a",
        ],
    )


@pytest.mark.parametrize(
    "argv",
    [
        [
            "--history",
            "history.json",
            "--window-name",
            "window-a",
            "--start-round",
            "1",
            "--end-round",
            "10",
        ],
        [
            "--history",
            "history.json",
            "--window-name",
            "window-a",
            "--start-round",
            "1",
            "--end-round",
            "10",
            "--artifact-root",
            "root",
        ],
        [
            "--history",
            "history.json",
            "--window-name",
            "window-a",
            "--start-round",
            "1",
            "--end-round",
            "10",
            "--candidate-selector",
            "10|model",
        ],
        [
            "--history",
            "history.json",
            "--window-name",
            "window-a",
            "--start-round",
            "1",
            "--end-round",
            "10",
            "--artifact-root",
            "root",
            "--candidate",
            "candidate.json|10|model",
        ],
        [
            "--history",
            "history.json",
            "--window-name",
            "window-a",
            "--start-round",
            "1",
            "--end-round",
            "10",
            "--candidate",
            "candidate.json|10|model",
            "--candidate-selector",
            "10|model",
        ],
    ],
)
def test_invalid_mode_combinations_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
) -> None:
    module = _module()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "durable-replay-evaluation",
            *argv,
        ],
    )

    with pytest.raises(
        SystemExit,
    ):
        module.main()


def test_selector_mode_preserves_composition_failure_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()

    failure = RuntimeError(
        "composition-failure"
    )

    class FakeCompositionService:
        def execute(
            self,
            *,
            request,
        ):
            raise failure

    monkeypatch.setattr(
        module,
        "DurableReplayCompositionService",
        FakeCompositionService,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "durable-replay-evaluation",
            "--history",
            "history.json",
            "--window-name",
            "window-a",
            "--start-round",
            "1",
            "--end-round",
            "10",
            "--artifact-root",
            "root",
            "--candidate-selector",
            "10|model",
        ],
    )

    with pytest.raises(
        RuntimeError,
    ) as exc_info:
        module.main()

    assert exc_info.value is failure


def test_explicit_mode_preserves_execution_failure_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()

    failure = RuntimeError(
        "execution-failure"
    )

    class FakeExecutionService:
        def execute(
            self,
            *,
            request,
        ):
            raise failure

    monkeypatch.setattr(
        module,
        "DurableReplayExecutionService",
        FakeExecutionService,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "durable-replay-evaluation",
            "--history",
            "history.json",
            "--window-name",
            "window-a",
            "--start-round",
            "1",
            "--end-round",
            "10",
            "--candidate",
            "candidate.json|10|model",
        ],
    )

    with pytest.raises(
        RuntimeError,
    ) as exc_info:
        module.main()

    assert exc_info.value is failure


def test_product_imports_composition_contract() -> None:
    source = _product_source()
    tree = ast.parse(source)

    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(
            node,
            ast.ImportFrom,
        )
        and node.module is not None
    }

    assert (
        "lrp.operations.durable_replay_composition"
        in imports
    )


def test_product_imports_artifact_selector_contract() -> None:
    source = _product_source()
    tree = ast.parse(source)

    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(
            node,
            ast.ImportFrom,
        )
        and node.module is not None
    }

    assert (
        "lrp.operations.durable_replay_artifact_discovery"
        in imports
    )


def test_product_has_no_direct_filesystem_artifact_io() -> None:
    source = _product_source()
    tree = ast.parse(source)

    calls = {
        ast.unparse(
            node.func
        )
        for node in ast.walk(tree)
        if isinstance(
            node,
            ast.Call,
        )
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
        calls & forbidden
    )


def test_product_has_no_exception_normalization_layer() -> None:
    source = _product_source()
    tree = ast.parse(source)

    handlers = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(
            node,
            ast.ExceptHandler,
        )
    )

    assert handlers == ()


def test_product_public_surface_remains_main_only() -> None:
    module = _module()

    public_functions = tuple(
        name
        for name, value in vars(
            module
        ).items()
        if inspect.isfunction(
            value
        )
        and value.__module__ == MODULE_NAME
        and not name.startswith("_")
    )

    assert public_functions == (
        "main",
    )


def test_root_command_name_is_unchanged() -> None:
    root_source = Path(
        "lrp/cli/__init__.py"
    ).read_text(
        encoding="utf-8-sig"
    )

    assert root_source.count(
        "durable-replay-evaluation"
    ) == 1
