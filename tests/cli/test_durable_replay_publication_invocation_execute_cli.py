from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
from pathlib import Path

import pytest


MODULE_NAME = (
    "lrp.cli."
    "durable_replay_publication_invocation_execute"
)

PRODUCT_PATH = Path(
    "lrp/cli/"
    "durable_replay_publication_invocation_execute.py"
)

ROOT_CLI_PATH = Path("lrp/cli/__init__.py")

INSPECTION_CLI_PATH = Path(
    "lrp/cli/"
    "durable_replay_publication_invocation_json_file.py"
)


def _module():
    return importlib.import_module(MODULE_NAME)


def test_target_module_exists() -> None:
    assert importlib.util.find_spec(MODULE_NAME) is not None


def test_public_api_is_main_only() -> None:
    module = _module()

    public_functions = {
        name
        for name, value in vars(module).items()
        if inspect.isfunction(value)
        and value.__module__ == MODULE_NAME
        and not name.startswith("_")
    }

    assert public_functions == {"main"}


def test_main_signature_is_exact() -> None:
    module = _module()

    assert str(inspect.signature(module.main)) == (
        "(argv: 'Sequence[str] | None' = None) -> 'int'"
    )


def test_private_parser_exists() -> None:
    module = _module()

    assert hasattr(module, "_parser")
    assert callable(module._parser)


def test_private_execution_service_builder_exists() -> None:
    module = _module()

    assert hasattr(module, "_build_execution_service")
    assert callable(module._build_execution_service)


def test_parser_requires_exact_input_argument() -> None:
    module = _module()
    parser = module._parser()

    args = parser.parse_args(
        [
            "--input",
            "explicit/invocation.json",
        ]
    )

    assert vars(args) == {
        "input": "explicit/invocation.json",
    }


def test_missing_input_is_rejected_by_argparse() -> None:
    module = _module()
    parser = module._parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args([])

    assert exc_info.value.code != 0


@pytest.mark.parametrize(
    "path",
    [
        "",
        " ",
        "invocation.json",
        r".\invocation.json",
        r"..\payload\invocation.json",
        r"%USERPROFILE%\invocation.json",
        "payload/../payload/invocation.json",
    ],
)
def test_main_forwards_explicit_input_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    module = _module()

    class Stub:
        def __init__(self) -> None:
            self.calls: list[object] = []

        def execute(self, actual_path: object) -> object:
            self.calls.append(actual_path)
            return object()

    stub = Stub()

    monkeypatch.setattr(
        module,
        "_build_execution_service",
        lambda: stub,
    )

    exit_code = module.main(
        [
            "--input",
            path,
        ]
    )

    assert exit_code == 0
    assert stub.calls == [path]
    assert stub.calls[0] is path


def test_main_calls_execution_service_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()

    class Stub:
        def __init__(self) -> None:
            self.calls: list[object] = []

        def execute(self, path: object) -> object:
            self.calls.append(path)
            return object()

    stub = Stub()

    build_calls: list[None] = []

    def build():
        build_calls.append(None)
        return stub

    monkeypatch.setattr(
        module,
        "_build_execution_service",
        build,
    )

    actual = module.main(
        [
            "--input",
            "explicit/invocation.json",
        ]
    )

    assert actual == 0
    assert build_calls == [None]
    assert stub.calls == [
        "explicit/invocation.json",
    ]


def test_execution_result_is_not_reinterpreted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()

    sentinel = object()

    class Stub:
        def execute(self, path: object) -> object:
            assert path == "explicit/invocation.json"
            return sentinel

    monkeypatch.setattr(
        module,
        "_build_execution_service",
        lambda: Stub(),
    )

    assert (
        module.main(
            [
                "--input",
                "explicit/invocation.json",
            ]
        )
        == 0
    )


def test_execution_failure_propagates_by_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()

    failure = RuntimeError("bs-owned failure")

    class Stub:
        def execute(self, path: object) -> object:
            assert path == "explicit/invocation.json"
            raise failure

    monkeypatch.setattr(
        module,
        "_build_execution_service",
        lambda: Stub(),
    )

    with pytest.raises(RuntimeError) as exc_info:
        module.main(
            [
                "--input",
                "explicit/invocation.json",
            ]
        )

    assert exc_info.value is failure


def test_product_has_no_exception_translation() -> None:
    tree = ast.parse(
        PRODUCT_PATH.read_text(encoding="utf-8-sig")
    )

    assert not any(
        isinstance(
            node,
            (
                ast.Try,
                ast.Raise,
            ),
        )
        for node in ast.walk(tree)
    )


def test_product_has_no_direct_stdout_or_result_renderer() -> None:
    source = PRODUCT_PATH.read_text(
        encoding="utf-8-sig"
    ).lower()

    forbidden = (
        "sys.stdout",
        "print(",
        "json.dumps",
        "json.dump",
        "presentation",
        "serializer",
        "render",
    )

    for token in forbidden:
        assert token not in source


def test_product_has_no_direct_file_read_or_json_parse() -> None:
    source = PRODUCT_PATH.read_text(
        encoding="utf-8-sig"
    ).lower()

    forbidden = (
        "open(",
        "read_text(",
        "read_bytes(",
        "json.load",
        "json.loads",
    )

    for token in forbidden:
        assert token not in source


def test_product_has_no_path_normalization_or_discovery() -> None:
    source = PRODUCT_PATH.read_text(
        encoding="utf-8-sig"
    ).lower()

    forbidden = (
        "resolve(",
        "absolute(",
        "expanduser",
        "expandvars",
        "getenv",
        "environ",
        "cwd",
        "home(",
        "glob(",
        "rglob(",
        "discover",
        "latest",
    )

    for token in forbidden:
        assert token not in source


def test_product_has_no_direct_publication_or_policy_owner() -> None:
    source = PRODUCT_PATH.read_text(
        encoding="utf-8-sig"
    ).lower()

    forbidden = (
        "productionchampionregistrypublisher",
        "run_publication_stage",
        ".publish(",
        "promotioneligibility",
        "promotionactionplan",
        "candidate_advantage_count",
        "baseline_advantage_count",
        "baseline_delta",
        "rollback",
    )

    for token in forbidden:
        assert token not in source


def test_product_imports_bs_execution_service() -> None:
    tree = ast.parse(
        PRODUCT_PATH.read_text(encoding="utf-8-sig")
    )

    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
    }

    assert (
        "lrp.operations."
        "durable_replay_publication_invocation_execution"
        in imports
    )


def test_main_calls_only_parser_builder_and_execution_service() -> None:
    tree = ast.parse(
        PRODUCT_PATH.read_text(encoding="utf-8-sig")
    )

    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "main"
    )

    calls = [
        ast.unparse(node.func)
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
    ]

    assert calls == [
        "_parser",
        "parser.parse_args",
        "_build_execution_service",
        "service.execute",
    ]


def test_main_does_not_reassign_input() -> None:
    tree = ast.parse(
        PRODUCT_PATH.read_text(encoding="utf-8-sig")
    )

    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "main"
    )

    assigned_names = {
        target.id
        for node in ast.walk(main)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
        )
        if isinstance(target, ast.Name)
    }

    assert "input" not in assigned_names


def test_product_does_not_import_inspection_cli() -> None:
    source = PRODUCT_PATH.read_text(
        encoding="utf-8-sig"
    )

    assert (
        "lrp.cli."
        "durable_replay_publication_invocation_json_file"
        not in source
    )


def test_product_does_not_import_root_cli() -> None:
    source = PRODUCT_PATH.read_text(
        encoding="utf-8-sig"
    )

    assert "lrp.cli.__init__" not in source


def test_bt_a_does_not_own_root_registration() -> None:
    source = PRODUCT_PATH.read_text(
        encoding="utf-8-sig"
    )

    assert "execute-replay-invocation" not in source

    assert "lrp.cli.__init__" not in source

    assert "_COMMANDS" not in source

    assert "parse_known_args" not in source


def test_existing_inspection_cli_remains_read_only() -> None:
    source = INSPECTION_CLI_PATH.read_text(
        encoding="utf-8-sig"
    )

    assert (
        "DurableReplayPublicationInvocationExecutionService"
        not in source
    )

    assert (
        "DurableReplayPublicationLifecycleEntrypoint"
        not in source
    )


def test_product_has_no_stdin_surface() -> None:
    source = PRODUCT_PATH.read_text(
        encoding="utf-8-sig"
    ).lower()

    forbidden = (
        "sys.stdin",
        "stdin",
        "input(",
    )

    for token in forbidden:
        assert token not in source