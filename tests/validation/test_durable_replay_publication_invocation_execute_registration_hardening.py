from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

import lrp.cli as root
import lrp.cli.durable_replay_publication_invocation_execute as bt_a


COMMAND = "execute-replay-invocation"

ROOT_PATH = Path("lrp/cli/__init__.py")

BT_A_PATH = Path(
    "lrp/cli/"
    "durable_replay_publication_invocation_execute.py"
)

INSPECTION_PATH = Path(
    "lrp/cli/"
    "durable_replay_publication_invocation_json_file.py"
)


def _reload_root():
    return importlib.reload(root)


def _root_source() -> str:
    return ROOT_PATH.read_text(
        encoding="utf-8-sig"
    )


def _root_tree() -> ast.Module:
    return ast.parse(_root_source())


def _adapter() -> ast.FunctionDef:
    tree = _root_tree()

    adapters = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        == "_execute_replay_invocation_main"
    ]

    assert len(adapters) == 1

    return adapters[0]


def test_root_command_count_is_exactly_eighteen() -> None:
    root_module = _reload_root()

    assert len(root_module._COMMANDS) == 18


def test_root_command_set_is_exactly_frozen() -> None:
    root_module = _reload_root()

    expected = {
        "predict",
        "weekly",
        "review",
        "round-complete",
        "verify",
        "backup",
        "restore",
        "status",
        "doctor",
        "export-history",
        "publish-champion",
        "audit-champion",
        "model-evaluation",
        "durable-replay-evaluation",
        "rollback-champion",
        "production-lifecycle",
        "inspect-replay-invocation",
        "execute-replay-invocation",
    }

    assert set(root_module._COMMANDS) == expected


def test_execute_command_handler_identity_is_exact() -> None:
    root_module = _reload_root()

    assert (
        root_module._COMMANDS[COMMAND]
        is root_module._execute_replay_invocation_main
    )


def test_inspection_handler_identity_is_preserved() -> None:
    root_module = _reload_root()

    assert (
        root_module._COMMANDS[
            "inspect-replay-invocation"
        ]
        is root_module._inspect_replay_invocation_main
    )


def test_execute_root_adapter_has_exact_one_call() -> None:
    adapter = _adapter()

    calls = [
        ast.unparse(node.func)
        for node in ast.walk(adapter)
        if isinstance(node, ast.Call)
    ]

    assert calls == [
        "durable_replay_publication_invocation_execute.main",
    ]


def test_execute_root_adapter_returns_bt_a_directly() -> None:
    adapter = _adapter()

    returns = [
        node
        for node in ast.walk(adapter)
        if isinstance(node, ast.Return)
    ]

    assert len(returns) == 1

    value = returns[0].value

    assert isinstance(value, ast.Call)

    assert ast.unparse(value.func) == (
        "durable_replay_publication_invocation_execute.main"
    )

    assert len(value.args) == 1

    assert isinstance(value.args[0], ast.Name)

    assert value.args[0].id == "argv"


def test_execute_root_adapter_does_not_rewrite_argv() -> None:
    adapter = _adapter()

    assigned_names = {
        target.id
        for node in ast.walk(adapter)
        if isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
            ),
        )
        for target in (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
        )
        if isinstance(target, ast.Name)
    }

    assert "argv" not in assigned_names


def test_execute_root_adapter_has_no_exception_translation() -> None:
    adapter = _adapter()

    assert not any(
        isinstance(
            node,
            (
                ast.Try,
                ast.Raise,
                ast.With,
                ast.AsyncWith,
            ),
        )
        for node in ast.walk(adapter)
    )


@pytest.mark.parametrize(
    "path_text",
    [
        "",
        " ",
        "invocation.json",
        r".\invocation.json",
        r"..\payload\invocation.json",
        r"%USERPROFILE%\invocation.json",
        "payload/../payload/invocation.json",
        "./payload/invocation.json",
        "payload//invocation.json",
    ],
)
def test_root_preserves_remaining_argv_exactly(
    monkeypatch: pytest.MonkeyPatch,
    path_text: str,
) -> None:
    root_module = _reload_root()

    observed: list[list[str] | None] = []

    def fake_main(argv=None):
        observed.append(argv)
        return 0

    monkeypatch.setattr(
        bt_a,
        "main",
        fake_main,
    )

    result = root_module.main(
        [
            COMMAND,
            "--input",
            path_text,
        ]
    )

    assert result == 0

    assert observed == [
        [
            "--input",
            path_text,
        ]
    ]


def test_bt_a_main_is_called_once_per_root_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_module = _reload_root()

    calls: list[list[str] | None] = []

    def fake_main(argv=None):
        calls.append(argv)
        return 0

    monkeypatch.setattr(
        bt_a,
        "main",
        fake_main,
    )

    result = root_module.main(
        [
            COMMAND,
            "--input",
            "invocation.json",
        ]
    )

    assert result == 0

    assert calls == [
        [
            "--input",
            "invocation.json",
        ]
    ]


@pytest.mark.parametrize(
    "exit_code",
    [
        0,
        1,
        7,
        23,
        255,
    ],
)
def test_root_propagates_bt_a_exit_code_exactly(
    monkeypatch: pytest.MonkeyPatch,
    exit_code: int,
) -> None:
    root_module = _reload_root()

    def fake_main(argv=None):
        assert argv == [
            "--input",
            "invocation.json",
        ]

        return exit_code

    monkeypatch.setattr(
        bt_a,
        "main",
        fake_main,
    )

    assert (
        root_module.main(
            [
                COMMAND,
                "--input",
                "invocation.json",
            ]
        )
        == exit_code
    )


def test_root_propagates_bt_a_exception_by_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_module = _reload_root()

    failure = RuntimeError(
        "bt-a execution failure"
    )

    def fake_main(argv=None):
        assert argv == [
            "--input",
            "invocation.json",
        ]

        raise failure

    monkeypatch.setattr(
        bt_a,
        "main",
        fake_main,
    )

    with pytest.raises(
        RuntimeError
    ) as exc_info:
        root_module.main(
            [
                COMMAND,
                "--input",
                "invocation.json",
            ]
        )

    assert exc_info.value is failure


def test_root_does_not_own_input_parser() -> None:
    source = _root_source()
    tree = _root_tree()

    input_option_owners = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and any(
            isinstance(argument, ast.Constant)
            and argument.value == "--input"
            for argument in node.args
        )
    ]

    assert input_option_owners == []

    assert '"execute-replay-invocation"' in source
    assert "parse_known_args" in source


def test_root_help_does_not_absorb_bt_a_input(
    capsys: pytest.CaptureFixture[str],
) -> None:
    root_module = _reload_root()

    with pytest.raises(SystemExit):
        root_module.main(["--help"])

    output = capsys.readouterr().out

    assert COMMAND in output

    assert "--input" not in output


def test_bt_a_remains_owner_of_input_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        bt_a.main(["--help"])

    output = capsys.readouterr().out

    assert "--input" in output


def test_root_has_no_bs_execution_service_import() -> None:
    source = _root_source()

    assert (
        "DurableReplayPublicationInvocationExecutionService"
        not in source
    )

    assert (
        "durable_replay_publication_invocation_execution"
        not in source
    )


def test_root_has_no_bt_a_lower_layer_dependencies() -> None:
    source = _root_source()

    forbidden = (
        "DurableReplayPublicationInvocationJsonFileCarrier",
        "DurableReplayPublicationInvocationTransportCodec",
        "DurableReplayPublicationLifecycleAdaptationService",
        "DurableReplayPublicationLifecycleEntrypoint",
    )

    for token in forbidden:
        assert token not in source


def test_execute_adapter_has_no_publication_policy() -> None:
    source = ast.unparse(
        _adapter()
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


def test_execute_adapter_has_no_file_or_json_logic() -> None:
    source = ast.unparse(
        _adapter()
    ).lower()

    forbidden = (
        "open(",
        "read_text(",
        "read_bytes(",
        "json.load",
        "json.loads",
        "json.dump",
        "json.dumps",
    )

    for token in forbidden:
        assert token not in source


def test_execute_adapter_has_no_path_normalization() -> None:
    source = ast.unparse(
        _adapter()
    ).lower()

    forbidden = (
        "resolve(",
        "absolute(",
        "expanduser",
        "expandvars",
        "getenv",
        "environ",
        "glob(",
        "rglob(",
        "discover",
        "latest",
    )

    for token in forbidden:
        assert token not in source


def test_execute_adapter_has_no_stdout_stdin_surface() -> None:
    source = ast.unparse(
        _adapter()
    ).lower()

    forbidden = (
        "sys.stdout",
        "sys.stdin",
        "print(",
        "input(",
    )

    for token in forbidden:
        assert token not in source


def test_bt_a_product_does_not_own_root_dispatcher() -> None:
    source = BT_A_PATH.read_text(
        encoding="utf-8-sig"
    )

    forbidden = (
        "execute-replay-invocation",
        "lrp.cli.__init__",
        "_COMMANDS",
        "parse_known_args",
    )

    for token in forbidden:
        assert token not in source


def test_inspection_cli_remains_independent() -> None:
    source = INSPECTION_PATH.read_text(
        encoding="utf-8-sig"
    )

    assert (
        "durable_replay_publication_invocation_execute"
        not in source
    )

    assert (
        "DurableReplayPublicationInvocationExecutionService"
        not in source
    )


def test_root_uses_existing_remainder_dispatch_model() -> None:
    source = _root_source()

    assert "parse_known_args" in source

    assert "return command(remaining)" in source

    assert "set_defaults(" not in source


def test_execute_command_literal_count_is_exact() -> None:
    source = _root_source()

    assert source.count(
        '"execute-replay-invocation"'
    ) == 2


def test_execute_root_adapter_definition_count_is_exact() -> None:
    tree = _root_tree()

    assert (
        sum(
            1
            for node in tree.body
            if isinstance(
                node,
                ast.FunctionDef,
            )
            and node.name
            == "_execute_replay_invocation_main"
        )
        == 1
    )


def test_execute_bt_a_module_import_is_exactly_once() -> None:
    source = _root_source()

    assert source.count(
        "from . import "
        "durable_replay_publication_invocation_execute"
    ) == 1


def test_inspection_registration_is_still_present() -> None:
    root_module = _reload_root()

    assert (
        "inspect-replay-invocation"
        in root_module._COMMANDS
    )


def test_unrelated_existing_commands_remain_present() -> None:
    root_module = _reload_root()

    required = {
        "predict",
        "weekly",
        "review",
        "round-complete",
        "verify",
        "backup",
        "restore",
        "status",
        "doctor",
        "export-history",
        "publish-champion",
        "audit-champion",
        "model-evaluation",
        "durable-replay-evaluation",
        "rollback-champion",
        "production-lifecycle",
    }

    assert required.issubset(
        root_module._COMMANDS
    )