from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

import lrp.cli as root
import lrp.cli.durable_replay_publication_invocation_execute as bt_a


COMMAND = "execute-replay-invocation"

ROOT_FILE = Path("lrp/cli/__init__.py")
BT_A_FILE = Path(
    "lrp/cli/"
    "durable_replay_publication_invocation_execute.py"
)

INSPECTION_FILE = Path(
    "lrp/cli/"
    "durable_replay_publication_invocation_json_file.py"
)


def _reload_root():
    return importlib.reload(root)


def _root_source() -> str:
    return ROOT_FILE.read_text(encoding="utf-8-sig")


def _root_tree() -> ast.Module:
    return ast.parse(_root_source())


def test_root_registers_execute_replay_invocation() -> None:
    source = _root_source()

    assert '"execute-replay-invocation"' in source


def test_root_imports_bt_a_execution_cli_owner() -> None:
    source = _root_source()

    assert (
        "durable_replay_publication_invocation_execute"
        in source
    )


def test_root_command_table_contains_exact_bt_b_command() -> None:
    root_module = _reload_root()

    assert COMMAND in root_module._COMMANDS


def test_root_dispatches_bt_a_main_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_module = _reload_root()

    observed: list[list[str] | None] = []

    def fake_main(argv=None):
        observed.append(argv)
        return 17

    monkeypatch.setattr(bt_a, "main", fake_main)

    result = root_module.main(
        [
            COMMAND,
            "--input",
            r"explicit\invocation.json",
        ]
    )

    assert result == 17

    assert observed == [
        [
            "--input",
            r"explicit\invocation.json",
        ]
    ]


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
    ],
)
def test_root_forwards_input_text_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    path_text: str,
) -> None:
    root_module = _reload_root()

    observed: list[list[str] | None] = []

    def fake_main(argv=None):
        observed.append(argv)
        return 0

    monkeypatch.setattr(bt_a, "main", fake_main)

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


def test_root_returns_bt_a_exit_code_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_module = _reload_root()

    def fake_main(argv=None):
        assert argv == [
            "--input",
            "invocation.json",
        ]
        return 23

    monkeypatch.setattr(bt_a, "main", fake_main)

    assert (
        root_module.main(
            [
                COMMAND,
                "--input",
                "invocation.json",
            ]
        )
        == 23
    )


def test_root_propagates_bt_a_exception_by_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_module = _reload_root()

    failure = RuntimeError("bt-a failure")

    def fake_main(argv=None):
        assert argv == [
            "--input",
            "invocation.json",
        ]
        raise failure

    monkeypatch.setattr(bt_a, "main", fake_main)

    with pytest.raises(RuntimeError) as exc_info:
        root_module.main(
            [
                COMMAND,
                "--input",
                "invocation.json",
            ]
        )

    assert exc_info.value is failure


def test_root_does_not_parse_bt_a_input_option() -> None:
    root_module = _reload_root()

    with pytest.raises(SystemExit):
        root_module.main(
            [
                "--input",
                "invocation.json",
            ]
        )


def test_root_help_does_not_claim_bt_a_input_ownership(
    capsys: pytest.CaptureFixture[str],
) -> None:
    root_module = _reload_root()

    with pytest.raises(SystemExit):
        root_module.main(["--help"])

    output = capsys.readouterr().out

    assert COMMAND in output
    assert "--input" not in output


def test_bt_a_help_still_owns_input_argument(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        bt_a.main(["--help"])

    output = capsys.readouterr().out

    assert "--input" in output


def test_root_registration_has_no_bs_execution_import() -> None:
    source = _root_source()

    assert (
        "durable_replay_publication_invocation_execution"
        not in source
    )


def test_root_registration_has_no_lower_layer_bt_dependencies() -> None:
    source = _root_source()

    forbidden = (
        "DurableReplayPublicationInvocationExecutionService",
        "DurableReplayPublicationInvocationJsonFileCarrier",
        "DurableReplayPublicationInvocationTransportCodec",
        "DurableReplayPublicationLifecycleAdaptationService",
        "DurableReplayPublicationLifecycleEntrypoint",
    )

    for token in forbidden:
        assert token not in source


def test_bt_b_adapter_has_no_publication_semantics() -> None:
    tree = _root_tree()

    adapters = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and "execute_replay_invocation" in node.name
    ]

    if not adapters:
        pytest.skip(
            "BT-B adapter absent during expected RED phase"
        )

    assert len(adapters) == 1

    source = ast.unparse(adapters[0]).lower()

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


def test_root_registration_has_no_file_or_json_logic() -> None:
    source = _root_source().lower()

    forbidden = (
        "json.load",
        "json.loads",
        "read_text(",
        "read_bytes(",
        "open(",
    )

    for token in forbidden:
        assert token not in source


def test_root_registration_has_no_path_normalization() -> None:
    source = _root_source().lower()

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


def test_root_registration_preserves_inspection_command() -> None:
    root_module = _reload_root()

    assert (
        "inspect-replay-invocation"
        in root_module._COMMANDS
    )


def test_root_registration_uses_existing_remainder_dispatch_model() -> None:
    source = _root_source()

    assert "parse_known_args" in source
    assert "return command(remaining)" in source


def test_root_registration_does_not_add_new_dispatch_style() -> None:
    source = _root_source()

    assert "set_defaults(" not in source


def test_bt_a_source_remains_unmodified_by_registration() -> None:
    source = BT_A_FILE.read_text(
        encoding="utf-8-sig"
    )

    assert (
        "execute-replay-invocation"
        not in source
    )

    assert (
        "lrp.cli.__init__"
        not in source
    )


def test_inspection_cli_remains_independent() -> None:
    source = INSPECTION_FILE.read_text(
        encoding="utf-8-sig"
    )

    assert (
        "durable_replay_publication_invocation_execute"
        not in source
    )


def test_root_adapter_has_exact_one_bt_a_main_call() -> None:
    tree = _root_tree()

    adapters = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and "execute_replay_invocation" in node.name
    ]

    assert len(adapters) == 1

    adapter = adapters[0]

    calls = [
        node
        for node in ast.walk(adapter)
        if isinstance(node, ast.Call)
    ]

    assert len(calls) == 1

    assert ast.unparse(calls[0].func).endswith(
        "durable_replay_publication_invocation_execute.main"
    )


def test_root_adapter_does_not_mutate_forwarded_argv() -> None:
    tree = _root_tree()

    adapter = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and "execute_replay_invocation" in node.name
    )

    assigned = {
        target.id
        for node in ast.walk(adapter)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
        )
        if isinstance(target, ast.Name)
    }

    assert "argv" not in assigned