from __future__ import annotations

import ast
import contextlib
import importlib
import inspect
import io
import sys
from pathlib import Path

import pytest

ROOT_MODULE = "lrp.cli"
ROOT_FILE = Path("lrp/cli/__init__.py")
COMMAND = "inspect-replay-invocation"
BH_MODULE = "lrp.cli.durable_replay_publication_invocation_json_file"


def _reload_root():
    module = importlib.import_module(ROOT_MODULE)
    return importlib.reload(module)


def _root_source() -> str:
    return ROOT_FILE.read_text(encoding="utf-8-sig")


def _root_ast() -> ast.Module:
    return ast.parse(_root_source())


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts: list[str] = []
        current: ast.expr = func
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))
    return "<dynamic>"


def test_root_cli_main_signature_remains_exact() -> None:
    root = _reload_root()
    assert str(inspect.signature(root.main)) == "(argv: 'Sequence[str] | None' = None) -> 'int'"


def test_root_cli_registers_exact_inspection_command() -> None:
    source = _root_source()
    assert f'"{COMMAND}"' in source or f"'{COMMAND}'" in source


def test_root_cli_imports_bh_command_owner_only() -> None:
    tree = _root_ast()
    imported_modules: list[str] = []

    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                if node.level:
                    relative = "." * node.level
                    imported_modules.append(
                        f"{relative}{module}.{alias.name}"
                        if module
                        else f"{relative}{alias.name}"
                    )
                else:
                    imported_modules.append(
                        f"{module}.{alias.name}" if module else alias.name
                    )
        elif isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)

    leaf = BH_MODULE.rsplit(".", 1)[-1]
    assert any(
        item == BH_MODULE
        or item == f".{leaf}"
        or item.endswith("." + leaf)
        for item in imported_modules
    )

def test_root_cli_has_no_direct_lower_layer_imports_for_registration() -> None:
    source = _root_source()

    forbidden = [
        "lrp.operations.durable_replay_publication_invocation_json_file_carrier",
        "lrp.operations.durable_replay_publication_invocation_json_presentation",
        "lrp.operations.durable_replay_publication_invocation_transport",
        "lrp.operations.durable_replay_publication_lifecycle_entrypoint",
        "lrp.production",
    ]

    for token in forbidden:
        assert token not in source


def test_root_cli_dispatches_bh_main_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    root = _reload_root()
    bh = importlib.import_module(BH_MODULE)

    observed: list[list[str] | None] = []

    def fake_main(argv=None):
        observed.append(argv)
        return 17

    monkeypatch.setattr(bh, "main", fake_main)

    result = root.main([COMMAND, "--input", r"explicit\carrier.json"])

    assert result == 17
    assert observed == [["--input", r"explicit\carrier.json"]]


def test_root_cli_forwards_remaining_argv_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    root = _reload_root()
    bh = importlib.import_module(BH_MODULE)

    original = [
        "--input",
        r".\folder with spaces\carrier.json",
    ]
    observed = {}

    def fake_main(argv=None):
        observed["argv"] = argv
        return 0

    monkeypatch.setattr(bh, "main", fake_main)

    result = root.main([COMMAND, *original])

    assert result == 0
    assert observed["argv"] == original


def test_root_cli_preserves_path_text_without_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    root = _reload_root()
    bh = importlib.import_module(BH_MODULE)

    samples = [
        r"%USERPROFILE%\carrier.json",
        r"~\carrier.json",
        r".\relative\carrier.json",
    ]

    observed: list[str] = []

    def fake_main(argv=None):
        assert argv is not None
        observed.append(argv[1])
        return 0

    monkeypatch.setattr(bh, "main", fake_main)

    for sample in samples:
        assert root.main([COMMAND, "--input", sample]) == 0

    assert observed == samples


def test_root_cli_passes_bh_exit_code_through_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _reload_root()
    bh = importlib.import_module(BH_MODULE)

    def fake_main(argv=None):
        return 23

    monkeypatch.setattr(bh, "main", fake_main)

    assert root.main([COMMAND, "--input", "carrier.json"]) == 23


def test_root_cli_help_exposes_command_only() -> None:
    root = _reload_root()

    stdout = io.StringIO()

    with pytest.raises(SystemExit) as exc_info:
        with contextlib.redirect_stdout(stdout):
            root.main(["--help"])

    assert exc_info.value.code == 0

    text = stdout.getvalue()
    assert COMMAND in text
    assert "--input" not in text


def test_bh_input_help_remains_bh_owned() -> None:
    bh = importlib.import_module(BH_MODULE)

    stdout = io.StringIO()

    with pytest.raises(SystemExit) as exc_info:
        with contextlib.redirect_stdout(stdout):
            bh.main(["--help"])

    assert exc_info.value.code == 0
    assert "--input" in stdout.getvalue()


def test_bh_argparse_failure_propagates_through_root() -> None:
    root = _reload_root()

    stderr = io.StringIO()

    with pytest.raises(SystemExit) as exc_info:
        with contextlib.redirect_stderr(stderr):
            root.main([COMMAND])

    assert exc_info.value.code == 2
    assert "required" in stderr.getvalue().lower()


def test_existing_root_no_command_behavior_remains_unchanged() -> None:
    root = _reload_root()

    stdout = io.StringIO()

    with contextlib.redirect_stdout(stdout):
        result = root.main([])

    assert result == 0
    assert stdout.getvalue()


def test_root_registration_does_not_construct_transport_or_ba_request() -> None:
    source = _root_source()

    forbidden = [
        "DurableReplayPromotionPublicationRequest(",
        "DurableReplayPublicationInvocationTransport(",
        "DurableReplayPublicationInvocationJsonFileCarrier(",
        "DurableReplayPublicationInvocationJsonCodec(",
    ]

    for token in forbidden:
        assert token not in source


def test_root_registration_has_no_execution_publisher_or_mutation_surface() -> None:
    source = _root_source()

    forbidden = [
        "DurableReplayPublicationLifecycleEntrypoint",
        "ProductionChampionRegistryPublisher",
        "run_publication_stage",
        ".publish(",
        ".run(",
        "write_text",
        "write_bytes",
        "mkdir(",
        "replace(",
        "unlink(",
    ]

    for token in forbidden:
        assert token not in source


def test_root_registration_has_no_discovery_defaulting_or_policy_surface() -> None:
    source = _root_source()

    forbidden = [
        "resolve()",
        "expanduser",
        "getenv",
        "environ",
        "discover",
        "latest",
        "eligibility",
        "promotion_policy",
        "source_decision=",
        "registry_root=",
    ]

    for token in forbidden:
        assert token not in source


def test_root_registration_has_no_packaging_entrypoint_change() -> None:
    candidates = [
        Path("pyproject.toml"),
        Path("setup.cfg"),
        Path("setup.py"),
    ]

    for path in candidates:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8-sig")
        assert "inspect-replay-invocation" not in text


def test_root_registration_ast_calls_do_not_duplicate_bh_logic() -> None:
    tree = _root_ast()

    main = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )

    calls = {
        _call_name(node)
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
    }

    forbidden = {
        "DurableReplayPublicationInvocationJsonFileCarrier",
        "carrier.read",
        "DurableReplayPublicationInvocationJsonCodec",
        "codec.encode",
        "DurableReplayPublicationLifecycleEntrypoint",
    }

    assert calls.isdisjoint(forbidden)


def test_root_registration_preserves_existing_root_cli_import_owners() -> None:
    source = _root_source()

    expected_existing = [
        "production_lifecycle",
        "audit_champion",
        "backup",
        "doctor",
        "export_history",
        "model_evaluation",
        "durable_replay_evaluation",
        "predict",
        "publish_champion",
        "rollback_champion",
        "restore",
        "review",
        "round_complete",
        "status",
        "verify",
        "weekly",
    ]

    for token in expected_existing:
        assert token in source