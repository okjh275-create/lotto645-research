from __future__ import annotations

import ast
import contextlib
import importlib
import io
from pathlib import Path

import pytest

ROOT_MODULE = "lrp.cli"
ROOT_FILE = Path("lrp/cli/__init__.py")
BH_MODULE = "lrp.cli.durable_replay_publication_invocation_json_file"
COMMAND = "inspect-replay-invocation"
ADAPTER = "_inspect_replay_invocation_main"


def _reload_root():
    module = importlib.import_module(ROOT_MODULE)
    return importlib.reload(module)


def _source() -> str:
    return ROOT_FILE.read_text(encoding="utf-8-sig")


def _tree() -> ast.Module:
    return ast.parse(_source())


@pytest.mark.parametrize(
    "path_text",
    [
        r"relative\carrier.json",
        r".\relative\carrier.json",
        r"folder with spaces\carrier.json",
        r"%USERPROFILE%\carrier.json",
        r"~\carrier.json",
        r"..\relative\carrier.json",
    ],
)
def test_root_forwards_input_path_text_exactly(
    monkeypatch: pytest.MonkeyPatch,
    path_text: str,
) -> None:
    root = _reload_root()
    bh = importlib.import_module(BH_MODULE)

    observed = []

    def fake_main(argv=None):
        observed.append(argv)
        return 0

    monkeypatch.setattr(bh, "main", fake_main)

    assert root.main([COMMAND, "--input", path_text]) == 0
    assert observed == [["--input", path_text]]


def test_root_calls_live_bh_main_once_per_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _reload_root()
    bh = importlib.import_module(BH_MODULE)

    observed = []

    def first(argv=None):
        observed.append(("first", argv))
        return 11

    def second(argv=None):
        observed.append(("second", argv))
        return 12

    monkeypatch.setattr(bh, "main", first)
    assert root.main([COMMAND, "--input", "a.json"]) == 11

    monkeypatch.setattr(bh, "main", second)
    assert root.main([COMMAND, "--input", "b.json"]) == 12

    assert observed == [
        ("first", ["--input", "a.json"]),
        ("second", ["--input", "b.json"]),
    ]


@pytest.mark.parametrize("exit_code", [0, 1, 7, 23, 127])
def test_root_returns_bh_exit_code_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    exit_code: int,
) -> None:
    root = _reload_root()
    bh = importlib.import_module(BH_MODULE)

    def fake_main(argv=None):
        return exit_code

    monkeypatch.setattr(bh, "main", fake_main)

    assert root.main([COMMAND, "--input", "carrier.json"]) == exit_code


def test_root_help_lists_command_without_bh_option_duplication() -> None:
    root = _reload_root()
    stdout = io.StringIO()

    with pytest.raises(SystemExit) as exc_info:
        with contextlib.redirect_stdout(stdout):
            root.main(["--help"])

    assert exc_info.value.code == 0
    text = stdout.getvalue()
    assert COMMAND in text
    assert "--input" not in text


def test_bh_help_still_owns_input_option() -> None:
    bh = importlib.import_module(BH_MODULE)
    stdout = io.StringIO()

    with pytest.raises(SystemExit) as exc_info:
        with contextlib.redirect_stdout(stdout):
            bh.main(["--help"])

    assert exc_info.value.code == 0
    assert "--input" in stdout.getvalue()


def test_missing_bh_input_propagates_argparse_failure() -> None:
    root = _reload_root()
    stderr = io.StringIO()

    with pytest.raises(SystemExit) as exc_info:
        with contextlib.redirect_stderr(stderr):
            root.main([COMMAND])

    assert exc_info.value.code == 2
    assert "required" in stderr.getvalue().lower()


def test_unknown_bh_option_propagates_argparse_failure() -> None:
    root = _reload_root()
    stderr = io.StringIO()

    with pytest.raises(SystemExit) as exc_info:
        with contextlib.redirect_stderr(stderr):
            root.main([COMMAND, "--unknown"])

    assert exc_info.value.code == 2


def test_bh_domain_failure_identity_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _reload_root()
    bh = importlib.import_module(BH_MODULE)

    error = ValueError("bh-domain-failure")

    def fake_main(argv=None):
        raise error

    monkeypatch.setattr(bh, "main", fake_main)

    with pytest.raises(ValueError) as exc_info:
        root.main([COMMAND, "--input", "carrier.json"])

    assert exc_info.value is error


def test_bh_file_failure_identity_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _reload_root()
    bh = importlib.import_module(BH_MODULE)

    error = FileNotFoundError("carrier.json")

    def fake_main(argv=None):
        raise error

    monkeypatch.setattr(bh, "main", fake_main)

    with pytest.raises(FileNotFoundError) as exc_info:
        root.main([COMMAND, "--input", "carrier.json"])

    assert exc_info.value is error


def test_no_command_baseline_remains_help_plus_zero() -> None:
    root = _reload_root()
    stdout = io.StringIO()

    with contextlib.redirect_stdout(stdout):
        result = root.main([])

    assert result == 0
    assert stdout.getvalue()


def test_existing_root_command_map_is_preserved() -> None:
    source = _source()

    expected = [
        "production-lifecycle",
        "audit-champion",
        "backup",
        "doctor",
        "export-history",
        "model-evaluation",
        "durable-replay-evaluation",
        "predict",
        "publish-champion",
        "rollback-champion",
        "restore",
        "review",
        "round-complete",
        "status",
        "verify",
        "weekly",
    ]

    for command in expected:
        assert command in source


def test_registration_adapter_has_exact_one_call() -> None:
    tree = _tree()
    adapter = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == ADAPTER
    )

    calls = [node for node in ast.walk(adapter) if isinstance(node, ast.Call)]

    assert len(calls) == 1

    call = calls[0]
    assert isinstance(call.func, ast.Attribute)
    assert call.func.attr == "main"
    assert isinstance(call.func.value, ast.Name)
    assert call.func.value.id == "durable_replay_publication_invocation_json_file"


def test_registration_adapter_does_not_mutate_or_rewrite_argv() -> None:
    tree = _tree()
    adapter = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == ADAPTER
    )

    forbidden_call_names = {
        "list",
        "tuple",
        "Path",
        "str",
        "resolve",
        "expanduser",
        "getenv",
    }

    for node in ast.walk(adapter):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden_call_names
            elif isinstance(node.func, ast.Attribute):
                assert node.func.attr not in forbidden_call_names


def test_registration_has_no_direct_lower_layer_dependency() -> None:
    tree = _tree()

    forbidden_fragments = [
        "durable_replay_publication_invocation_json_file_carrier",
        "durable_replay_publication_invocation_json_presentation",
        "durable_replay_publication_invocation_transport",
        "durable_replay_publication_lifecycle_entrypoint",
        "champion_registry_publisher",
    ]

    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imported.append(module)
            imported.extend(alias.name for alias in node.names)

    for item in imported:
        for fragment in forbidden_fragments:
            assert fragment not in item


def test_registration_has_no_new_file_io_or_execution_surface() -> None:
    tree = _tree()
    adapter = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == ADAPTER
    )

    forbidden = {
        "open",
        "read_text",
        "read_bytes",
        "write_text",
        "write_bytes",
        "mkdir",
        "unlink",
        "replace",
        "publish",
        "run",
    }

    for node in ast.walk(adapter):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            assert node.func.id not in forbidden
        elif isinstance(node.func, ast.Attribute):
            assert node.func.attr not in forbidden


def test_registration_has_no_path_discovery_or_policy_surface() -> None:
    source = _source()

    forbidden_new = [
        "expanduser(",
        "getenv(",
        "os.environ",
        "source_decision=",
        "registry_root=",
        "eligibility=",
        "promotion_policy=",
    ]

    for token in forbidden_new:
        assert token not in source


def test_registration_does_not_change_packaging_metadata() -> None:
    for path in [
        Path("pyproject.toml"),
        Path("setup.cfg"),
        Path("setup.py"),
    ]:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8-sig")
        assert COMMAND not in text