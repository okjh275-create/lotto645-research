"""Executable RED contract for durable replay result artifact persistence."""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
from pathlib import Path


PRODUCT_MODULE = "lrp.cli.durable_replay_evaluation"


def _module():
    return importlib.import_module(PRODUCT_MODULE)


def _product_source() -> str:
    module = _module()
    return Path(module.__file__).read_text(encoding="utf-8-sig")


def _parser() -> argparse.ArgumentParser:
    return _module()._parser()


def _option_actions() -> dict[str, argparse.Action]:
    parser = _parser()
    return {
        option: action
        for action in parser._actions
        for option in action.option_strings
    }


def test_parser_has_optional_output_option() -> None:
    actions = _option_actions()
    assert "--output" in actions
    assert actions["--output"].required is False
    assert actions["--output"].default is None


def test_main_public_signature_remains_unchanged() -> None:
    module = _module()
    assert str(inspect.signature(module.main)) == (
        "(argv: 'Sequence[str] | None' = None) -> 'int'"
    )


def test_product_reuses_existing_operation_artifact_writer() -> None:
    source = _product_source()
    tree = ast.parse(source)

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.append(
                (node.module, tuple(alias.name for alias in node.names))
            )

    assert any(
        module == "lrp.operations"
        and "write_operation_artifact" in names
        for module, names in imports
    )


def test_product_contains_exact_result_artifact_identity() -> None:
    source = _product_source()
    assert '"durable-replay-evaluations"' in source
    assert '"evaluation_result.json"' in source


def test_result_artifact_uses_end_round_partition() -> None:
    source = _product_source()
    assert "round_no=args.end_round" in source


def test_result_artifact_uses_cli_output_root() -> None:
    source = _product_source()
    assert "output_root=args.output" in source


def test_result_artifact_persistence_is_opt_in() -> None:
    source = _product_source()
    assert "if args.output is not None:" in source


def test_result_artifact_payload_reuses_stdout_payload() -> None:
    source = _product_source()
    tree = ast.parse(source)

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "write_operation_artifact"
    ]

    assert len(calls) == 1
    call = calls[0]
    assert len(call.args) == 1
    assert isinstance(call.args[0], ast.Name)
    assert call.args[0].id == "payload"


def test_result_artifact_writer_keyword_contract_is_exact() -> None:
    source = _product_source()
    tree = ast.parse(source)

    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "write_operation_artifact"
    )

    keywords = {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in call.keywords
        if keyword.arg is not None
    }

    assert keywords == {
        "output_root": "args.output",
        "artifact_type": "'durable-replay-evaluations'",
        "round_no": "args.end_round",
        "filename": "'evaluation_result.json'",
    }


def test_stdout_json_dump_remains_present() -> None:
    source = _product_source()
    assert "json.dumps(" in source
    assert "sort_keys=True" in source


def test_no_output_path_parses_as_none() -> None:
    parser = _parser()
    namespace = parser.parse_args(
        [
            "--history", "history.json",
            "--window-name", "window",
            "--start-round", "1",
            "--end-round", "1",
            "--candidate", "candidate.json|1|candidate-model",
            "--baseline", "baseline.json|1|baseline-model",
        ]
    )
    assert namespace.output is None


def test_output_option_accepts_explicit_path() -> None:
    parser = _parser()
    namespace = parser.parse_args(
        [
            "--history", "history.json",
            "--window-name", "window",
            "--start-round", "1",
            "--end-round", "1",
            "--candidate", "candidate.json|1|candidate-model",
            "--baseline", "baseline.json|1|baseline-model",
            "--output", "artifacts/out",
        ]
    )
    assert namespace.output == "artifacts/out"