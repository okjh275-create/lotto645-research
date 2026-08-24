"""Hardening contracts for durable replay result artifact persistence."""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
from pathlib import Path

import pytest


PRODUCT_MODULE = "lrp.cli.durable_replay_evaluation"


def _module():
    return importlib.import_module(PRODUCT_MODULE)


def _source() -> str:
    module = _module()
    return Path(module.__file__).read_text(
        encoding="utf-8-sig"
    )


def test_writer_call_is_inside_output_guard() -> None:
    source = _source()
    tree = ast.parse(source)

    writer_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "write_operation_artifact"
    )

    guarded = False

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue

        test_text = ast.unparse(node.test)

        if test_text != "args.output is not None":
            continue

        for child in ast.walk(node):
            if child is writer_call:
                guarded = True
                break

    assert guarded is True


def test_writer_call_precedes_stdout_print() -> None:
    source = _source()
    tree = ast.parse(source)

    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "main"
    )

    writer_call = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "write_operation_artifact"
    )

    stdout_print = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
        and node.args
        and isinstance(node.args[0], ast.Call)
        and isinstance(node.args[0].func, ast.Attribute)
        and isinstance(node.args[0].func.value, ast.Name)
        and node.args[0].func.value.id == "json"
        and node.args[0].func.attr == "dumps"
    )

    assert writer_call.lineno < stdout_print.lineno


def test_payload_is_built_before_writer_call() -> None:
    source = _source()
    tree = ast.parse(source)

    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "main"
    )

    payload_assign = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "payload"
            for target in node.targets
        )
    )

    writer_call = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "write_operation_artifact"
    )

    assert payload_assign.lineno < writer_call.lineno


def test_no_direct_filesystem_write_primitive_is_added() -> None:
    source = _source()

    forbidden = (
        "write_text",
        "write_bytes",
        ".mkdir(",
        "open(",
        "atomic_write",
        "append_operation_log",
        "sha256_file",
    )

    assert all(
        token not in source
        for token in forbidden
    )


def test_generic_writer_import_is_exactly_once() -> None:
    source = _source()

    assert source.count(
        "from lrp.operations import write_operation_artifact"
    ) == 1

    assert source.count(
        "write_operation_artifact("
    ) == 1


def test_result_artifact_identity_tokens_are_exactly_once() -> None:
    source = _source()

    assert source.count(
        '"durable-replay-evaluations"'
    ) == 1

    assert source.count(
        '"evaluation_result.json"'
    ) == 1


def test_writer_uses_no_artifact_key_for_result_artifact() -> None:
    source = _source()
    tree = ast.parse(source)

    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "write_operation_artifact"
    )

    keyword_names = tuple(
        keyword.arg
        for keyword in call.keywords
        if keyword.arg is not None
    )

    assert keyword_names == (
        "output_root",
        "artifact_type",
        "round_no",
        "filename",
    )


def test_output_help_is_opt_in_worded() -> None:
    parser = _module()._parser()

    action = next(
        action
        for action in parser._actions
        if action.dest == "output"
    )

    assert action.required is False
    assert action.default is None
    assert action.help is not None
    assert "Optional" in action.help


def test_stdout_sort_keys_contract_is_preserved() -> None:
    source = _source()
    tree = ast.parse(source)

    json_dumps = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "json"
        and node.func.attr == "dumps"
    ]

    assert len(json_dumps) >= 1

    stdout_dump = json_dumps[-1]

    keywords = {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in stdout_dump.keywords
        if keyword.arg is not None
    }

    assert keywords["ensure_ascii"] == "False"
    assert keywords["sort_keys"] == "True"