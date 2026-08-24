"""Executable RED contract for durable replay result artifact consumer."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Mapping

import pytest


MODULE_NAME = "lrp.operations.durable_replay_result_artifact_consumer"
PRODUCT_PATH = Path(
    "lrp/operations/durable_replay_result_artifact_consumer.py"
)


def _module():
    return importlib.import_module(MODULE_NAME)


def _source() -> str:
    return PRODUCT_PATH.read_text(encoding="utf-8-sig")


def test_consumer_product_module_exists() -> None:
    assert importlib.util.find_spec(MODULE_NAME) is not None
    assert PRODUCT_PATH.exists()


def test_consumer_request_is_frozen_dataclass() -> None:
    module = _module()
    request_type = module.DurableReplayResultArtifactConsumerRequest

    assert is_dataclass(request_type)
    assert request_type.__dataclass_params__.frozen is True


def test_consumer_request_fields_are_exact() -> None:
    module = _module()
    request_type = module.DurableReplayResultArtifactConsumerRequest

    assert tuple(field.name for field in fields(request_type)) == (
        "artifact_root",
        "end_round",
    )


def test_consumer_request_signature_is_exact() -> None:
    module = _module()
    request_type = module.DurableReplayResultArtifactConsumerRequest

    assert str(inspect.signature(request_type)) == (
        "(artifact_root: 'str | Path', end_round: 'int') -> None"
    )


def test_consumer_service_class_exists() -> None:
    module = _module()

    assert hasattr(
        module,
        "DurableReplayResultArtifactConsumer",
    )


def test_consumer_public_method_is_consume() -> None:
    module = _module()
    service = module.DurableReplayResultArtifactConsumer

    assert hasattr(service, "consume")

    signature = str(inspect.signature(service.consume))

    assert signature == (
        "(self, *, request: "
        "'DurableReplayResultArtifactConsumerRequest') "
        "-> 'Mapping[str, Any]'"
    )


def test_consumer_uses_frozen_result_artifact_identity() -> None:
    source = _source()

    assert '"durable-replay-evaluations"' in source
    assert '"evaluation_result.json"' in source
    assert "round_" in source


def test_consumer_reuses_existing_manifest_verifier() -> None:
    source = _source()

    assert "verify_manifest" in source
    assert "lrp.operations.runtime" in source


def test_manifest_verification_precedes_result_json_read() -> None:
    source = _source()
    tree = ast.parse(source)

    consume = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "DurableReplayResultArtifactConsumer"
    )

    method = next(
        node
        for node in consume.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "consume"
    )

    verify_call = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and (
            (
                isinstance(node.func, ast.Name)
                and node.func.id == "verify_manifest"
            )
            or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "verify_manifest"
            )
        )
    )

    read_call = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"read_text", "read_bytes"}
    )

    assert verify_call.lineno < read_call.lineno


def test_consumer_rejects_non_object_top_level_payload() -> None:
    source = _source()

    assert "Mapping" in source
    assert "dict" in source or "isinstance" in source


def test_consumer_has_no_result_mutation_api() -> None:
    source = _source()

    forbidden = (
        "write_text",
        "write_bytes",
        "atomic_write",
        "write_operation_artifact",
        "append_operation_log",
        "unlink(",
        "rename(",
        "replace(",
    )

    assert all(token not in source for token in forbidden)


def test_consumer_has_no_auto_discovery_surface() -> None:
    source = _source()

    forbidden = (
        ".glob(",
        ".rglob(",
        ".iterdir(",
        "latest",
        "candidate_selector",
        "baseline_selector",
        "artifact_key",
    )

    assert all(token not in source for token in forbidden)


def test_consumer_has_no_replay_or_production_dependency() -> None:
    source = _source()
    tree = ast.parse(source)

    imports: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)

    forbidden_prefixes = (
        "lrp.evaluation",
        "lrp.production",
        "lrp.operations.durable_replay_execution",
        "lrp.operations.durable_replay_composition",
    )

    assert not any(
        imported.startswith(prefix)
        for imported in imports
        for prefix in forbidden_prefixes
    )


def test_consumer_return_annotation_is_mapping() -> None:
    module = _module()
    service = module.DurableReplayResultArtifactConsumer

    annotation = inspect.signature(
        service.consume
    ).return_annotation

    assert annotation in {
        Mapping[str, object],
        "Mapping[str, Any]",
    }
