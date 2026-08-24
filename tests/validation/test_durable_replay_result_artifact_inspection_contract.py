from __future__ import annotations

import ast
import importlib.util
import inspect
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, get_type_hints

import pytest


MODULE_NAME = "lrp.operations.durable_replay_result_artifact_inspection"
PRODUCT_PATH = Path("lrp/operations/durable_replay_result_artifact_inspection.py")

EXPECTED_RESULT_FIELDS = (
    "status",
    "round_count",
    "candidate_model_name",
    "baseline_model_name",
    "evaluation",
)


def _module():
    __import__(MODULE_NAME)
    import sys
    return sys.modules[MODULE_NAME]


def _source() -> str:
    return PRODUCT_PATH.read_text(encoding="utf-8-sig")


def _tree() -> ast.Module:
    return ast.parse(_source())


def test_inspection_product_module_exists() -> None:
    assert importlib.util.find_spec(MODULE_NAME) is not None


def test_inspection_result_is_frozen_dataclass() -> None:
    module = _module()
    result_type = module.DurableReplayResultArtifactInspection
    assert is_dataclass(result_type)
    assert result_type.__dataclass_params__.frozen is True


def test_inspection_result_fields_are_exact() -> None:
    module = _module()
    result_type = module.DurableReplayResultArtifactInspection
    assert tuple(field.name for field in fields(result_type)) == EXPECTED_RESULT_FIELDS


def test_inspection_result_signature_is_exact() -> None:
    module = _module()
    result_type = module.DurableReplayResultArtifactInspection
    assert tuple(inspect.signature(result_type).parameters) == EXPECTED_RESULT_FIELDS


def test_inspection_service_class_exists() -> None:
    module = _module()
    assert hasattr(module, "DurableReplayResultArtifactInspectionService")


def test_inspection_public_method_is_inspect() -> None:
    module = _module()
    service = module.DurableReplayResultArtifactInspectionService
    public = [
        name
        for name, value in service.__dict__.items()
        if callable(value) and not name.startswith("_")
    ]
    assert public == ["inspect"]


def test_inspection_method_uses_au_request_directly() -> None:
    module = _module()
    from lrp.operations.durable_replay_result_artifact_consumer import (
        DurableReplayResultArtifactConsumerRequest,
    )

    sig = inspect.signature(
        module.DurableReplayResultArtifactInspectionService.inspect
    )
    params = list(sig.parameters.values())
    assert [p.name for p in params] == ["self", "request"]
    hints = get_type_hints(
        module.DurableReplayResultArtifactInspectionService.inspect
    )
    assert hints["request"] is DurableReplayResultArtifactConsumerRequest


def test_inspection_return_annotation_is_exact_result_type() -> None:
    module = _module()
    hints = get_type_hints(
        module.DurableReplayResultArtifactInspectionService.inspect
    )
    assert hints["return"] is module.DurableReplayResultArtifactInspection


def test_inspection_reuses_au_consumer() -> None:
    source = _source()
    assert "DurableReplayResultArtifactConsumer" in source
    assert ".consume(" in source


def test_inspection_projects_only_frozen_fields() -> None:
    source = _source()
    for field in EXPECTED_RESULT_FIELDS:
        assert field in source
    forbidden = (
        "artifact_path",
        "history_rounds",
        "latest",
        "glob(",
        "rglob(",
        "os.walk",
    )
    for token in forbidden:
        assert token not in source


def test_inspection_has_no_result_mutation_or_writer_dependency() -> None:
    source = _source()
    forbidden = (
        "write_operation_artifact",
        "atomic_write",
        "write_text(",
        "write_bytes(",
        ".mkdir(",
        "open(",
    )
    for token in forbidden:
        assert token not in source


def test_inspection_has_no_replay_or_production_dependency() -> None:
    tree = _tree()
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)

    forbidden_fragments = (
        "durable_replay_consumer",
        "durable_replay_composition",
        "durable_replay_execution",
        "lrp.production",
        "lrp.evaluation",
    )
    for name in imported:
        for fragment in forbidden_fragments:
            assert fragment not in name


def test_evaluation_annotation_is_mapping_compatible() -> None:
    from collections.abc import Mapping as AbcMapping
    from typing import get_origin

    module = _module()
    hints = get_type_hints(
        module.DurableReplayResultArtifactInspection
    )
    annotation = hints["evaluation"]

    assert get_origin(annotation) in (
        AbcMapping,
        dict,
    )


def test_contract_declares_read_only_evaluation_projection() -> None:
    module = _module()
    result = module.DurableReplayResultArtifactInspection(
        status="PASS",
        round_count=1,
        candidate_model_name="candidate-model",
        baseline_model_name="baseline-model",
        evaluation=MappingProxyType({"model_name": "candidate-model"}),
    )
    assert isinstance(result.evaluation, Mapping)
    with pytest.raises(TypeError):
        result.evaluation["x"] = 1  # type: ignore[index]
