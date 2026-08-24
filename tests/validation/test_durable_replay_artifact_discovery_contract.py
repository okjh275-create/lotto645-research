from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from lrp.contracts.exceptions import ContractError
from lrp.operations.durable_replay_execution import (
    DurableReplayExecutionSource,
)


def _product():
    return importlib.import_module(
        "lrp.operations.durable_replay_artifact_discovery"
    )


def _selector(
    *,
    round_no: int = 1234,
    model_name: str = "candidate-model",
    regime_id: str | None = "candidate-regime",
    strategy_name: str | None = "candidate-strategy",
):
    product = _product()

    return product.DurableReplayArtifactSelector(
        round_no=round_no,
        model_name=model_name,
        regime_id=regime_id,
        strategy_name=strategy_name,
    )


def _request():
    product = _product()

    return product.DurableReplayArtifactDiscoveryRequest(
        artifact_root=Path("predictions"),
        candidate_selectors=(
            product.DurableReplayArtifactSelector(
                round_no=1234,
                model_name="candidate-model",
                regime_id="candidate-regime",
                strategy_name="candidate-strategy",
            ),
        ),
        baseline_selectors=(
            product.DurableReplayArtifactSelector(
                round_no=1200,
                model_name="baseline-model",
                regime_id="baseline-regime",
                strategy_name="baseline-strategy",
            ),
        ),
    )


def test_selector_is_frozen() -> None:
    selector = _selector()

    assert dataclasses.is_dataclass(selector)
    assert selector.__dataclass_params__.frozen is True


def test_selector_fields_are_exact() -> None:
    product = _product()

    fields = tuple(
        field.name
        for field in dataclasses.fields(
            product.DurableReplayArtifactSelector
        )
    )

    assert fields == (
        "round_no",
        "model_name",
        "regime_id",
        "strategy_name",
        "artifact_key",
    )


def test_selector_public_signature_is_exact() -> None:
    product = _product()

    assert str(
        inspect.signature(
            product.DurableReplayArtifactSelector
        )
    ) == (
        "(round_no: 'int', model_name: 'str', "
        "regime_id: 'str | None' = None, "
        "strategy_name: 'str | None' = None, "
        "artifact_key: 'str | None' = None) -> None"
    )


def test_request_is_frozen() -> None:
    request = _request()

    assert dataclasses.is_dataclass(request)
    assert request.__dataclass_params__.frozen is True


def test_request_fields_are_exact() -> None:
    product = _product()

    fields = tuple(
        field.name
        for field in dataclasses.fields(
            product.DurableReplayArtifactDiscoveryRequest
        )
    )

    assert fields == (
        "artifact_root",
        "candidate_selectors",
        "baseline_selectors",
    )


def test_request_public_signature_is_exact() -> None:
    product = _product()

    assert str(
        inspect.signature(
            product.DurableReplayArtifactDiscoveryRequest
        )
    ) == (
        "(artifact_root: 'str | Path', "
        "candidate_selectors: "
        "'tuple[DurableReplayArtifactSelector, ...]', "
        "baseline_selectors: "
        "'tuple[DurableReplayArtifactSelector, ...]') -> None"
    )


def test_service_is_parameterless() -> None:
    product = _product()

    assert str(
        inspect.signature(
            product.DurableReplayArtifactDiscoveryService
        )
    ) == "()"


def test_discover_public_signature_is_exact() -> None:
    product = _product()

    assert str(
        inspect.signature(
            product.DurableReplayArtifactDiscoveryService.discover
        )
    ) == (
        "(self, *, request: "
        "'DurableReplayArtifactDiscoveryRequest') -> "
        "'tuple["
        "tuple[DurableReplayExecutionSource, ...], "
        "tuple[DurableReplayExecutionSource, ...]"
        "]'"
    )


def test_discover_returns_candidate_and_baseline_tuples() -> None:
    product = _product()

    service = product.DurableReplayArtifactDiscoveryService()

    candidate, baseline = service.discover(
        request=_request()
    )

    assert isinstance(candidate, tuple)
    assert isinstance(baseline, tuple)

    assert len(candidate) == 1
    assert len(baseline) == 1

    assert isinstance(
        candidate[0],
        DurableReplayExecutionSource,
    )

    assert isinstance(
        baseline[0],
        DurableReplayExecutionSource,
    )


def test_candidate_path_is_canonical() -> None:
    product = _product()

    candidate, _ = (
        product.DurableReplayArtifactDiscoveryService().discover(
            request=_request()
        )
    )

    assert candidate[0].artifact_path == (
        Path("predictions")
        / "prediction-evaluation-sources"
        / "round_1234"
        / "evaluation_source.json"
    )


def test_baseline_path_is_canonical() -> None:
    product = _product()

    _, baseline = (
        product.DurableReplayArtifactDiscoveryService().discover(
            request=_request()
        )
    )

    assert baseline[0].artifact_path == (
        Path("predictions")
        / "prediction-evaluation-sources"
        / "round_1200"
        / "evaluation_source.json"
    )


def test_candidate_context_is_projected_exactly() -> None:
    product = _product()

    candidate, _ = (
        product.DurableReplayArtifactDiscoveryService().discover(
            request=_request()
        )
    )

    assert candidate == (
        DurableReplayExecutionSource(
            artifact_path=(
                Path("predictions")
                / "prediction-evaluation-sources"
                / "round_1234"
                / "evaluation_source.json"
            ),
            round_no=1234,
            model_name="candidate-model",
            regime_id="candidate-regime",
            strategy_name="candidate-strategy",
        ),
    )


def test_baseline_context_is_projected_exactly() -> None:
    product = _product()

    _, baseline = (
        product.DurableReplayArtifactDiscoveryService().discover(
            request=_request()
        )
    )

    assert baseline == (
        DurableReplayExecutionSource(
            artifact_path=(
                Path("predictions")
                / "prediction-evaluation-sources"
                / "round_1200"
                / "evaluation_source.json"
            ),
            round_no=1200,
            model_name="baseline-model",
            regime_id="baseline-regime",
            strategy_name="baseline-strategy",
        ),
    )


def test_candidate_order_is_preserved() -> None:
    product = _product()

    selectors = (
        product.DurableReplayArtifactSelector(
            round_no=1300,
            model_name="model-c",
        ),
        product.DurableReplayArtifactSelector(
            round_no=1100,
            model_name="model-a",
        ),
        product.DurableReplayArtifactSelector(
            round_no=1200,
            model_name="model-b",
        ),
    )

    request = product.DurableReplayArtifactDiscoveryRequest(
        artifact_root="predictions",
        candidate_selectors=selectors,
        baseline_selectors=(),
    )

    candidate, _ = (
        product.DurableReplayArtifactDiscoveryService().discover(
            request=request
        )
    )

    assert tuple(
        item.round_no
        for item in candidate
    ) == (
        1300,
        1100,
        1200,
    )


def test_baseline_order_is_preserved() -> None:
    product = _product()

    selectors = (
        product.DurableReplayArtifactSelector(
            round_no=1250,
            model_name="baseline-b",
        ),
        product.DurableReplayArtifactSelector(
            round_no=1000,
            model_name="baseline-a",
        ),
    )

    request = product.DurableReplayArtifactDiscoveryRequest(
        artifact_root="predictions",
        candidate_selectors=(),
        baseline_selectors=selectors,
    )

    _, baseline = (
        product.DurableReplayArtifactDiscoveryService().discover(
            request=request
        )
    )

    assert tuple(
        item.round_no
        for item in baseline
    ) == (
        1250,
        1000,
    )


def test_duplicate_selectors_are_not_deduplicated() -> None:
    product = _product()

    selector = product.DurableReplayArtifactSelector(
        round_no=1234,
        model_name="same-model",
    )

    request = product.DurableReplayArtifactDiscoveryRequest(
        artifact_root="predictions",
        candidate_selectors=(
            selector,
            selector,
        ),
        baseline_selectors=(),
    )

    candidate, _ = (
        product.DurableReplayArtifactDiscoveryService().discover(
            request=request
        )
    )

    assert len(candidate) == 2
    assert candidate[0] == candidate[1]


def test_optional_context_none_is_preserved() -> None:
    product = _product()

    request = product.DurableReplayArtifactDiscoveryRequest(
        artifact_root="predictions",
        candidate_selectors=(
            product.DurableReplayArtifactSelector(
                round_no=1234,
                model_name="model",
            ),
        ),
        baseline_selectors=(),
    )

    candidate, _ = (
        product.DurableReplayArtifactDiscoveryService().discover(
            request=request
        )
    )

    assert candidate[0].regime_id is None
    assert candidate[0].strategy_name is None


def test_empty_candidate_and_baseline_are_preserved() -> None:
    product = _product()

    request = product.DurableReplayArtifactDiscoveryRequest(
        artifact_root="predictions",
        candidate_selectors=(),
        baseline_selectors=(),
    )

    candidate, baseline = (
        product.DurableReplayArtifactDiscoveryService().discover(
            request=request
        )
    )

    assert candidate == ()
    assert baseline == ()


@pytest.mark.parametrize(
    "value",
    [
        None,
        object(),
        {},
        [],
        (),
        "request",
        123,
        True,
    ],
)
def test_discover_rejects_invalid_request_type(
    value,
) -> None:
    product = _product()

    with pytest.raises(
        ContractError,
        match=(
            "request must be "
            "DurableReplayArtifactDiscoveryRequest"
        ),
    ):
        product.DurableReplayArtifactDiscoveryService().discover(
            request=value
        )


@pytest.mark.parametrize(
    "value",
    [
        None,
        object(),
        {},
        [],
        (),
        "selector",
        123,
        True,
    ],
)
def test_discover_rejects_invalid_candidate_selector_item(
    value,
) -> None:
    product = _product()

    request = product.DurableReplayArtifactDiscoveryRequest(
        artifact_root="predictions",
        candidate_selectors=(value,),
        baseline_selectors=(),
    )

    with pytest.raises(
        ContractError,
        match=(
            "candidate selector must be "
            "DurableReplayArtifactSelector"
        ),
    ):
        product.DurableReplayArtifactDiscoveryService().discover(
            request=request
        )


@pytest.mark.parametrize(
    "value",
    [
        None,
        object(),
        {},
        [],
        (),
        "selector",
        123,
        True,
    ],
)
def test_discover_rejects_invalid_baseline_selector_item(
    value,
) -> None:
    product = _product()

    request = product.DurableReplayArtifactDiscoveryRequest(
        artifact_root="predictions",
        candidate_selectors=(),
        baseline_selectors=(value,),
    )

    with pytest.raises(
        ContractError,
        match=(
            "baseline selector must be "
            "DurableReplayArtifactSelector"
        ),
    ):
        product.DurableReplayArtifactDiscoveryService().discover(
            request=request
        )


def test_discover_does_not_check_artifact_existence() -> None:
    product = _product()

    request = product.DurableReplayArtifactDiscoveryRequest(
        artifact_root=Path(
            "definitely-not-existing-root"
        ),
        candidate_selectors=(
            product.DurableReplayArtifactSelector(
                round_no=9999,
                model_name="model",
            ),
        ),
        baseline_selectors=(),
    )

    candidate, _ = (
        product.DurableReplayArtifactDiscoveryService().discover(
            request=request
        )
    )

    assert candidate[0].artifact_path == (
        Path("definitely-not-existing-root")
        / "prediction-evaluation-sources"
        / "round_9999"
        / "evaluation_source.json"
    )


def test_discover_does_not_mutate_request() -> None:
    product = _product()

    request = _request()
    before = repr(request)

    product.DurableReplayArtifactDiscoveryService().discover(
        request=request
    )

    assert repr(request) == before


def test_discover_is_semantically_deterministic() -> None:
    product = _product()

    service = product.DurableReplayArtifactDiscoveryService()
    request = _request()

    first = service.discover(
        request=request
    )

    second = service.discover(
        request=request
    )

    assert first == second


def test_product_has_no_direct_json_dependency() -> None:
    product = _product()

    path = Path(product.__file__)
    source = path.read_text(
        encoding="utf-8-sig"
    )

    assert "import json" not in source
    assert "from json" not in source


def test_product_has_no_direct_filesystem_read_dependency() -> None:
    product = _product()

    path = Path(product.__file__)
    source = path.read_text(
        encoding="utf-8-sig"
    )

    forbidden = (
        "read_text",
        "read_bytes",
        ".open(",
        "is_file(",
        "exists(",
        "stat(",
    )

    assert not any(
        token in source
        for token in forbidden
    )


def test_product_has_no_filesystem_write_dependency() -> None:
    product = _product()

    path = Path(product.__file__)
    source = path.read_text(
        encoding="utf-8-sig"
    )

    forbidden = (
        "write_text",
        "write_bytes",
        "mkdir(",
        "atomic_write",
        "write_operation_artifact",
        "write_prediction_artifacts",
    )

    assert not any(
        token in source
        for token in forbidden
    )


def test_product_has_no_auto_discovery_dependency() -> None:
    product = _product()

    path = Path(product.__file__)
    source = path.read_text(
        encoding="utf-8-sig"
    )

    forbidden = (
        ".glob(",
        ".rglob(",
        ".iterdir(",
        "latest(",
        "load_latest",
        "discover(",
    )

    # Allow the public method name "discover", but not discovery calls.
    normalized = source.replace(
        "def discover(",
        "def PUBLIC_METHOD(",
    )

    assert not any(
        token in normalized
        for token in forbidden
    )


def test_product_has_no_manifest_dependency() -> None:
    product = _product()

    path = Path(product.__file__)
    source = path.read_text(
        encoding="utf-8-sig"
    )

    assert "manifest" not in source


def test_product_has_no_cli_dependency() -> None:
    product = _product()

    path = Path(product.__file__)
    source = path.read_text(
        encoding="utf-8-sig"
    )

    assert "lrp.cli" not in source
    assert "argparse" not in source


def test_product_has_no_validation_tool_dependency() -> None:
    product = _product()

    path = Path(product.__file__)
    source = path.read_text(
        encoding="utf-8-sig"
    )

    assert "tools.validation" not in source


def test_product_has_no_replay_execution_dependency() -> None:
    product = _product()

    path = Path(product.__file__)
    source = path.read_text(
        encoding="utf-8-sig"
    )

    forbidden = (
        "DurableReplayExecutionService",
        "DurableReplayEvaluationOrchestrator",
        "TopKReplayEvaluationService",
        ".execute(",
        ".evaluate(",
    )

    assert not any(
        token in source
        for token in forbidden
    )


def test_product_public_surface_remains_minimal() -> None:
    product = _product()

    path = Path(product.__file__)
    source = path.read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(source)

    classes = tuple(
        node.name
        for node in tree.body
        if isinstance(
            node,
            ast.ClassDef,
        )
    )

    functions = tuple(
        node.name
        for node in tree.body
        if isinstance(
            node,
            ast.FunctionDef,
        )
    )

    assert classes == (
        "DurableReplayArtifactSelector",
        "DurableReplayArtifactDiscoveryRequest",
        "DurableReplayArtifactDiscoveryService",
    )

    assert functions == ()