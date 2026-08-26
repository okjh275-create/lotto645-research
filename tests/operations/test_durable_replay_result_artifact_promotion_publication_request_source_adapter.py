from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import lrp.operations.durable_replay_result_artifact_promotion_publication_request_source_adapter as product
from lrp.operations.durable_replay_promotion_publication_request import (
    DurableReplayPromotionPublicationRequest,
    DurableReplayPromotionPublicationRequestService,
)
from lrp.operations.durable_replay_result_artifact_promotion_action_plan_source_adapter import (
    DurableReplayResultArtifactPromotionActionPlanSourceAdapter,
)


class _SourceAdapterStub:
    def __init__(self, action_plan: object) -> None:
        self.action_plan = action_plan
        self.calls: list[tuple[object, object]] = []

    def adapt(self, artifact_root: object, end_round: object) -> object:
        self.calls.append((artifact_root, end_round))
        return self.action_plan


class _RequestServiceStub:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def build(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        return self.result


def test_module_exposes_expected_adapter_class() -> None:
    assert hasattr(
        product,
        "DurableReplayResultArtifactPromotionPublicationRequestSourceAdapter",
    )


def test_adapter_public_surface_is_exact() -> None:
    cls = product.DurableReplayResultArtifactPromotionPublicationRequestSourceAdapter

    public_methods = {
        name
        for name, value in cls.__dict__.items()
        if callable(value) and not name.startswith("_")
    }

    assert public_methods == {"adapt"}


def test_constructor_signature_is_exact() -> None:
    cls = product.DurableReplayResultArtifactPromotionPublicationRequestSourceAdapter

    assert str(inspect.signature(cls)) == (
        "(source_adapter: "
        "'DurableReplayResultArtifactPromotionActionPlanSourceAdapter | None' = None, "
        "publication_request_service: "
        "'DurableReplayPromotionPublicationRequestService | None' = None) -> 'None'"
    )


def test_adapt_signature_is_exact() -> None:
    method = (
        product
        .DurableReplayResultArtifactPromotionPublicationRequestSourceAdapter
        .adapt
    )

    assert str(inspect.signature(method)) == (
        "(self, artifact_root: 'str | Path', end_round: 'int', *, "
        "source_decision: 'str | Path', registry_root: 'str | Path') "
        "-> 'DurableReplayPromotionPublicationRequest'"
    )


def test_default_dependencies_have_expected_types() -> None:
    adapter = (
        product
        .DurableReplayResultArtifactPromotionPublicationRequestSourceAdapter()
    )

    assert isinstance(
        adapter._source_adapter,
        DurableReplayResultArtifactPromotionActionPlanSourceAdapter,
    )
    assert isinstance(
        adapter._publication_request_service,
        DurableReplayPromotionPublicationRequestService,
    )


def test_adapt_composes_dependencies_exactly_once_and_preserves_identity() -> None:
    action_plan = object()
    publication_request = object()

    source_adapter = _SourceAdapterStub(action_plan)
    request_service = _RequestServiceStub(publication_request)

    adapter = (
        product
        .DurableReplayResultArtifactPromotionPublicationRequestSourceAdapter(
            source_adapter=source_adapter,
            publication_request_service=request_service,
        )
    )

    artifact_root = Path("artifact-root")
    source_decision = Path("decision.json")
    registry_root = Path("registry")

    result = adapter.adapt(
        artifact_root,
        1234,
        source_decision=source_decision,
        registry_root=registry_root,
    )

    assert result is publication_request

    assert source_adapter.calls == [
        (artifact_root, 1234),
    ]

    assert len(request_service.calls) == 1

    call = request_service.calls[0]

    assert set(call) == {
        "action_plan",
        "source_decision",
        "registry_root",
    }

    assert call["action_plan"] is action_plan
    assert call["source_decision"] is source_decision
    assert call["registry_root"] is registry_root


@pytest.mark.parametrize(
    "owner",
    [
        "source_adapter",
        "publication_request_service",
    ],
)
def test_dependency_failures_propagate_by_identity(owner: str) -> None:
    failure = RuntimeError(owner)

    class FailingSource:
        def adapt(self, artifact_root: object, end_round: object) -> object:
            raise failure

    class PassingSource:
        def adapt(self, artifact_root: object, end_round: object) -> object:
            return object()

    class FailingService:
        def build(self, **kwargs: object) -> object:
            raise failure

    class PassingService:
        def build(self, **kwargs: object) -> object:
            return object()

    if owner == "source_adapter":
        source = FailingSource()
        service = PassingService()
    else:
        source = PassingSource()
        service = FailingService()

    adapter = (
        product
        .DurableReplayResultArtifactPromotionPublicationRequestSourceAdapter(
            source_adapter=source,
            publication_request_service=service,
        )
    )

    with pytest.raises(RuntimeError) as exc_info:
        adapter.adapt(
            "artifact-root",
            1234,
            source_decision="decision.json",
            registry_root="registry",
        )

    assert exc_info.value is failure


def test_product_has_exact_operational_imports() -> None:
    source = inspect.getsource(product)
    tree = ast.parse(source)

    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith("lrp.operations.")
    }

    assert imports == {
        (
            "lrp.operations."
            "durable_replay_result_artifact_promotion_action_plan_source_adapter"
        ),
        "lrp.operations.durable_replay_promotion_publication_request",
    }


def test_product_has_one_class_only() -> None:
    tree = ast.parse(inspect.getsource(product))

    classes = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    ]

    assert classes == [
        "DurableReplayResultArtifactPromotionPublicationRequestSourceAdapter"
    ]


def test_product_has_no_forbidden_operational_surface() -> None:
    source = inspect.getsource(product)

    forbidden = (
        "resolve()",
        "expanduser",
        "getenv(",
        "environ",
        "discover",
        "latest",
        "open(",
        "write_text",
        "write_bytes",
        "unlink(",
        "replace(",
        "rename(",
        "rollback",
        "publish(",
        "execute(",
    )

    for token in forbidden:
        assert token not in source
