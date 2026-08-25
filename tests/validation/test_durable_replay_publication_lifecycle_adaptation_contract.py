from __future__ import annotations

import importlib
import importlib.util
import inspect
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import MappingProxyType

import pytest

from lrp.operations.durable_replay_promotion_publication_request import (
    DurableReplayPromotionPublicationRequest,
)
from lrp.operations.durable_replay_promotion_publication_execution import (
    DurableReplayPromotionPublicationExecutionService,
)
from lrp.production.champion_registry_publisher import (
    ProductionChampionPublicationResult,
)
from lrp.production.production_lifecycle import ProductionLifecycleStageResult


MODULE_NAME = "lrp.operations.durable_replay_publication_lifecycle_adaptation"


def _request() -> DurableReplayPromotionPublicationRequest:
    return DurableReplayPromotionPublicationRequest(
        status="PASS",
        round_count=1,
        candidate_model_name="candidate-model",
        baseline_model_name="baseline-model",
        recommendation="eligible",
        action="prepare_publish",
        window=MappingProxyType({"name": "bc02-contract"}),
        source_decision=Path("explicit/source_decision.json"),
        registry_root=Path("explicit/registry_root"),
    )


class _ExecutionStub:
    def __init__(self) -> None:
        self.calls: list[DurableReplayPromotionPublicationRequest] = []
        self.result = ProductionChampionPublicationResult(
            source_path=Path("explicit/source_decision.json"),
            source_sha256="a" * 64,
            published_path=Path(
                "explicit/registry_root/active/champion_decision.json"
            ),
            published_at_kst="2026-08-25T17:00:00+09:00",
            selected_model="candidate-model",
        )

    def execute(
        self,
        request: DurableReplayPromotionPublicationRequest,
    ) -> ProductionChampionPublicationResult:
        self.calls.append(request)
        return self.result


def _module():
    return importlib.import_module(MODULE_NAME)


def _service_class():
    return getattr(
        _module(),
        "DurableReplayPublicationLifecycleAdaptationService",
    )


def test_lifecycle_adaptation_product_module_exists() -> None:
    assert importlib.util.find_spec(MODULE_NAME) is not None


def test_lifecycle_adaptation_service_class_exists() -> None:
    assert hasattr(
        _module(),
        "DurableReplayPublicationLifecycleAdaptationService",
    )


def test_lifecycle_adaptation_public_method_is_adapt() -> None:
    cls = _service_class()
    assert hasattr(cls, "adapt")
    assert callable(cls.adapt)


def test_adapt_accepts_bb_request_directly() -> None:
    sig = inspect.signature(_service_class().adapt)
    params = list(sig.parameters.values())
    assert [param.name for param in params] == ["self", "request"]
    assert (
        params[1].annotation
        in (
            "DurableReplayPromotionPublicationRequest",
            DurableReplayPromotionPublicationRequest,
        )
    )


def test_adapt_return_annotation_is_existing_lifecycle_result() -> None:
    sig = inspect.signature(_service_class().adapt)
    assert (
        sig.return_annotation
        in (
            "ProductionLifecycleStageResult",
            ProductionLifecycleStageResult,
        )
    )


def test_service_owns_or_receives_bb_execution_dependency() -> None:
    sig = inspect.signature(_service_class())
    names = set(sig.parameters)
    assert "execution_service" in names or not names


def test_adaptation_delegates_exact_request_to_bb_execution() -> None:
    stub = _ExecutionStub()
    service = _service_class()(execution_service=stub)
    request = _request()

    service.adapt(request)

    assert stub.calls == [request]


def test_adaptation_calls_bb_execution_exactly_once() -> None:
    stub = _ExecutionStub()
    service = _service_class()(execution_service=stub)

    service.adapt(_request())

    assert len(stub.calls) == 1


def test_adaptation_returns_existing_lifecycle_stage_result() -> None:
    stub = _ExecutionStub()
    service = _service_class()(execution_service=stub)

    result = service.adapt(_request())

    assert isinstance(result, ProductionLifecycleStageResult)


def test_lifecycle_stage_name_is_publication() -> None:
    stub = _ExecutionStub()
    service = _service_class()(execution_service=stub)

    result = service.adapt(_request())

    assert result.name == "publication"


def test_lifecycle_stage_detail_preserves_publication_result_fields() -> None:
    stub = _ExecutionStub()
    service = _service_class()(execution_service=stub)

    result = service.adapt(_request())

    expected = {
        "source_path": stub.result.source_path,
        "source_sha256": stub.result.source_sha256,
        "published_path": stub.result.published_path,
        "published_at_kst": stub.result.published_at_kst,
        "selected_model": stub.result.selected_model,
    }

    for key, value in expected.items():
        assert key in result.detail
        assert result.detail[key] == value


@pytest.mark.parametrize("action", ["hold", "block", "unknown"])
def test_non_prepare_publish_request_fails_before_execution(
    action: str,
) -> None:
    request = _request()
    object.__setattr__(request, "action", action)

    stub = _ExecutionStub()
    service = _service_class()(execution_service=stub)

    with pytest.raises((TypeError, ValueError)):
        service.adapt(request)

    assert stub.calls == []


def test_bb_execution_exception_propagates() -> None:
    sentinel = RuntimeError("bc02 sentinel")

    class FailingExecution:
        def execute(self, request):
            raise sentinel

    service = _service_class()(execution_service=FailingExecution())

    with pytest.raises(RuntimeError) as exc_info:
        service.adapt(_request())

    assert exc_info.value is sentinel


def test_adaptation_does_not_mutate_request() -> None:
    request = _request()
    before = (
        request.status,
        request.round_count,
        request.candidate_model_name,
        request.baseline_model_name,
        request.recommendation,
        request.action,
        dict(request.window),
        request.source_decision,
        request.registry_root,
    )

    stub = _ExecutionStub()
    service = _service_class()(execution_service=stub)
    service.adapt(request)

    after = (
        request.status,
        request.round_count,
        request.candidate_model_name,
        request.baseline_model_name,
        request.recommendation,
        request.action,
        dict(request.window),
        request.source_decision,
        request.registry_root,
    )

    assert after == before


def test_adaptation_declares_no_second_publication_result_model() -> None:
    module = _module()
    names = {
        name
        for name in vars(module)
        if "PublicationResult" in name
        and name != "ProductionChampionPublicationResult"
    }
    assert not names


def test_adaptation_has_no_discovery_policy_cli_or_rollback_surface() -> None:
    source = inspect.getsource(_module()).lower()
    forbidden = (
        "discover",
        "latest",
        "getenv",
        "environ",
        "candidate_advantage_count",
        "baseline_advantage_count",
        "baseline_delta",
        "rollback",
        "lrp.cli",
        "argparse",
    )
    for token in forbidden:
        assert token not in source


def test_adaptation_depends_on_exact_existing_owners() -> None:
    source = inspect.getsource(_module())
    assert "DurableReplayPromotionPublicationRequest" in source
    assert "DurableReplayPromotionPublicationExecutionService" in source
    assert "ProductionChampionPublicationResult" in source
    assert "ProductionLifecycleStageResult" in source