from __future__ import annotations

import importlib
import importlib.util
import inspect
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import get_type_hints

import pytest

from lrp.operations.durable_replay_promotion_publication_request import (
    DurableReplayPromotionPublicationRequest,
)
from lrp.production.champion_registry_publisher import (
    ProductionChampionPublicationResult,
    ProductionChampionRegistryPublisher,
)


MODULE_NAME = "lrp.operations.durable_replay_promotion_publication_execution"


def _module():
    return importlib.import_module(MODULE_NAME)


def _request(
    *,
    action: str = "prepare_publish",
) -> DurableReplayPromotionPublicationRequest:
    return DurableReplayPromotionPublicationRequest(
        status="PASS",
        round_count=1,
        candidate_model_name="candidate-model",
        baseline_model_name="baseline-model",
        recommendation="eligible",
        action=action,
        window=MappingProxyType({"name": "bb02-contract"}),
        source_decision=Path("explicit/source_decision.json"),
        registry_root=Path("explicit/registry_root"),
    )


def test_publication_execution_product_module_exists() -> None:
    assert importlib.util.find_spec(MODULE_NAME) is not None


def test_execution_service_class_exists() -> None:
    module = _module()
    assert hasattr(
        module,
        "DurableReplayPromotionPublicationExecutionService",
    )


def test_execution_public_method_is_execute() -> None:
    module = _module()
    service = module.DurableReplayPromotionPublicationExecutionService
    assert hasattr(service, "execute")
    assert callable(service.execute)


def test_execute_accepts_ba_request_directly() -> None:
    module = _module()
    service = module.DurableReplayPromotionPublicationExecutionService
    signature = inspect.signature(service.execute)
    parameters = list(signature.parameters.values())

    assert [p.name for p in parameters] == ["self", "request"]
    hints = get_type_hints(service.execute)
    assert hints["request"] is DurableReplayPromotionPublicationRequest


def test_execute_return_annotation_is_existing_publication_result() -> None:
    module = _module()
    service = module.DurableReplayPromotionPublicationExecutionService
    hints = get_type_hints(service.execute)
    assert hints["return"] is ProductionChampionPublicationResult


def test_execution_service_owns_or_receives_publisher_dependency() -> None:
    module = _module()
    service = module.DurableReplayPromotionPublicationExecutionService
    source = inspect.getsource(service)
    assert "ProductionChampionRegistryPublisher" in source


def test_execution_delegates_exact_request_identity_to_publisher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    calls: list[dict[str, object]] = []

    expected = ProductionChampionPublicationResult(
        source_path="explicit/source_decision.json",
        source_sha256="a" * 64,
        published_path="explicit/registry_root/champion.json",
        published_at_kst="2026-08-25T13:30:00+09:00",
        selected_model="candidate-model",
    )

    def fake_publish(
        self: ProductionChampionRegistryPublisher,
        *,
        source_decision: str | Path,
        registry_root: str | Path,
    ) -> ProductionChampionPublicationResult:
        calls.append(
            {
                "source_decision": source_decision,
                "registry_root": registry_root,
            }
        )
        return expected

    monkeypatch.setattr(
        ProductionChampionRegistryPublisher,
        "publish",
        fake_publish,
    )

    service = module.DurableReplayPromotionPublicationExecutionService()
    request = _request()
    result = service.execute(request)

    assert calls == [
        {
            "source_decision": request.source_decision,
            "registry_root": request.registry_root,
        }
    ]
    assert result is expected


def test_execution_returns_publisher_result_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()

    expected = ProductionChampionPublicationResult(
        source_path="source.json",
        source_sha256="b" * 64,
        published_path="registry/champion.json",
        published_at_kst="2026-08-25T13:31:00+09:00",
        selected_model="candidate-model",
    )

    monkeypatch.setattr(
        ProductionChampionRegistryPublisher,
        "publish",
        lambda self, *, source_decision, registry_root: expected,
    )

    result = (
        module.DurableReplayPromotionPublicationExecutionService()
        .execute(_request())
    )

    assert result is expected


@pytest.mark.parametrize("action", ["hold", "block"])
def test_execution_requires_prepare_publish(action: str) -> None:
    module = _module()
    with pytest.raises((TypeError, ValueError)):
        (
            module.DurableReplayPromotionPublicationExecutionService()
            .execute(_request(action=action))
        )


def test_publisher_exception_propagates_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()

    class SentinelError(RuntimeError):
        pass

    def fail_publish(
        self: ProductionChampionRegistryPublisher,
        *,
        source_decision: str | Path,
        registry_root: str | Path,
    ) -> ProductionChampionPublicationResult:
        raise SentinelError("publisher failed")

    monkeypatch.setattr(
        ProductionChampionRegistryPublisher,
        "publish",
        fail_publish,
    )

    with pytest.raises(SentinelError, match="publisher failed"):
        (
            module.DurableReplayPromotionPublicationExecutionService()
            .execute(_request())
        )


def test_execution_does_not_mutate_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
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

    expected = ProductionChampionPublicationResult(
        source_path="source.json",
        source_sha256="c" * 64,
        published_path="registry/champion.json",
        published_at_kst="2026-08-25T13:32:00+09:00",
        selected_model="candidate-model",
    )

    monkeypatch.setattr(
        ProductionChampionRegistryPublisher,
        "publish",
        lambda self, *, source_decision, registry_root: expected,
    )

    (
        module.DurableReplayPromotionPublicationExecutionService()
        .execute(request)
    )

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


def test_execution_declares_no_second_result_model() -> None:
    module = _module()
    names = {
        name
        for name in vars(module)
        if name.startswith("DurableReplayPromotionPublicationExecution")
    }
    assert names == {"DurableReplayPromotionPublicationExecutionService"}


def test_execution_has_no_identity_discovery_surface() -> None:
    module = _module()
    source = inspect.getsource(module).lower()

    forbidden = (
        "discover",
        "latest",
        "getenv",
        "environ",
        "default_registry",
        "glob(",
        "rglob(",
    )
    assert all(token not in source for token in forbidden)


def test_execution_has_no_policy_recomputation_or_rollback_surface() -> None:
    module = _module()
    source = inspect.getsource(module).lower()

    forbidden = (
        "candidate_advantage_count",
        "baseline_advantage_count",
        "baseline_delta",
        "eligibility",
        "threshold",
        "rollback",
    )
    assert all(token not in source for token in forbidden)


def test_execution_does_not_duplicate_registry_write_logic() -> None:
    module = _module()
    source = inspect.getsource(module).lower()

    forbidden = (
        "write_text",
        "write_bytes",
        "open(",
        "json.dump",
        "replace(",
        "unlink(",
        "mkdir(",
    )
    assert all(token not in source for token in forbidden)


def test_execution_depends_on_exact_existing_owners() -> None:
    module = _module()
    source = inspect.getsource(module)

    assert (
        "lrp.operations.durable_replay_promotion_publication_request"
        in source
    )
    assert "lrp.production.champion_registry_publisher" in source