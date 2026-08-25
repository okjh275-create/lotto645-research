from __future__ import annotations

import ast
import dataclasses
import importlib
import importlib.util
import inspect
from pathlib import Path
from types import MappingProxyType
from typing import get_type_hints

import pytest

from lrp.operations.durable_replay_promotion_publication_request import (
    DurableReplayPromotionPublicationRequest,
)
from lrp.production.production_lifecycle import (
    ProductionLifecycleStageResult,
)


MODULE_NAME = "lrp.operations.durable_replay_publication_lifecycle_entrypoint"
SERVICE_NAME = "DurableReplayPublicationLifecycleEntrypoint"
MODULE_PATH = Path(
    "lrp/operations/durable_replay_publication_lifecycle_entrypoint.py"
)


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
        window=MappingProxyType({"name": "bd02"}),
        source_decision=Path("explicit/source_decision.json"),
        registry_root=Path("explicit/registry_root"),
    )


class _AdaptationStub:
    def __init__(
        self,
        *,
        result: ProductionLifecycleStageResult | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.calls: list[DurableReplayPromotionPublicationRequest] = []
        self.result = result or ProductionLifecycleStageResult(
            name="publication",
            status="PASS",
            detail={
                "source_path": Path("explicit/source_decision.json"),
                "source_sha256": "a" * 64,
                "published_path": Path(
                    "explicit/registry_root/active/champion_decision.json"
                ),
                "published_at_kst": "2026-08-25T19:45:00+09:00",
                "selected_model": "candidate-model",
            },
        )
        self.error = error

    def adapt(
        self,
        request: DurableReplayPromotionPublicationRequest,
    ) -> ProductionLifecycleStageResult:
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        return self.result


def _module():
    return importlib.import_module(MODULE_NAME)


def _service_class():
    return getattr(_module(), SERVICE_NAME)


def test_lifecycle_entrypoint_product_module_exists() -> None:
    assert importlib.util.find_spec(MODULE_NAME) is not None


def test_entrypoint_service_class_exists() -> None:
    assert inspect.isclass(_service_class())


def test_entrypoint_public_method_is_run() -> None:
    cls = _service_class()
    public = [
        name
        for name, value in cls.__dict__.items()
        if callable(value) and not name.startswith("_")
    ]
    assert public == ["run"]


def test_run_accepts_typed_request_directly() -> None:
    sig = inspect.signature(_service_class().run)
    params = list(sig.parameters.values())
    assert [p.name for p in params] == ["self", "request"]

    hints = get_type_hints(_service_class().run)
    assert hints["request"] is DurableReplayPromotionPublicationRequest


def test_run_return_annotation_is_existing_lifecycle_result() -> None:
    hints = get_type_hints(_service_class().run)
    assert hints["return"] is ProductionLifecycleStageResult


def test_service_owns_or_receives_bc_adaptation_dependency() -> None:
    sig = inspect.signature(_service_class())
    params = list(sig.parameters.values())
    assert len(params) == 1
    assert params[0].name == "adaptation_service"


def test_entrypoint_delegates_exact_request_identity_to_bc() -> None:
    stub = _AdaptationStub()
    service = _service_class()(adaptation_service=stub)
    request = _request()

    service.run(request)

    assert len(stub.calls) == 1
    assert stub.calls[0] is request


def test_entrypoint_calls_bc_adaptation_exactly_once() -> None:
    stub = _AdaptationStub()
    service = _service_class()(adaptation_service=stub)

    service.run(_request())

    assert len(stub.calls) == 1


def test_entrypoint_returns_bc_result_unchanged() -> None:
    expected = ProductionLifecycleStageResult(
        name="publication",
        status="PASS",
        detail={
            "source_path": Path("explicit/source_decision.json"),
            "source_sha256": "b" * 64,
            "published_path": Path(
                "explicit/registry_root/active/champion_decision.json"
            ),
            "published_at_kst": "2026-08-25T19:46:00+09:00",
            "selected_model": "candidate-model",
        },
    )
    stub = _AdaptationStub(result=expected)
    service = _service_class()(adaptation_service=stub)

    actual = service.run(_request())

    assert actual is expected


@pytest.mark.parametrize(
    "action",
    ["hold", "block", "unknown"],
)
def test_entrypoint_requires_prepare_publish_before_adaptation(
    action: str,
) -> None:
    stub = _AdaptationStub()
    service = _service_class()(adaptation_service=stub)

    with pytest.raises(ValueError):
        service.run(_request(action=action))

    assert stub.calls == []


def test_bc_adaptation_exception_propagates_unchanged() -> None:
    class SentinelError(RuntimeError):
        pass

    sentinel = SentinelError("bc-owned failure")
    stub = _AdaptationStub(error=sentinel)
    service = _service_class()(adaptation_service=stub)

    with pytest.raises(SentinelError) as exc_info:
        service.run(_request())

    assert exc_info.value is sentinel
    assert len(stub.calls) == 1


def test_entrypoint_does_not_mutate_request() -> None:
    stub = _AdaptationStub()
    service = _service_class()(adaptation_service=stub)
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

    service.run(request)

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


def test_entrypoint_declares_no_second_result_model() -> None:
    module = _module()
    classes = [
        value
        for _, value in vars(module).items()
        if inspect.isclass(value)
        and value.__module__ == MODULE_NAME
    ]
    assert [cls.__name__ for cls in classes] == [SERVICE_NAME]
    assert not any(dataclasses.is_dataclass(cls) for cls in classes)


def test_entrypoint_has_no_namespace_cli_or_lifecycle_adapter_dependency() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8-sig").lower()
    forbidden = (
        "argparse",
        "namespace",
        "lrp.cli",
        "production_lifecycle_adapters",
        "run_publication_stage",
        "publish_champion",
    )
    for token in forbidden:
        assert token not in source


def test_entrypoint_has_no_direct_publisher_or_mutation_surface() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8-sig").lower()
    forbidden = (
        "productionchampionregistrypublisher",
        ".publish(",
        "write_text",
        "write_bytes",
        "json.dump",
        "mkdir(",
        "unlink(",
        "os.replace",
    )
    for token in forbidden:
        assert token not in source


def test_entrypoint_has_no_discovery_policy_or_rollback_surface() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8-sig").lower()
    forbidden = (
        "discover",
        "latest",
        "getenv",
        "environ",
        "candidate_advantage_count",
        "baseline_advantage_count",
        "baseline_delta",
        "eligibility",
        "promotion_policy",
        "rollback",
    )
    for token in forbidden:
        assert token not in source


def test_entrypoint_depends_on_exact_existing_owners() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8-sig"))

    imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")

    assert (
        "lrp.operations.durable_replay_promotion_publication_request"
        in imports
    )
    assert (
        "lrp.operations.durable_replay_publication_lifecycle_adaptation"
        in imports
    )
    assert "lrp.production.production_lifecycle" in imports

    forbidden = {
        "lrp.production.production_lifecycle_adapters",
        "lrp.production.champion_registry_publisher",
        "lrp.cli.production_lifecycle",
        "lrp.cli.publish_champion",
    }
    assert imports.isdisjoint(forbidden)