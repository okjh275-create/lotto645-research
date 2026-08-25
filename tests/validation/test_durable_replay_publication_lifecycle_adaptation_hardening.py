from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import pytest

from lrp.operations.durable_replay_publication_lifecycle_adaptation import (
    DurableReplayPublicationLifecycleAdaptationService,
)
from lrp.operations.durable_replay_promotion_publication_request import (
    DurableReplayPromotionPublicationRequest,
)
from lrp.production.champion_registry_publisher import (
    ProductionChampionPublicationResult,
)
from lrp.production.production_lifecycle import ProductionLifecycleStageResult


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
        window=MappingProxyType({"name": "bc04-hardening"}),
        source_decision=Path("explicit/source_decision.json"),
        registry_root=Path("explicit/registry_root"),
    )


def _publication_result() -> ProductionChampionPublicationResult:
    return ProductionChampionPublicationResult(
        source_path=Path("explicit/source_decision.json"),
        source_sha256="c" * 64,
        published_path=Path(
            "explicit/registry_root/active/champion_decision.json"
        ),
        published_at_kst="2026-08-25T17:20:00+09:00",
        selected_model="candidate-model",
    )


_DEFAULT_RESULT = object()


class _ExecutionStub:
    def __init__(
        self,
        result: object = _DEFAULT_RESULT,
        error: BaseException | None = None,
    ) -> None:
        self.calls: list[DurableReplayPromotionPublicationRequest] = []
        self.result = (
            _publication_result()
            if result is _DEFAULT_RESULT
            else result
        )
        self.error = error

    def execute(self, request):
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        return self.result


def test_valid_adaptation_round_trip() -> None:
    stub = _ExecutionStub()
    request = _request()
    result = DurableReplayPublicationLifecycleAdaptationService(
        execution_service=stub
    ).adapt(request)

    assert isinstance(result, ProductionLifecycleStageResult)
    assert result.name == "publication"
    assert result.status == "PASS"
    assert stub.calls == [request]
    assert result.detail == {
        "source_path": stub.result.source_path,
        "source_sha256": stub.result.source_sha256,
        "published_path": stub.result.published_path,
        "published_at_kst": stub.result.published_at_kst,
        "selected_model": stub.result.selected_model,
    }


@pytest.mark.parametrize(
    "value",
    [None, 1, "request", object()],
)
def test_invalid_request_type_fails_before_execution(value: object) -> None:
    stub = _ExecutionStub()
    service = DurableReplayPublicationLifecycleAdaptationService(
        execution_service=stub
    )

    with pytest.raises(TypeError):
        service.adapt(value)  # type: ignore[arg-type]

    assert stub.calls == []


@pytest.mark.parametrize(
    "action",
    ["hold", "block", "unknown", "", "PREPARE_PUBLISH"],
)
def test_non_prepare_publish_action_fails_before_execution(
    action: str,
) -> None:
    request = _request()
    object.__setattr__(request, "action", action)

    stub = _ExecutionStub()
    service = DurableReplayPublicationLifecycleAdaptationService(
        execution_service=stub
    )

    with pytest.raises(ValueError):
        service.adapt(request)

    assert stub.calls == []


def test_execution_exception_propagates_unchanged() -> None:
    sentinel = RuntimeError("bc04 execution failure")
    stub = _ExecutionStub(error=sentinel)
    service = DurableReplayPublicationLifecycleAdaptationService(
        execution_service=stub
    )

    with pytest.raises(RuntimeError) as exc_info:
        service.adapt(_request())

    assert exc_info.value is sentinel
    assert len(stub.calls) == 1


@pytest.mark.parametrize(
    "bad_result",
    [None, {}, object(), "publication"],
)
def test_invalid_execution_result_type_fails_closed(
    bad_result: object,
) -> None:
    stub = _ExecutionStub(result=bad_result)
    service = DurableReplayPublicationLifecycleAdaptationService(
        execution_service=stub
    )

    with pytest.raises((TypeError, AttributeError)):
        service.adapt(_request())

    assert len(stub.calls) == 1


def test_detail_contains_exact_existing_publication_fields_only() -> None:
    stub = _ExecutionStub()
    result = DurableReplayPublicationLifecycleAdaptationService(
        execution_service=stub
    ).adapt(_request())

    assert set(result.detail) == {
        "source_path",
        "source_sha256",
        "published_path",
        "published_at_kst",
        "selected_model",
    }


def test_detail_preserves_none_selected_model() -> None:
    publication = ProductionChampionPublicationResult(
        source_path=Path("explicit/source_decision.json"),
        source_sha256="d" * 64,
        published_path=Path(
            "explicit/registry_root/active/champion_decision.json"
        ),
        published_at_kst="2026-08-25T17:21:00+09:00",
        selected_model=None,
    )
    stub = _ExecutionStub(result=publication)

    result = DurableReplayPublicationLifecycleAdaptationService(
        execution_service=stub
    ).adapt(_request())

    assert result.detail["selected_model"] is None


def test_adaptation_is_deterministic_for_same_stub_result() -> None:
    publication = _publication_result()

    first = DurableReplayPublicationLifecycleAdaptationService(
        execution_service=_ExecutionStub(result=publication)
    ).adapt(_request())

    second = DurableReplayPublicationLifecycleAdaptationService(
        execution_service=_ExecutionStub(result=publication)
    ).adapt(_request())

    assert first == second


def test_adaptation_does_not_mutate_request_or_window() -> None:
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

    DurableReplayPublicationLifecycleAdaptationService(
        execution_service=_ExecutionStub()
    ).adapt(request)

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


def test_service_calls_execution_exactly_once() -> None:
    stub = _ExecutionStub()
    DurableReplayPublicationLifecycleAdaptationService(
        execution_service=stub
    ).adapt(_request())

    assert len(stub.calls) == 1


def test_no_direct_publisher_or_lifecycle_cli_dependency() -> None:
    import inspect
    import lrp.operations.durable_replay_publication_lifecycle_adaptation as module

    source = inspect.getsource(module).lower()

    assert "productionchampionregistrypublisher" not in source
    assert "run_publication_stage" not in source
    assert "publish_champion" not in source
    assert "lrp.cli" not in source
    assert "argparse" not in source


def test_no_discovery_policy_recomputation_or_rollback_surface() -> None:
    import inspect
    import lrp.operations.durable_replay_publication_lifecycle_adaptation as module

    source = inspect.getsource(module).lower()

    forbidden = (
        "discover",
        "latest",
        "getenv",
        "environ",
        "candidate_advantage_count",
        "baseline_advantage_count",
        "baseline_delta",
        "rollback",
    )

    for token in forbidden:
        assert token not in source