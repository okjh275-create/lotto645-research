from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import pytest

from lrp.operations.durable_replay_promotion_publication_execution import (
    DurableReplayPromotionPublicationExecutionService,
)
from lrp.operations.durable_replay_promotion_publication_request import (
    DurableReplayPromotionPublicationRequest,
)
from lrp.production.champion_registry_publisher import (
    ProductionChampionPublicationResult,
)


class StubPublisher:
    def __init__(
        self,
        *,
        result: ProductionChampionPublicationResult | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.calls: list[tuple[object, object]] = []

    def publish(
        self,
        *,
        source_decision,
        registry_root,
    ) -> ProductionChampionPublicationResult:
        self.calls.append((source_decision, registry_root))
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def _result() -> ProductionChampionPublicationResult:
    return ProductionChampionPublicationResult(
        source_path="explicit/source_decision.json",
        source_sha256="e" * 64,
        published_path="explicit/registry_root/champion.json",
        published_at_kst="2026-08-25T14:30:00+09:00",
        selected_model="candidate-model",
    )


def _request(
    *,
    action: str = "prepare_publish",
    source_decision: object = Path("explicit/source_decision.json"),
    registry_root: object = Path("explicit/registry_root"),
) -> DurableReplayPromotionPublicationRequest:
    return DurableReplayPromotionPublicationRequest(
        status="PASS",
        round_count=1,
        candidate_model_name="candidate-model",
        baseline_model_name="baseline-model",
        recommendation="eligible",
        action=action,
        window=MappingProxyType({"name": "bb04-hardening"}),
        source_decision=source_decision,
        registry_root=registry_root,
    )


def test_valid_execution_round_trip() -> None:
    expected = _result()
    publisher = StubPublisher(result=expected)
    service = DurableReplayPromotionPublicationExecutionService(
        publisher=publisher,
    )
    request = _request()

    result = service.execute(request)

    assert result is expected
    assert publisher.calls == [
        (request.source_decision, request.registry_root)
    ]


@pytest.mark.parametrize("action", ["hold", "block", "unknown", ""])
def test_non_prepare_publish_action_fails_before_publisher_call(
    action: str,
) -> None:
    publisher = StubPublisher(result=_result())
    service = DurableReplayPromotionPublicationExecutionService(
        publisher=publisher,
    )

    with pytest.raises((TypeError, ValueError)):
        service.execute(_request(action=action))

    assert publisher.calls == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_decision", None),
        ("source_decision", 123),
        ("source_decision", ""),
        ("registry_root", None),
        ("registry_root", 123),
        ("registry_root", ""),
    ],
)
def test_malformed_publication_identity_fails_before_publisher_call(
    field: str,
    value: object,
) -> None:
    publisher = StubPublisher(result=_result())
    service = DurableReplayPromotionPublicationExecutionService(
        publisher=publisher,
    )
    kwargs = {field: value}

    with pytest.raises((TypeError, ValueError)):
        service.execute(_request(**kwargs))

    assert publisher.calls == []


def test_invalid_request_type_fails_before_publisher_call() -> None:
    publisher = StubPublisher(result=_result())
    service = DurableReplayPromotionPublicationExecutionService(
        publisher=publisher,
    )

    with pytest.raises((TypeError, AttributeError, ValueError)):
        service.execute(object())  # type: ignore[arg-type]

    assert publisher.calls == []


def test_publisher_exception_propagates_same_instance() -> None:
    sentinel = RuntimeError("sentinel publisher failure")
    publisher = StubPublisher(error=sentinel)
    service = DurableReplayPromotionPublicationExecutionService(
        publisher=publisher,
    )

    with pytest.raises(RuntimeError) as caught:
        service.execute(_request())

    assert caught.value is sentinel
    assert len(publisher.calls) == 1


def test_publisher_result_is_returned_by_identity() -> None:
    expected = _result()
    publisher = StubPublisher(result=expected)
    service = DurableReplayPromotionPublicationExecutionService(
        publisher=publisher,
    )

    actual = service.execute(_request())

    assert actual is expected


def test_request_is_not_mutated_by_execution() -> None:
    expected = _result()
    publisher = StubPublisher(result=expected)
    service = DurableReplayPromotionPublicationExecutionService(
        publisher=publisher,
    )
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

    service.execute(request)

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


def test_injected_publisher_is_used_exactly_once() -> None:
    publisher = StubPublisher(result=_result())
    service = DurableReplayPromotionPublicationExecutionService(
        publisher=publisher,
    )

    service.execute(_request())

    assert len(publisher.calls) == 1


def test_service_does_not_require_runtime_or_cli_context() -> None:
    expected = _result()
    publisher = StubPublisher(result=expected)
    service = DurableReplayPromotionPublicationExecutionService(
        publisher=publisher,
    )

    result = service.execute(_request())

    assert result is expected


def test_window_is_not_read_or_rewritten_for_publication() -> None:
    expected = _result()
    publisher = StubPublisher(result=expected)
    service = DurableReplayPromotionPublicationExecutionService(
        publisher=publisher,
    )
    request = _request()
    before = request.window

    service.execute(request)

    assert request.window is before
    assert dict(request.window) == {"name": "bb04-hardening"}