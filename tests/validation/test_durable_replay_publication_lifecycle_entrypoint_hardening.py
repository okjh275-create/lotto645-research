from __future__ import annotations

import ast
from pathlib import Path
from types import MappingProxyType

import pytest

from lrp.operations.durable_replay_publication_lifecycle_entrypoint import (
    DurableReplayPublicationLifecycleEntrypoint,
)
from lrp.operations.durable_replay_promotion_publication_request import (
    DurableReplayPromotionPublicationRequest,
)
from lrp.production.production_lifecycle import (
    ProductionLifecycleStageResult,
)


PRODUCT_PATH = Path(
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
        window=MappingProxyType({"name": "bd04"}),
        source_decision=Path("explicit/source_decision.json"),
        registry_root=Path("explicit/registry_root"),
    )


class _AdaptationStub:
    def __init__(
        self,
        *,
        result: object | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.calls: list[object] = []
        self.result = (
            ProductionLifecycleStageResult(
                name="publication",
                status="PASS",
                detail={
                    "source_path": Path("explicit/source_decision.json"),
                    "source_sha256": "d" * 64,
                    "published_path": Path(
                        "explicit/registry_root/active/champion_decision.json"
                    ),
                    "published_at_kst": "2026-08-25T19:55:00+09:00",
                    "selected_model": "candidate-model",
                },
            )
            if result is None
            else result
        )
        self.error = error

    def adapt(self, request: object) -> object:
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        return self.result


def test_valid_entrypoint_round_trip() -> None:
    stub = _AdaptationStub()
    service = DurableReplayPublicationLifecycleEntrypoint(
        adaptation_service=stub
    )
    request = _request()

    result = service.run(request)

    assert len(stub.calls) == 1
    assert stub.calls[0] is request
    assert isinstance(result, ProductionLifecycleStageResult)
    assert result.name == "publication"
    assert result.status == "PASS"


@pytest.mark.parametrize(
    "bad_request",
    [None, 1, "request", object()],
)
def test_invalid_request_type_fails_before_adaptation(
    bad_request: object,
) -> None:
    stub = _AdaptationStub()
    service = DurableReplayPublicationLifecycleEntrypoint(
        adaptation_service=stub
    )

    with pytest.raises(TypeError):
        service.run(bad_request)  # type: ignore[arg-type]

    assert stub.calls == []


@pytest.mark.parametrize(
    "action",
    ["hold", "block", "unknown", "", "PREPARE_PUBLISH"],
)
def test_non_prepare_publish_fails_before_adaptation(
    action: str,
) -> None:
    stub = _AdaptationStub()
    service = DurableReplayPublicationLifecycleEntrypoint(
        adaptation_service=stub
    )

    with pytest.raises(ValueError):
        service.run(_request(action=action))

    assert stub.calls == []


def test_adaptation_exception_propagates_unchanged() -> None:
    class SentinelError(RuntimeError):
        pass

    sentinel = SentinelError("bc-owned failure")
    stub = _AdaptationStub(error=sentinel)
    service = DurableReplayPublicationLifecycleEntrypoint(
        adaptation_service=stub
    )

    with pytest.raises(SentinelError) as exc_info:
        service.run(_request())

    assert exc_info.value is sentinel
    assert len(stub.calls) == 1


@pytest.mark.parametrize(
    "bad_result",
    [None, {}, object(), "publication"],
)
def test_invalid_adaptation_result_type_fails_closed(
    bad_result: object,
) -> None:
    class Stub:
        def __init__(self, result: object) -> None:
            self.result = result
            self.calls: list[object] = []

        def adapt(self, request: object) -> object:
            self.calls.append(request)
            return self.result

    stub = Stub(bad_result)
    service = DurableReplayPublicationLifecycleEntrypoint(
        adaptation_service=stub  # type: ignore[arg-type]
    )

    with pytest.raises(TypeError):
        service.run(_request())

    assert len(stub.calls) == 1


def test_entrypoint_returns_exact_adaptation_result_identity() -> None:
    expected = ProductionLifecycleStageResult(
        name="publication",
        status="PASS",
        detail={
            "source_path": Path("explicit/source_decision.json"),
            "source_sha256": "e" * 64,
            "published_path": Path(
                "explicit/registry_root/active/champion_decision.json"
            ),
            "published_at_kst": "2026-08-25T19:56:00+09:00",
            "selected_model": None,
        },
    )

    stub = _AdaptationStub(result=expected)
    service = DurableReplayPublicationLifecycleEntrypoint(
        adaptation_service=stub
    )

    actual = service.run(_request())

    assert actual is expected


def test_entrypoint_does_not_mutate_request_or_window() -> None:
    stub = _AdaptationStub()
    service = DurableReplayPublicationLifecycleEntrypoint(
        adaptation_service=stub
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


def test_entrypoint_calls_adaptation_exactly_once() -> None:
    stub = _AdaptationStub()
    service = DurableReplayPublicationLifecycleEntrypoint(
        adaptation_service=stub
    )

    service.run(_request())

    assert len(stub.calls) == 1


def test_dependency_boundary_is_exact() -> None:
    tree = ast.parse(PRODUCT_PATH.read_text(encoding="utf-8-sig"))

    imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")

    assert imports == {
        "__future__",
        "lrp.operations.durable_replay_promotion_publication_request",
        "lrp.operations.durable_replay_publication_lifecycle_adaptation",
        "lrp.production.production_lifecycle",
    }


def test_no_namespace_cli_lifecycle_adapter_or_publisher_dependency() -> None:
    source = PRODUCT_PATH.read_text(encoding="utf-8-sig").lower()

    forbidden = (
        "argparse",
        "namespace",
        "lrp.cli",
        "production_lifecycle_adapters",
        "run_publication_stage",
        "publish_champion",
        "productionchampionregistrypublisher",
        ".publish(",
    )

    for token in forbidden:
        assert token not in source


def test_no_discovery_policy_recomputation_rollback_or_write_surface() -> None:
    source = PRODUCT_PATH.read_text(encoding="utf-8-sig").lower()

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
        "write_text",
        "write_bytes",
        "json.dump",
        "mkdir(",
        "unlink(",
        "os.replace",
    )

    for token in forbidden:
        assert token not in source