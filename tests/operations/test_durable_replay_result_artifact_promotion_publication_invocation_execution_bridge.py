from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import lrp.operations.durable_replay_result_artifact_promotion_publication_invocation_execution_bridge as product


class InvocationSourceSpy:
    def __init__(self, result: Path) -> None:
        self.result = result
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def adapt(
        self,
        artifact_root: str | Path,
        end_round: int,
        *,
        source_decision: str | Path,
        registry_root: str | Path,
        output_path: str | Path,
    ) -> Path:
        self.calls.append(
            (
                (artifact_root, end_round),
                {
                    "source_decision": source_decision,
                    "registry_root": registry_root,
                    "output_path": output_path,
                },
            )
        )
        return self.result


class ExecutionSpy:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[Path] = []

    def execute(self, path: str | Path) -> object:
        self.calls.append(Path(path))
        return self.result


def _build(
    *,
    invocation_path: Path,
    execution_result: object,
) -> tuple[
    product.DurableReplayResultArtifactPromotionPublicationInvocationExecutionBridge,
    InvocationSourceSpy,
    ExecutionSpy,
]:
    source = InvocationSourceSpy(invocation_path)
    execution = ExecutionSpy(execution_result)

    bridge = (
        product.DurableReplayResultArtifactPromotionPublicationInvocationExecutionBridge(
            source_adapter=source,
            execution_service=execution,
        )
    )

    return bridge, source, execution


def test_bridge_exposes_execute_as_only_public_operation() -> None:
    public = {
        name
        for name in vars(
            product.DurableReplayResultArtifactPromotionPublicationInvocationExecutionBridge
        )
        if not name.startswith("_")
    }

    assert public == {"execute"}


def test_execute_forwards_source_arguments_exactly(
    tmp_path: Path,
) -> None:
    invocation_path = tmp_path / "invocation.json"
    result = object()

    bridge, source, _ = _build(
        invocation_path=invocation_path,
        execution_result=result,
    )

    artifact_root = tmp_path / "artifacts"
    source_decision = tmp_path / "decision.json"
    registry_root = tmp_path / "registry"
    output_path = tmp_path / "generated.json"

    bridge.execute(
        artifact_root,
        1234,
        source_decision=source_decision,
        registry_root=registry_root,
        output_path=output_path,
    )

    assert source.calls == [
        (
            (artifact_root, 1234),
            {
                "source_decision": source_decision,
                "registry_root": registry_root,
                "output_path": output_path,
            },
        )
    ]


def test_execute_passes_exact_source_path_to_execution(
    tmp_path: Path,
) -> None:
    invocation_path = tmp_path / "source-result.json"
    result = object()

    bridge, _, execution = _build(
        invocation_path=invocation_path,
        execution_result=result,
    )

    bridge.execute(
        tmp_path / "artifacts",
        1234,
        source_decision=tmp_path / "decision.json",
        registry_root=tmp_path / "registry",
        output_path=tmp_path / "requested-output.json",
    )

    assert execution.calls == [invocation_path]


def test_execute_returns_execution_result_unchanged(
    tmp_path: Path,
) -> None:
    invocation_path = tmp_path / "invocation.json"
    expected = object()

    bridge, _, _ = _build(
        invocation_path=invocation_path,
        execution_result=expected,
    )

    actual = bridge.execute(
        tmp_path / "artifacts",
        1234,
        source_decision=tmp_path / "decision.json",
        registry_root=tmp_path / "registry",
        output_path=tmp_path / "invocation.json",
    )

    assert actual is expected


def test_source_failure_propagates_without_execution(
    tmp_path: Path,
) -> None:
    expected = RuntimeError("source failure")

    class FailingSource:
        def adapt(self, *args: Any, **kwargs: Any) -> Path:
            raise expected

    execution = ExecutionSpy(object())

    bridge = (
        product.DurableReplayResultArtifactPromotionPublicationInvocationExecutionBridge(
            source_adapter=FailingSource(),
            execution_service=execution,
        )
    )

    with pytest.raises(RuntimeError) as captured:
        bridge.execute(
            tmp_path / "artifacts",
            1234,
            source_decision=tmp_path / "decision.json",
            registry_root=tmp_path / "registry",
            output_path=tmp_path / "invocation.json",
        )

    assert captured.value is expected
    assert execution.calls == []


def test_execution_failure_propagates_unchanged(
    tmp_path: Path,
) -> None:
    invocation_path = tmp_path / "invocation.json"
    expected = RuntimeError("execution failure")

    source = InvocationSourceSpy(invocation_path)

    class FailingExecution:
        def execute(self, path: str | Path) -> object:
            assert Path(path) == invocation_path
            raise expected

    bridge = (
        product.DurableReplayResultArtifactPromotionPublicationInvocationExecutionBridge(
            source_adapter=source,
            execution_service=FailingExecution(),
        )
    )

    with pytest.raises(RuntimeError) as captured:
        bridge.execute(
            tmp_path / "artifacts",
            1234,
            source_decision=tmp_path / "decision.json",
            registry_root=tmp_path / "registry",
            output_path=tmp_path / "invocation.json",
        )

    assert captured.value is expected


def test_execute_does_not_rewrite_source_path(
    tmp_path: Path,
) -> None:
    source_result = tmp_path / "different" / "actual.json"

    bridge, _, execution = _build(
        invocation_path=source_result,
        execution_result=object(),
    )

    requested_output = tmp_path / "requested" / "output.json"

    bridge.execute(
        tmp_path / "artifacts",
        1234,
        source_decision=tmp_path / "decision.json",
        registry_root=tmp_path / "registry",
        output_path=requested_output,
    )

    assert execution.calls == [source_result]
    assert execution.calls != [requested_output]
