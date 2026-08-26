from __future__ import annotations

from pathlib import Path
from typing import Protocol, TypeVar


_ResultT = TypeVar("_ResultT")


class _InvocationSourceAdapter(Protocol):
    def adapt(
        self,
        artifact_root: str | Path,
        end_round: int,
        *,
        source_decision: str | Path,
        registry_root: str | Path,
        output_path: str | Path,
    ) -> Path: ...


class _InvocationExecutionService(Protocol[_ResultT]):
    def execute(
        self,
        path: str | Path,
    ) -> _ResultT: ...


class DurableReplayResultArtifactPromotionPublicationInvocationExecutionBridge:
    def __init__(
        self,
        source_adapter: _InvocationSourceAdapter,
        execution_service: _InvocationExecutionService[_ResultT],
    ) -> None:
        self._source_adapter = source_adapter
        self._execution_service = execution_service

    def execute(
        self,
        artifact_root: str | Path,
        end_round: int,
        *,
        source_decision: str | Path,
        registry_root: str | Path,
        output_path: str | Path,
    ) -> _ResultT:
        invocation_path = self._source_adapter.adapt(
            artifact_root,
            end_round,
            source_decision=source_decision,
            registry_root=registry_root,
            output_path=output_path,
        )
        return self._execution_service.execute(
            invocation_path
        )