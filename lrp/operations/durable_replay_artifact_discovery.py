"""Deterministic durable replay artifact path projection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lrp.contracts.exceptions import ContractError
from lrp.operations.durable_replay_execution import (
    DurableReplayExecutionSource,
)


@dataclass(frozen=True)
class DurableReplayArtifactSelector:
    round_no: int
    model_name: str
    regime_id: str | None = None
    strategy_name: str | None = None


@dataclass(frozen=True)
class DurableReplayArtifactDiscoveryRequest:
    artifact_root: str | Path
    candidate_selectors: tuple[DurableReplayArtifactSelector, ...]
    baseline_selectors: tuple[DurableReplayArtifactSelector, ...]


class DurableReplayArtifactDiscoveryService:
    def discover(
        self,
        *,
        request: DurableReplayArtifactDiscoveryRequest,
    ) -> tuple[
        tuple[DurableReplayExecutionSource, ...],
        tuple[DurableReplayExecutionSource, ...],
    ]:
        if not isinstance(
            request,
            DurableReplayArtifactDiscoveryRequest,
        ):
            raise ContractError(
                "request must be "
                "DurableReplayArtifactDiscoveryRequest"
            )

        candidate_sources = tuple(
            self._execution_source(
                artifact_root=request.artifact_root,
                selector=selector,
                label="candidate",
            )
            for selector in request.candidate_selectors
        )

        baseline_sources = tuple(
            self._execution_source(
                artifact_root=request.artifact_root,
                selector=selector,
                label="baseline",
            )
            for selector in request.baseline_selectors
        )

        return (
            candidate_sources,
            baseline_sources,
        )

    def _execution_source(
        self,
        *,
        artifact_root: str | Path,
        selector: DurableReplayArtifactSelector,
        label: str,
    ) -> DurableReplayExecutionSource:
        if not isinstance(
            selector,
            DurableReplayArtifactSelector,
        ):
            raise ContractError(
                f"{label} selector must be "
                "DurableReplayArtifactSelector"
            )

        artifact_path = (
            Path(artifact_root)
            / "prediction-evaluation-sources"
            / f"round_{selector.round_no:04d}"
            / "evaluation_source.json"
        )

        return DurableReplayExecutionSource(
            artifact_path=artifact_path,
            round_no=selector.round_no,
            model_name=selector.model_name,
            regime_id=selector.regime_id,
            strategy_name=selector.strategy_name,
        )