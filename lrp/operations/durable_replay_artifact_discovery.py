"""Deterministic durable replay artifact path projection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

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
    artifact_key: str | None = None

    def __post_init__(self) -> None:
        if self.artifact_key is not None:
            if not isinstance(self.artifact_key, str):
                raise ContractError("artifact_key must be str or None")
            if not self.artifact_key:
                raise ContractError("artifact_key must not be empty")
            if len(self.artifact_key) > 128:
                raise ContractError(
                    "artifact_key must be at most 128 characters"
                )
            if (
                re.fullmatch(
                    r"[A-Za-z0-9][A-Za-z0-9._-]*",
                    self.artifact_key,
                )
                is None
            ):
                raise ContractError(
                    "artifact_key contains invalid characters"
                )
            if self.artifact_key in {".", ".."}:
                raise ContractError("artifact_key must not be dot path")
    artifact_key: str | None = None


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

        round_directory = (
            Path(artifact_root)
            / "prediction-evaluation-sources"
            / f"round_{selector.round_no:04d}"
        )
        if selector.artifact_key is not None:
            round_directory = (
                round_directory
                / selector.artifact_key
            )
        artifact_path = (
            round_directory
            / "evaluation_source.json"
        )

        return DurableReplayExecutionSource(
            artifact_path=artifact_path,
            round_no=selector.round_no,
            model_name=selector.model_name,
            regime_id=selector.regime_id,
            strategy_name=selector.strategy_name,
        )