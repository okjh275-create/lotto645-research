"""Operational orchestration for durable Top-K replay evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lrp.contracts.exceptions import ContractError
from lrp.evaluation.contracts import EvaluationWindow
from lrp.evaluation.topk_replay_evaluation import (
    TopKReplayEvaluationRequest,
    TopKReplayEvaluationResult,
    TopKReplayEvaluationService,
)
from lrp.operations.durable_replay_consumer import (
    DurableReplayOperationalConsumer,
)


@dataclass(frozen=True)
class DurableReplayEvaluationSourceSpec:
    """Caller-supplied durable replay source and replay context."""

    artifact_path: str | Path
    history_rounds: tuple[int, ...]
    model_name: str
    regime_id: str | None = None
    strategy_name: str | None = None


class DurableReplayEvaluationOrchestrator:
    """Compose durable replay sources into an existing replay evaluation."""

    def evaluate(
        self,
        *,
        window: EvaluationWindow,
        candidate_sources: tuple[
            DurableReplayEvaluationSourceSpec,
            ...
        ],
        baseline_sources: tuple[
            DurableReplayEvaluationSourceSpec,
            ...
        ],
        actual_draws: tuple[Any, ...],
    ) -> TopKReplayEvaluationResult:
        consumer = DurableReplayOperationalConsumer()

        candidate_predictions = tuple(
            self._load_source(
                consumer=consumer,
                source=source,
                label="candidate",
            )
            for source in candidate_sources
        )

        baseline_predictions = tuple(
            self._load_source(
                consumer=consumer,
                source=source,
                label="baseline",
            )
            for source in baseline_sources
        )

        request = TopKReplayEvaluationRequest(
            window=window,
            candidate_predictions=candidate_predictions,
            baseline_predictions=baseline_predictions,
            actual_draws=actual_draws,
        )

        return TopKReplayEvaluationService().evaluate(
            request=request
        )

    @staticmethod
    def _load_source(
        *,
        consumer: DurableReplayOperationalConsumer,
        source: DurableReplayEvaluationSourceSpec,
        label: str,
    ):
        if not isinstance(
            source,
            DurableReplayEvaluationSourceSpec,
        ):
            raise ContractError(
                f"{label} source must be "
                "DurableReplayEvaluationSourceSpec"
            )

        return consumer.load(
            artifact_path=source.artifact_path,
            history_rounds=source.history_rounds,
            model_name=source.model_name,
            regime_id=source.regime_id,
            strategy_name=source.strategy_name,
        )