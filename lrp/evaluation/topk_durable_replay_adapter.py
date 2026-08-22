"""Adapter from durable prediction evaluation source to replay prediction."""

from __future__ import annotations

from lrp.contracts.exceptions import ContractError
from lrp.evaluation.topk_replay_adapter import TopKReplayPrediction
from lrp.pipelines.durable_prediction_evaluation_source import (
    DurablePredictionEvaluationSource,
)


class TopKDurableReplayAdapter:
    """Project a durable evaluation source into a replay prediction."""

    def adapt(
        self,
        *,
        source: DurablePredictionEvaluationSource,
        history_rounds: tuple[int, ...],
        model_name: str,
        regime_id: str | None = None,
        strategy_name: str | None = None,
    ) -> TopKReplayPrediction:
        if not isinstance(
            source,
            DurablePredictionEvaluationSource,
        ):
            raise ContractError(
                "source must be DurablePredictionEvaluationSource"
            )

        return TopKReplayPrediction(
            round_no=source.round_no,
            history_rounds=history_rounds,
            predictions=source.selected_sets,
            model_name=model_name,
            regime_id=regime_id,
            strategy_name=strategy_name,
        )