"""Operational consumer for durable replay prediction sources."""

from __future__ import annotations

from pathlib import Path

from lrp.contracts.exceptions import ContractError
from lrp.evaluation.topk_durable_replay_adapter import (
    TopKDurableReplayAdapter,
)
from lrp.evaluation.topk_replay_adapter import (
    TopKReplayPrediction,
)
from lrp.pipelines.durable_prediction_evaluation_source import (
    source_from_json,
)


class DurableReplayOperationalConsumer:
    """Load one durable source artifact and project it for replay."""

    def load(
        self,
        *,
        artifact_path: str | Path,
        history_rounds: tuple[int, ...],
        model_name: str,
        regime_id: str | None = None,
        strategy_name: str | None = None,
    ) -> TopKReplayPrediction:
        if not isinstance(
            artifact_path,
            (str, Path),
        ):
            raise ContractError(
                "artifact_path must be str or Path"
            )

        payload = Path(
            artifact_path
        ).read_text(
            encoding="utf-8"
        )

        source = source_from_json(
            payload
        )

        return TopKDurableReplayAdapter().adapt(
            source=source,
            history_rounds=history_rounds,
            model_name=model_name,
            regime_id=regime_id,
            strategy_name=strategy_name,
        )