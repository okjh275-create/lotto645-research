from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lrp.contracts.exceptions import ContractError
from lrp.io.draws import HistoryRow
from lrp.pipelines.models import PredictionResult

from lrp.evaluation.topk_live_evaluation_orchestrator import (
    TopKLiveEvaluationOrchestrator,
    TopKLiveEvaluationRequest,
    TopKLiveEvaluationResult,
)

from lrp.evaluation.topk_live_evaluation_snapshot_factory import (
    TopKLiveEvaluationSnapshotBuildRequest,
    TopKLiveEvaluationSnapshotFactory,
)

from lrp.evaluation.topk_live_evaluation_source_snapshot import (
    TopKLiveEvaluationSourcePair,
)


@dataclass(frozen=True)
class TopKLiveEvaluationRuntimeRequest:
    window: Any
    candidate_prediction_result: PredictionResult
    candidate_history_rows: tuple[HistoryRow, ...]
    candidate_model_name: str
    candidate_source_artifact_sha256: str
    baseline_prediction_result: PredictionResult
    baseline_history_rows: tuple[HistoryRow, ...]
    baseline_model_name: str
    baseline_source_artifact_sha256: str
    actual_draws: tuple[Any, ...]
    candidate_regime_id: str | None = None
    candidate_strategy_name: str | None = None
    baseline_regime_id: str | None = None
    baseline_strategy_name: str | None = None

    def __post_init__(self) -> None:
        candidate_build_request = (
            TopKLiveEvaluationSnapshotBuildRequest(
                prediction_result=(
                    self.candidate_prediction_result
                ),
                history_rows=(
                    self.candidate_history_rows
                ),
                model_name=(
                    self.candidate_model_name
                ),
                source_artifact_sha256=(
                    self.candidate_source_artifact_sha256
                ),
                regime_id=(
                    self.candidate_regime_id
                ),
                strategy_name=(
                    self.candidate_strategy_name
                ),
            )
        )

        baseline_build_request = (
            TopKLiveEvaluationSnapshotBuildRequest(
                prediction_result=(
                    self.baseline_prediction_result
                ),
                history_rows=(
                    self.baseline_history_rows
                ),
                model_name=(
                    self.baseline_model_name
                ),
                source_artifact_sha256=(
                    self.baseline_source_artifact_sha256
                ),
                regime_id=(
                    self.baseline_regime_id
                ),
                strategy_name=(
                    self.baseline_strategy_name
                ),
            )
        )

        object.__setattr__(
            self,
            "candidate_prediction_result",
            candidate_build_request.prediction_result,
        )

        object.__setattr__(
            self,
            "candidate_history_rows",
            candidate_build_request.history_rows,
        )

        object.__setattr__(
            self,
            "candidate_model_name",
            candidate_build_request.model_name,
        )

        object.__setattr__(
            self,
            "candidate_source_artifact_sha256",
            candidate_build_request.source_artifact_sha256,
        )

        object.__setattr__(
            self,
            "candidate_regime_id",
            candidate_build_request.regime_id,
        )

        object.__setattr__(
            self,
            "candidate_strategy_name",
            candidate_build_request.strategy_name,
        )

        object.__setattr__(
            self,
            "baseline_prediction_result",
            baseline_build_request.prediction_result,
        )

        object.__setattr__(
            self,
            "baseline_history_rows",
            baseline_build_request.history_rows,
        )

        object.__setattr__(
            self,
            "baseline_model_name",
            baseline_build_request.model_name,
        )

        object.__setattr__(
            self,
            "baseline_source_artifact_sha256",
            baseline_build_request.source_artifact_sha256,
        )

        object.__setattr__(
            self,
            "baseline_regime_id",
            baseline_build_request.regime_id,
        )

        object.__setattr__(
            self,
            "baseline_strategy_name",
            baseline_build_request.strategy_name,
        )


@dataclass(frozen=True)
class TopKLiveEvaluationRuntimeResult:
    evaluation: TopKLiveEvaluationResult
    source_pair: TopKLiveEvaluationSourcePair


class TopKLiveEvaluationRuntimeService:
    def execute(
        self,
        *,
        request: TopKLiveEvaluationRuntimeRequest,
    ) -> TopKLiveEvaluationRuntimeResult:
        if not isinstance(
            request,
            TopKLiveEvaluationRuntimeRequest,
        ):
            raise ContractError(
                "request must be "
                "TopKLiveEvaluationRuntimeRequest"
            )

        snapshot_factory = (
            TopKLiveEvaluationSnapshotFactory()
        )

        candidate_snapshot = (
            snapshot_factory.build(
                request=(
                    TopKLiveEvaluationSnapshotBuildRequest(
                        prediction_result=(
                            request.candidate_prediction_result
                        ),
                        history_rows=(
                            request.candidate_history_rows
                        ),
                        model_name=(
                            request.candidate_model_name
                        ),
                        source_artifact_sha256=(
                            request
                            .candidate_source_artifact_sha256
                        ),
                        regime_id=(
                            request.candidate_regime_id
                        ),
                        strategy_name=(
                            request.candidate_strategy_name
                        ),
                    )
                )
            )
        )

        baseline_snapshot = (
            snapshot_factory.build(
                request=(
                    TopKLiveEvaluationSnapshotBuildRequest(
                        prediction_result=(
                            request.baseline_prediction_result
                        ),
                        history_rows=(
                            request.baseline_history_rows
                        ),
                        model_name=(
                            request.baseline_model_name
                        ),
                        source_artifact_sha256=(
                            request
                            .baseline_source_artifact_sha256
                        ),
                        regime_id=(
                            request.baseline_regime_id
                        ),
                        strategy_name=(
                            request.baseline_strategy_name
                        ),
                    )
                )
            )
        )

        source_pair = TopKLiveEvaluationSourcePair(
            candidate=candidate_snapshot,
            baseline=baseline_snapshot,
        )

        evaluation_request = TopKLiveEvaluationRequest(
            window=request.window,
            candidate_prediction_result=(
                request.candidate_prediction_result
            ),
            candidate_history_rows=(
                request.candidate_history_rows
            ),
            candidate_model_name=(
                request.candidate_model_name
            ),
            baseline_prediction_result=(
                request.baseline_prediction_result
            ),
            baseline_history_rows=(
                request.baseline_history_rows
            ),
            baseline_model_name=(
                request.baseline_model_name
            ),
            actual_draws=request.actual_draws,
            candidate_regime_id=(
                request.candidate_regime_id
            ),
            candidate_strategy_name=(
                request.candidate_strategy_name
            ),
            baseline_regime_id=(
                request.baseline_regime_id
            ),
            baseline_strategy_name=(
                request.baseline_strategy_name
            ),
        )

        evaluation = (
            TopKLiveEvaluationOrchestrator()
            .evaluate(
                request=evaluation_request
            )
        )

        return TopKLiveEvaluationRuntimeResult(
            evaluation=evaluation,
            source_pair=source_pair,
        )