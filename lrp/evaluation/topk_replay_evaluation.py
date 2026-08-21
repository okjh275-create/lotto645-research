from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lrp.contracts import ContractError
from lrp.evaluation import EvaluationWindow
from lrp.evaluation.topk_replay_adapter import (
    TopKReplayAdapter,
    TopKReplayBaselineProvider,
    TopKReplayPrediction,
)
from lrp.evaluation.topk_walkforward import (
    TopKWalkForwardEvaluator,
    WalkForwardEvaluation,
)


@dataclass(frozen=True)
class TopKReplayEvaluationRequest:
    window: EvaluationWindow
    candidate_predictions: tuple[
        TopKReplayPrediction,
        ...,
    ]
    baseline_predictions: tuple[
        TopKReplayPrediction,
        ...,
    ]
    actual_draws: tuple[Any, ...]


@dataclass(frozen=True)
class TopKReplayEvaluationResult:
    evaluation: WalkForwardEvaluation
    candidate_model_name: str
    baseline_model_name: str
    round_count: int


class TopKReplayEvaluationService:

    def evaluate(
        self,
        *,
        request: TopKReplayEvaluationRequest,
    ) -> TopKReplayEvaluationResult:
        self._validate_request(
            request
        )

        self._validate_prediction_source(
            request.candidate_predictions,
            source_name="candidate",
        )

        self._validate_prediction_source(
            request.baseline_predictions,
            source_name="baseline",
        )

        candidate_predictions = tuple(
            sorted(
                request.candidate_predictions,
                key=lambda row:
                    row.round_no,
            )
        )

        baseline_predictions = tuple(
            sorted(
                request.baseline_predictions,
                key=lambda row:
                    row.round_no,
            )
        )

        actual_draws = tuple(
            request.actual_draws
        )

        candidate_rounds = tuple(
            row.round_no
            for row in candidate_predictions
        )

        baseline_rounds = tuple(
            row.round_no
            for row in baseline_predictions
        )

        if (
            candidate_rounds
            != baseline_rounds
        ):
            raise ContractError(
                "candidate and baseline rounds must match"
            )

        self._validate_window(
            request.window,
            candidate_rounds,
        )

        self._validate_actual_draws(
            actual_draws,
            required_rounds=set(
                candidate_rounds
            ),
        )

        candidate_model_name = (
            candidate_predictions[
                0
            ].model_name
        )

        baseline_model_name = (
            baseline_predictions[
                0
            ].model_name
        )

        adapter = TopKReplayAdapter()

        candidate_rows = adapter.adapt(
            prediction_rows=(
                candidate_predictions
            ),
            actual_draws=actual_draws,
        )

        baseline_rows = adapter.adapt(
            prediction_rows=(
                baseline_predictions
            ),
            actual_draws=actual_draws,
        )

        baseline_provider = (
            TopKReplayBaselineProvider(
                baseline_rows
            )
        )

        evaluator = (
            TopKWalkForwardEvaluator(
                baseline_provider=(
                    baseline_provider
                )
            )
        )

        evaluation = evaluator.evaluate(
            window=request.window,
            replay_rows=candidate_rows,
        )

        return TopKReplayEvaluationResult(
            evaluation=evaluation,
            candidate_model_name=(
                candidate_model_name
            ),
            baseline_model_name=(
                baseline_model_name
            ),
            round_count=len(
                evaluation.rounds
            ),
        )

    @staticmethod
    def _validate_request(
        request: TopKReplayEvaluationRequest,
    ) -> None:
        if not isinstance(
            request,
            TopKReplayEvaluationRequest,
        ):
            raise ContractError(
                "request must be TopKReplayEvaluationRequest"
            )

        if not isinstance(
            request.window,
            EvaluationWindow,
        ):
            raise ContractError(
                "window must be EvaluationWindow"
            )

        if not isinstance(
            request.candidate_predictions,
            tuple,
        ):
            raise ContractError(
                "candidate_predictions must be tuple"
            )

        if not isinstance(
            request.baseline_predictions,
            tuple,
        ):
            raise ContractError(
                "baseline_predictions must be tuple"
            )

        if not isinstance(
            request.actual_draws,
            tuple,
        ):
            raise ContractError(
                "actual_draws must be tuple"
            )

    @staticmethod
    def _validate_prediction_source(
        rows: tuple[
            TopKReplayPrediction,
            ...,
        ],
        *,
        source_name: str,
    ) -> None:
        if not rows:
            raise ContractError(
                f"{source_name} predictions must not be empty"
            )

        for row in rows:
            if not isinstance(
                row,
                TopKReplayPrediction,
            ):
                raise ContractError(
                    f"{source_name} prediction item has invalid type"
                )

        round_numbers = [
            row.round_no
            for row in rows
        ]

        if (
            len(
                set(
                    round_numbers
                )
            )
            != len(
                round_numbers
            )
        ):
            raise ContractError(
                f"{source_name} contains duplicate round"
            )

        model_names = {
            row.model_name
            for row in rows
        }

        if (
            len(
                model_names
            )
            != 1
        ):
            raise ContractError(
                f"{source_name} must contain exactly one model_name"
            )

        model_name = next(
            iter(
                model_names
            )
        )

        if (
            not isinstance(
                model_name,
                str,
            )
            or not model_name.strip()
        ):
            raise ContractError(
                f"{source_name} model_name must be non-empty"
            )

    @staticmethod
    def _validate_window(
        window: EvaluationWindow,
        rounds: tuple[int, ...],
    ) -> None:
        for round_no in rounds:
            if (
                round_no
                < window.start_round
                or round_no
                > window.end_round
            ):
                raise ContractError(
                    "prediction round outside evaluation window"
                )

    @staticmethod
    def _validate_actual_draws(
        draws: tuple[Any, ...],
        *,
        required_rounds: set[int],
    ) -> None:
        draw_rounds: list[int] = []

        for draw in draws:
            if not hasattr(
                draw,
                "round_no",
            ):
                raise ContractError(
                    "actual draw missing round_no"
                )

            if not hasattr(
                draw,
                "numbers",
            ):
                raise ContractError(
                    "actual draw missing numbers"
                )

            round_no = getattr(
                draw,
                "round_no",
            )

            if (
                not isinstance(
                    round_no,
                    int,
                )
                or isinstance(
                    round_no,
                    bool,
                )
            ):
                raise ContractError(
                    "actual draw round_no must be int"
                )

            draw_rounds.append(
                round_no
            )

        if (
            len(
                draw_rounds
            )
            != len(
                set(
                    draw_rounds
                )
            )
        ):
            raise ContractError(
                "actual draws contain duplicate round"
            )

        available = set(
            draw_rounds
        )

        missing = (
            required_rounds
            - available
        )

        if missing:
            raise ContractError(
                "actual draw missing for evaluation round"
            )
