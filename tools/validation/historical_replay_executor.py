"""Real Project G executor for historical replay validation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

from lrp.adapters.statistics import StatisticsAdapter
from lrp.evolution.algorithms.adaptive import (
    AdaptiveWeightCalculator,
)
from lrp.evolution.contracts.learning_context import (
    LearningContext,
)
from lrp.evolution.integration.noop_weight_adapter import (
    NoOpEvolutionWeightAdapter,
)
from lrp.evolution.policies import AdaptiveWeightPolicy
from lrp.evolution.repositories.file_snapshot_repository import (
    FileSnapshotRepository,
)
from lrp.evolution.services.adaptive_pipeline import (
    AdaptiveEvolutionPipeline,
)
from lrp.evolution.services.coordinator import EvolutionCoordinator
from lrp.evolution.services.persistent_learning_runner import (
    PersistentLearningRunner,
)
from lrp.evolution.services.persistent_learning_service import (
    PersistentLearningService,
)
from lrp.evolution.services.review_learning_service import (
    ReviewLearningService,
)
from lrp.evolution.services.review_profile_evolution_service import (
    ReviewProfileEvolutionService,
)
from lrp.evolution.storage import SnapshotRepository
from lrp.regimes.bayesian_repository import (
    RegimeBayesianRepository,
)
from lrp.regimes.bayesian_updater import (
    RegimeBayesianUpdater,
)
from lrp.regimes.calibration_repository import (
    RegimeCalibrationRepository,
)
from lrp.regimes.calibration_updater import (
    RegimeCalibrationUpdater,
)
from lrp.regimes.learning_rate import (
    AdaptiveLearningRatePolicy,
)
from lrp.regimes.reward_calculator import (
    RegimeRewardCalculator,
)
from lrp.io import (
    history_until_round,
    long_gap_numbers,
    previous_numbers,
    to_statistics_draws,
)
from lrp.operations import review_prediction
from lrp.pipelines import (
    PredictionPipeline,
    PredictionRequest,
    prediction_to_dict,
)

from .historical_replay_models import (
    ReplayConfig,
    ReplayRoundResult,
)


@dataclass(frozen=True, slots=True)
class ReplayState:
    """State carried from one replay round to the next."""

    learning_context: LearningContext


@dataclass(frozen=True, slots=True)
class HistoricalReplayExecutor:
    """Execute one leak-free historical replay round."""

    history: tuple[object, ...]
    config: ReplayConfig
    learning_root: Path
    profile_root: Path
    regime_calibration_root: Path | None = None
    regime_bayesian_root: Path | None = None
    policy: str = "thompson"
    adaptive_calculator: (
        AdaptiveWeightCalculator | None
    ) = None

    def __post_init__(self) -> None:
        if not self.history:
            raise ValueError(
                "history must not be empty"
            )

        if (
            self.adaptive_calculator is not None
            and not isinstance(
                self.adaptive_calculator,
                AdaptiveWeightCalculator,
            )
        ):
            raise TypeError(
                "adaptive_calculator must be an "
                "AdaptiveWeightCalculator or None"
            )

    def __call__(
        self,
        round_no: int,
        seed: int,
        draw: object,
        state: object | None,
    ) -> tuple[
        ReplayRoundResult,
        ReplayState,
    ]:
        started = perf_counter()

        bounded = history_until_round(
            self.history,
            target_round=round_no,
        )

        statistics = StatisticsAdapter.load()
        draw_type = getattr(
            statistics.module,
            "DrawRecord",
        )
        statistics_draws = to_statistics_draws(
            bounded,
            draw_type=draw_type,
        )

        request = PredictionRequest(
            round_no=round_no,
            seed=seed,
            temperature=self.config.temperature,
            candidate_count=(
                self.config.candidate_count
            ),
            max_attempts_multiplier=50,
            top_k=self.config.top_k,
            practical_k=self.config.practical_k,
            previous_numbers=previous_numbers(
                bounded
            ),
            long_gap_numbers=long_gap_numbers(
                bounded,
                recent_draw_count=(
                    self.config.long_gap_window
                ),
            ),
        )

        analysis_config = statistics.create_config(
            short_window=10,
            mid_window=20,
            long_window=50,
            bootstrap_iterations=(
                100
                if self.config.mode == "fast"
                else 1000
            ),
            confidence_level=0.95,
            seed=20260719,
            top_n=10,
            backtest_minimum_history=max(
                50,
                len(statistics_draws) + 1,
            ),
            serialization_precision=8,
        )

        noop_pipeline = PredictionPipeline.load(
            evolution=NoOpEvolutionWeightAdapter()
        )
        applied_calibration_revision = None
        applied_calibration_sample_size = None
        applied_bayesian_revision = None
        applied_bayesian_sample_size = None

        if self.regime_calibration_root is not None:
            try:
                snapshot = (
                    RegimeCalibrationRepository(
                        self.regime_calibration_root
                    ).load_latest()
                )
            except Exception as exc:
                from lrp.regimes.calibration_repository import (
                    RegimeCalibrationNotFoundError,
                )
                from lrp.regimes.calibration_serializer import (
                    RegimeCalibrationSerializationError,
                )

                if not isinstance(
                    exc,
                    (
                        RegimeCalibrationNotFoundError,
                        RegimeCalibrationSerializationError,
                    ),
                ):
                    raise
            else:
                applied_calibration_revision = (
                    snapshot.revision
                )
                applied_calibration_sample_size = (
                    snapshot.sample_size
                )

        if self.regime_bayesian_root is not None:
            try:
                snapshot = (
                    RegimeBayesianRepository(
                        self.regime_bayesian_root
                    ).load_latest()
                )
            except Exception as exc:
                from lrp.regimes.bayesian_repository import (
                    RegimeBayesianNotFoundError,
                )
                from lrp.regimes.bayesian_serializer import (
                    RegimeBayesianSerializationError,
                )

                if not isinstance(
                    exc,
                    (
                        RegimeBayesianNotFoundError,
                        RegimeBayesianSerializationError,
                    ),
                ):
                    raise
            else:
                applied_bayesian_revision = (
                    snapshot.revision
                )
                applied_bayesian_sample_size = (
                    snapshot.sample_size
                )

        adaptive_pipeline = PredictionPipeline.load(
            evolution_snapshot_root=self.profile_root,
            regime_calibration_snapshot_root=(
                self.regime_calibration_root
            ),
            regime_bayesian_snapshot_root=(
                self.regime_bayesian_root
            ),
        )

        noop_result = noop_pipeline.run(
            statistics_draws,
            request,
            analysis_config=analysis_config,
        )
        adaptive_result = adaptive_pipeline.run(
            statistics_draws,
            request,
            analysis_config=analysis_config,
        )

        noop_payload = prediction_to_dict(
            noop_result
        )
        adaptive_payload = prediction_to_dict(
            adaptive_result
        )

        winning_numbers = self._draw_numbers(draw)
        bonus = self._draw_bonus(draw)

        noop_review = review_prediction(
            noop_payload,
            winning_numbers=winning_numbers,
            bonus=bonus,
        )
        adaptive_review = review_prediction(
            adaptive_payload,
            winning_numbers=winning_numbers,
            bonus=bonus,
        )

        learning_context = self._learning_context(
            state=state,
            round_no=round_no,
        )

        learning_service = (
            self._build_learning_service()
        )
        learning = learning_service.learn(
            context=learning_context,
            review_payload=adaptive_review,
            prediction_payload=adaptive_payload,
            winning_numbers=winning_numbers,
            snapshot_id=f"review-{round_no}",
            policy=self.policy,
            metadata={
                "round": round_no,
                "validation": (
                    "historical_replay"
                ),
            },
            overwrite=False,
        )

        profile_service = (
            self._build_profile_service()
        )
        evolution = profile_service.evolve(
            context=learning.final_context,
            generated_at=datetime.now(
                timezone.utc
            ),
            confidence=self.config.confidence,
        )

        probability_metrics = (
            self._probability_metrics(
                noop_result,
                adaptive_result,
            )
        )

        changed_sets = self._changed_set_count(
            noop_payload,
            adaptive_payload,
        )

        profile = evolution.decision.profile

        row = ReplayRoundResult(
            round_no=round_no,
            seed=seed,
            history_draws=len(bounded),
            noop_best_hits=int(
                noop_review["summary"][
                    "best_main_hits"
                ]
            ),
            adaptive_best_hits=int(
                adaptive_review["summary"][
                    "best_main_hits"
                ]
            ),
            noop_practical_hits=int(
                noop_review["summary"][
                    "practical_best_hits"
                ]
            ),
            adaptive_practical_hits=int(
                adaptive_review["summary"][
                    "practical_best_hits"
                ]
            ),
            noop_avg_jaccard=float(
                noop_payload["diversity"][
                    "avg_jaccard"
                ]
            ),
            adaptive_avg_jaccard=float(
                adaptive_payload["diversity"][
                    "avg_jaccard"
                ]
            ),
            probability_l1_delta=(
                probability_metrics["l1"]
            ),
            probability_max_delta=(
                probability_metrics["max"]
            ),
            changed_probability_count=(
                probability_metrics["changed"]
            ),
            changed_set_count=changed_sets,
            profile_applied=bool(
                evolution.decision.applied
            ),
            profile_revision=(
                profile.revision
                if evolution.decision.applied
                else None
            ),
            profile_sample_size=(
                profile.sample_size
                if evolution.decision.applied
                else None
            ),
            regime_calibration_revision=(
                applied_calibration_revision
            ),
            regime_calibration_sample_size=(
                applied_calibration_sample_size
            ),
            regime_bayesian_revision=(
                applied_bayesian_revision
            ),
            regime_bayesian_sample_size=(
                applied_bayesian_sample_size
            ),
            elapsed_seconds=(
                perf_counter() - started
            ),
        )

        return (
            row,
            ReplayState(
                learning_context=(
                    learning.final_context
                )
            ),
        )

    def _learning_context(
        self,
        *,
        state: object | None,
        round_no: int,
    ) -> LearningContext:
        """Advance the carried learning context to the current round."""

        if isinstance(state, ReplayState):
            return replace(
                state.learning_context,
                round_no=round_no,
            )

        return LearningContext(
            cycle_id=(
                "historical-replay-"
                f"{self.config.start_round}-"
                f"{self.config.end_round}"
            ),
            round_no=round_no,
        )

    def _build_learning_service(
        self,
    ) -> ReviewLearningService:
        persistence = PersistentLearningService(
            FileSnapshotRepository(
                self.learning_root
            )
        )
        runner = PersistentLearningRunner(
            persistence
        )
        if (
            self.regime_calibration_root is None
            and self.regime_bayesian_root is None
        ):
            return ReviewLearningService(runner)

        calibration_updater = None
        calibration_repository = None
        bayesian_updater = None
        bayesian_repository = None

        if self.regime_calibration_root is not None:
            calibration_updater = (
                RegimeCalibrationUpdater(
                    learning_rate_policy=(
                        AdaptiveLearningRatePolicy()
                    )
                )
            )
            calibration_repository = (
                RegimeCalibrationRepository(
                    self.regime_calibration_root
                )
            )

        if self.regime_bayesian_root is not None:
            bayesian_updater = RegimeBayesianUpdater()
            bayesian_repository = (
                RegimeBayesianRepository(
                    self.regime_bayesian_root
                )
            )

        return ReviewLearningService(
            runner,
            regime_reward_calculator=(
                RegimeRewardCalculator()
            ),
            regime_calibration_updater=(
                calibration_updater
            ),
            regime_calibration_repository=(
                calibration_repository
            ),
            regime_bayesian_updater=(
                bayesian_updater
            ),
            regime_bayesian_repository=(
                bayesian_repository
            ),
        )

    def _build_profile_service(
        self,
    ) -> ReviewProfileEvolutionService:
        coordinator = EvolutionCoordinator(
            pipeline=AdaptiveEvolutionPipeline(
                calculator=self.adaptive_calculator
            ),
            policy=AdaptiveWeightPolicy(),
            repository=SnapshotRepository(
                self.profile_root
            ),
        )
        return ReviewProfileEvolutionService(
            coordinator
        )

    @staticmethod
    def _draw_numbers(
        draw: object,
    ) -> tuple[int, ...]:
        value = getattr(
            draw,
            "numbers",
            None,
        )

        if value is None and isinstance(
            draw,
            Mapping,
        ):
            value = draw.get("numbers")

        if value is None:
            value = tuple(
                getattr(draw, f"n{index}")
                for index in range(1, 7)
            )

        return tuple(int(item) for item in value)

    @staticmethod
    def _draw_bonus(
        draw: object,
    ) -> int | None:
        value = getattr(draw, "bonus", None)

        if value is None and isinstance(
            draw,
            Mapping,
        ):
            value = draw.get("bonus")

        return (
            None
            if value is None
            else int(value)
        )

    @staticmethod
    def _probability_metrics(
        noop_result: object,
        adaptive_result: object,
    ) -> dict[str, float | int]:
        noop_vector = (
            noop_result.generation
            .probability_vector
        )
        adaptive_vector = (
            adaptive_result.generation
            .probability_vector
        )

        noop = {
            item.number: item.probability
            for item in noop_vector.probabilities
        }
        adaptive = {
            item.number: item.probability
            for item in adaptive_vector.probabilities
        }

        deltas = tuple(
            abs(
                adaptive[number]
                - noop[number]
            )
            for number in range(1, 46)
        )

        return {
            "l1": sum(deltas),
            "max": max(deltas),
            "changed": sum(
                delta > 1e-15
                for delta in deltas
            ),
        }

    @staticmethod
    def _changed_set_count(
        noop_payload: Mapping[str, Any],
        adaptive_payload: Mapping[str, Any],
    ) -> int:
        noop_sets = {
            str(item["id"]): tuple(
                item["numbers"]
            )
            for item in noop_payload["sets"]
        }
        adaptive_sets = {
            str(item["id"]): tuple(
                item["numbers"]
            )
            for item in adaptive_payload["sets"]
        }

        return sum(
            noop_sets.get(set_id)
            != adaptive_sets.get(set_id)
            for set_id in (
                set(noop_sets)
                | set(adaptive_sets)
            )
        )
