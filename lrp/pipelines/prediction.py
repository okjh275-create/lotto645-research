"""Project A complete prediction pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping
from zoneinfo import ZoneInfo

from lrp.adapters import (
    CandidateAdapter,
    StatisticsAdapter,
    build_statistics_signals,
)
from lrp.contracts import (
    CompatibilityError,
    ContractError,
)
from lrp.core import RuntimeContext
from lrp.ensemble import (
    PipelineRescoringBridge,
    PipelineRescoringResult,
)
from lrp.evolution.integration import (
    EvolutionAdapterFactory,
    EvolutionWeightAdapter,
    NoOpEvolutionWeightAdapter,
)
from lrp.regimes import (
    RegimeFeatureExtractor as GlobalRegimeFeatureExtractor,
    RegimeStabilityPolicy as GlobalRegimeStabilityPolicy,
)
from lrp.regimes.integration import (
    GlobalRegimeAdjustmentAdapter,
    NoOpGlobalRegimeAdjustmentAdapter,
)
from lrp.prediction import (
    ProbabilityFusionEngine,
    RegimeDetector,
)

from .models import (
    PredictionGenerationResult,
    PredictionRequest,
    PredictionResult,
)


_KST = ZoneInfo("Asia/Seoul")


def _analysis_snapshot(
    report: object,
) -> object:
    snapshot = getattr(
        report,
        "snapshot",
        None,
    )

    if snapshot is not None:
        return snapshot

    statistics = getattr(
        report,
        "statistics",
        None,
    )

    if statistics is not None:
        return statistics

    required = (
        "windows",
        "numbers",
        "relationships",
    )

    if all(
        hasattr(report, name)
        for name in required
    ):
        return report

    raise ContractError(
        "Project C analysis output does not expose "
        "a stable snapshot"
    )


@dataclass(frozen=True, slots=True)
class _RegimeFeature:
    """Project C number statistics adapted for RegimeDetector."""

    number: int
    freq_all: float
    freq10: float
    freq20: float
    freq50: float
    gap: float


def _feature_value(
    feature: object,
    name: str,
) -> object:
    if isinstance(feature, Mapping):
        if name not in feature:
            raise ContractError(
                f"Project C number feature is missing {name}"
            )
        return feature[name]

    if not hasattr(feature, name):
        raise ContractError(
            f"Project C number feature is missing {name}"
        )

    return getattr(feature, name)


def _feature_value_any(
    feature: object,
    *names: str,
) -> object:
    for name in names:
        if isinstance(feature, Mapping):
            if name in feature:
                return feature[name]
        elif hasattr(feature, name):
            return getattr(feature, name)

    raise ContractError(
        "Project C number feature is missing "
        + " or ".join(names)
    )


def _regime_features(
    snapshot: object,
) -> tuple[_RegimeFeature, ...]:
    """Adapt canonical Project C features for RegimeDetector."""

    numbers = getattr(
        snapshot,
        "numbers",
        None,
    )

    if numbers is None:
        raise ContractError(
            "Project C snapshot does not expose "
            "number features"
        )

    if isinstance(numbers, Mapping):
        source_features = tuple(
            numbers.values()
        )
    else:
        try:
            source_features = tuple(numbers)
        except TypeError as exc:
            raise ContractError(
                "Project C snapshot numbers "
                "must be iterable"
            ) from exc

    if len(source_features) != 45:
        raise ContractError(
            "Project C snapshot must contain "
            "exactly 45 number features"
        )

    converted: list[_RegimeFeature] = []

    for feature in source_features:
        try:
            converted.append(
                _RegimeFeature(
                    number=int(
                        _feature_value(
                            feature,
                            "number",
                        )
                    ),
                    freq_all=float(
                        _feature_value_any(
                            feature,
                            "total_frequency",
                            "freq_all",
                        )
                    ),
                    freq10=float(
                        _feature_value_any(
                            feature,
                            "short_frequency",
                            "freq10",
                        )
                    ),
                    freq20=float(
                        _feature_value_any(
                            feature,
                            "mid_frequency",
                            "freq20",
                        )
                    ),
                    freq50=float(
                        _feature_value_any(
                            feature,
                            "long_frequency",
                            "freq50",
                        )
                    ),
                    gap=float(
                        _feature_value(
                            feature,
                            "gap",
                        )
                    ),
                )
            )
        except ContractError:
            raise
        except (TypeError, ValueError) as exc:
            raise ContractError(
                "Project C regime feature fields "
                "must be numeric"
            ) from exc

    converted.sort(
        key=lambda item: item.number
    )

    actual_numbers = tuple(
        item.number
        for item in converted
    )

    expected_numbers = tuple(range(1, 46))

    if actual_numbers != expected_numbers:
        raise ContractError(
            "Project C number features must "
            "contain exactly numbers 1..45"
        )

    return tuple(converted)

def _pair_affinities(
    snapshot: object,
) -> Mapping[tuple[int, int], float]:
    relationships = getattr(
        snapshot,
        "relationships",
        None,
    )

    if relationships is None:
        return {}

    graph = getattr(
        relationships,
        "affinity_graph",
        None,
    )

    if graph is None:
        return {}

    values = getattr(
        graph,
        "pair_affinity",
        None,
    )

    if not isinstance(values, Mapping):
        return {}

    normalized: dict[
        tuple[int, int],
        float,
    ] = {}

    for key, value in values.items():
        if (
            isinstance(key, tuple)
            and len(key) == 2
        ):
            try:
                left = int(key[0])
                right = int(key[1])
                score = float(value)
            except (TypeError, ValueError):
                continue

            normalized[
                (left, right)
            ] = score

    return normalized


@dataclass(slots=True)
class PredictionPipeline:
    """Orchestrate Projects C, D, and optional Project E."""

    statistics: StatisticsAdapter
    candidate: CandidateAdapter
    ensemble: PipelineRescoringBridge | None = None
    evolution: EvolutionWeightAdapter[object] = field(
        default_factory=NoOpEvolutionWeightAdapter
    )
    global_regime_adjustment: (
        GlobalRegimeAdjustmentAdapter[object]
    ) = field(
        default_factory=NoOpGlobalRegimeAdjustmentAdapter
    )

    def __post_init__(self) -> None:
        if (
            self.ensemble is not None
            and not isinstance(
                self.ensemble,
                PipelineRescoringBridge,
            )
        ):
            raise ContractError(
                "ensemble must be a "
                "PipelineRescoringBridge or None"
            )

        if not isinstance(
            self.evolution,
            EvolutionWeightAdapter,
        ):
            raise ContractError(
                "evolution must implement "
                "EvolutionWeightAdapter"
            )

        if not isinstance(
            self.global_regime_adjustment,
            GlobalRegimeAdjustmentAdapter,
        ):
            raise ContractError(
                "global_regime_adjustment must implement "
                "GlobalRegimeAdjustmentAdapter"
            )

    @classmethod
    def load(
        cls,
        *,
        ensemble: (
            PipelineRescoringBridge
            | None
        ) = None,
        evolution: (
            EvolutionWeightAdapter[object]
            | None
        ) = None,
        evolution_snapshot_root: (
            str | Path | None
        ) = None,
        global_regime_adjustment: (
            GlobalRegimeAdjustmentAdapter[object]
            | None
        ) = None,
    ) -> "PredictionPipeline":
        resolved_evolution = (
            EvolutionAdapterFactory.build(
                evolution=evolution,
                snapshot_root=(
                    evolution_snapshot_root
                ),
            )
        )

        resolved_global_regime_adjustment = (
            global_regime_adjustment
            if global_regime_adjustment is not None
            else NoOpGlobalRegimeAdjustmentAdapter()
        )

        return cls(
            statistics=StatisticsAdapter.load(),
            candidate=CandidateAdapter.load(),
            ensemble=ensemble,
            evolution=resolved_evolution,
            global_regime_adjustment=(
                resolved_global_regime_adjustment
            ),
        )

    def analyze(
        self,
        draws: Iterable[object],
        *,
        analysis_config: object | None = None,
    ) -> object:
        return self.statistics.analyze_all(
            draws,
            config=analysis_config,
        )

    def generate_from_snapshot(
        self,
        snapshot: object,
        request: PredictionRequest,
    ) -> PredictionGenerationResult:
        if not isinstance(
            request,
            PredictionRequest,
        ):
            raise ContractError(
                "request must be a PredictionRequest"
            )

        bridge = build_statistics_signals(
            snapshot
        )

        global_regime_features = (
            GlobalRegimeFeatureExtractor().extract(
                bridge
            )
        )

        global_regime_context = (
            GlobalRegimeStabilityPolicy().decide(
                global_regime_features
            )
        )

        statistics_payload = bridge.to_dict()

        contract_report = (
            self.candidate.validate_statistics(
                statistics_payload
            )
        )

        if not bool(
            getattr(
                contract_report,
                "compatible",
                False,
            )
        ):
            reasons = getattr(
                contract_report,
                "reasons",
                (),
            )

            raise CompatibilityError(
                "Project C to D statistics contract "
                "failed: "
                + ", ".join(
                    str(reason)
                    for reason in reasons
                )
            )

        number_signals = (
            self.candidate.number_signals(
                statistics_payload
            )
        )

        regime_profile = RegimeDetector().detect(
            _regime_features(snapshot),
            round_no=request.round_no,
            metadata={
                "source": "Project C",
                "pipeline": type(self).__name__,
            },
        )

        probability_vector = (
            ProbabilityFusionEngine().build(
                regime_profile,
                metadata={
                    "pipeline": type(self).__name__,
                },
            )
        )

        probability_vector = (
            self._adjust_probability_vector(
                probability_vector,
                request=request,
            )
        )

        probability_vector = (
            self._adjust_global_regime_probability_vector(
                probability_vector,
                global_regime=global_regime_context,
                request=request,
            )
        )

        probabilities = (
            self.candidate.probability_mapping(
                probability_vector
            )
        )

        module = self.candidate.module

        candidate_config = module.CandidateConfig(
            seed=request.seed,
            temperature=request.temperature,
            candidate_count=(
                request.candidate_count
            ),
            max_attempts_multiplier=(
                request.max_attempts_multiplier
            ),
        )

        risk_config = module.RiskFilterConfig(
            sum_min=request.sum_min,
            sum_max=request.sum_max,
            min_odd=2,
            max_odd=4,
            low_high_min_each=1,
            max_consecutive_run=2,
            max_same_ending=2,
            max_overlap_previous=1,
            min_long_gap_inclusion=1,
            max_same_decade=3,
        )

        candidates = (
            self.candidate.generate_candidates(
                probabilities,
                candidate_config=(
                    candidate_config
                ),
                risk_config=risk_config,
                previous_numbers=(
                    request.previous_numbers
                ),
                long_gap_numbers=(
                    request.long_gap_numbers
                ),
            )
        )

        return PredictionGenerationResult(
            request=request,
            windows=bridge.windows,
            probabilities=probabilities,
            statistics_contract=(
                contract_report
            ),
            number_signals=number_signals,
            candidates=tuple(candidates),
            statistics_version=(
                self.statistics.version
            ),
            candidate_version=(
                self.candidate.version
            ),
            regime_profile=regime_profile,
            probability_vector=(
                probability_vector
            ),
            global_regime_context=(
                global_regime_context
            ),
        )

    def _adjust_probability_vector(
        self,
        probability_vector: object,
        *,
        request: PredictionRequest,
    ) -> object:
        adjusted = self.evolution.adjust(
            probability_vector,
            round_no=request.round_no,
            seed=request.seed,
        )

        if adjusted is None:
            raise ContractError(
                "evolution adapter returned None"
            )

        return adjusted

    def _adjust_global_regime_probability_vector(
        self,
        probability_vector: object,
        *,
        global_regime: object | None,
        request: PredictionRequest,
    ) -> object:
        adjusted = self.global_regime_adjustment.adjust(
            probability_vector,
            global_regime=global_regime,
            round_no=request.round_no,
            seed=request.seed,
        )

        if adjusted is None:
            raise ContractError(
                "global regime adjustment adapter returned None"
            )

        return adjusted
    def _apply_ensemble(
        self,
        scored: tuple[object, ...],
        *,
        round_no: int,
    ) -> PipelineRescoringResult | None:
        if self.ensemble is None:
            return None

        return self.ensemble.apply(
            scored,
            round_no=round_no,
        )

    def complete_from_snapshot(
        self,
        snapshot: object,
        request: PredictionRequest,
    ) -> PredictionResult:
        generation = (
            self.generate_from_snapshot(
                snapshot,
                request,
            )
        )

        module = self.candidate.module

        score_weights = module.ScoreWeights(
            recency=request.weights["recency"],
            frequency=request.weights[
                "frequency"
            ],
            gap_reversion=request.weights[
                "gap_reversion"
            ],
            pair_graph=request.weights[
                "pair_graph"
            ],
            terminal_dispersion=(
                request.weights[
                    "terminal_dispersion"
                ]
            ),
            sum_band=request.weights[
                "sum_band"
            ],
            parity_balance=request.weights[
                "parity_balance"
            ],
        )

        scored = self.candidate.score_candidates(
            generation.candidates,
            signals=(
                generation.number_signals
            ),
            weights=score_weights,
            pair_affinities=(
                _pair_affinities(snapshot)
            ),
            sum_band_center=145.0,
            sum_band_scale=35.0,
        )
        scored = tuple(scored)

        if not scored:
            raise ContractError(
                "Project D produced no "
                "scored candidates"
            )

        ensemble_result = (
            self._apply_ensemble(
                scored,
                round_no=request.round_no,
            )
        )

        effective_scored = (
            ensemble_result
            .effective_candidates
            if ensemble_result is not None
            else scored
        )

        ranking_config = module.RankingConfig(
            score_weight=0.70,
            robustness_weight=0.15,
            diversity_weight=0.15,
            pareto_penalty=0.05,
        )

        ranking = self.candidate.rank_candidates(
            effective_scored,
            config=ranking_config,
        )

        diversity_config = (
            module.DiversityConfig(
                k=request.top_k,
                mmr_lambda=request.mmr_lambda,
                jaccard_max=(
                    request.jaccard_max
                ),
                max_overlap_between_sets=(
                    request
                    .max_overlap_between_sets
                ),
            )
        )

        diversity = (
            self.candidate
            .select_diverse_candidates(
                effective_scored,
                config=diversity_config,
            )
        )

        practical_config = (
            module.PracticalSelectionConfig(
                k=request.practical_k,
                preferred_sum_min=(
                    request.preferred_sum_min
                ),
                preferred_sum_max=(
                    request.preferred_sum_max
                ),
                max_overlap_previous_draw=1,
                require_no_risk_flags=True,
                jaccard_max=(
                    request.jaccard_max
                ),
                max_overlap_between_sets=(
                    request
                    .max_overlap_between_sets
                ),
            )
        )

        practical = (
            self.candidate
            .select_practical_sets(
                ranking,
                config=practical_config,
                previous_numbers=(
                    request.previous_numbers
                ),
            )
        )

        return PredictionResult(
            generation=generation,
            scored_candidates=tuple(
                effective_scored
            ),
            ranking=ranking,
            diversity=diversity,
            practical=practical,
            generated_at_kst=(
                datetime.now(_KST)
            ),
            ensemble=ensemble_result,
        )

    def run(
        self,
        draws: Iterable[object],
        request: PredictionRequest,
        *,
        analysis_config: object | None = None,
    ) -> PredictionResult:
        report = self.analyze(
            draws,
            analysis_config=analysis_config,
        )
        snapshot = _analysis_snapshot(report)

        return self.complete_from_snapshot(
            snapshot,
            request,
        )

    def runtime_context(
        self,
        request: PredictionRequest,
    ) -> RuntimeContext:
        return RuntimeContext.create(
            seed=request.seed,
            execution_id=(
                "lrp-prediction-round-"
                f"{request.round_no}"
            ),
            parameters=request.to_dict(),
        )
