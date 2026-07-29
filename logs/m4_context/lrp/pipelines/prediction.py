"""Project A complete prediction pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

from lrp.adapters import (
    CandidateAdapter,
    StatisticsAdapter,
    build_statistics_signals,
)
from lrp.contracts import CompatibilityError, ContractError
from lrp.core import RuntimeContext

from .models import (
    PredictionGenerationResult,
    PredictionRequest,
    PredictionResult,
)
from .probability import build_probability_vector


_KST = ZoneInfo("Asia/Seoul")


def _analysis_snapshot(report: object) -> object:
    snapshot = getattr(report, "snapshot", None)
    if snapshot is not None:
        return snapshot

    statistics = getattr(report, "statistics", None)
    if statistics is not None:
        return statistics

    required = (
        "windows",
        "numbers",
        "relationships",
    )
    if all(hasattr(report, name) for name in required):
        return report

    raise ContractError(
        "Project C analysis output does not expose a stable snapshot"
    )


def _pair_affinities(
    snapshot: object,
) -> Mapping[tuple[int, int], float]:
    relationships = getattr(snapshot, "relationships", None)
    if relationships is None:
        return {}

    graph = getattr(relationships, "affinity_graph", None)
    if graph is None:
        return {}

    values = getattr(graph, "pair_affinity", None)
    if not isinstance(values, Mapping):
        return {}

    normalized: dict[tuple[int, int], float] = {}

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

            normalized[(left, right)] = score

    return normalized


@dataclass(slots=True)
class PredictionPipeline:
    """Orchestrate Project C analysis and Project D prediction."""

    statistics: StatisticsAdapter
    candidate: CandidateAdapter

    @classmethod
    def load(cls) -> "PredictionPipeline":
        return cls(
            statistics=StatisticsAdapter.load(),
            candidate=CandidateAdapter.load(),
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
        if not isinstance(request, PredictionRequest):
            raise ContractError(
                "request must be a PredictionRequest"
            )

        bridge = build_statistics_signals(snapshot)
        statistics_payload = bridge.to_dict()

        contract_report = self.candidate.validate_statistics(
            statistics_payload
        )

        if not bool(
            getattr(contract_report, "compatible", False)
        ):
            reasons = getattr(contract_report, "reasons", ())
            raise CompatibilityError(
                "Project C to D statistics contract failed: "
                + ", ".join(str(reason) for reason in reasons)
            )

        number_signals = self.candidate.number_signals(
            statistics_payload
        )

        probabilities = build_probability_vector(
            number_signals,
            weights=request.weights,
        )

        module = self.candidate.module

        candidate_config = module.CandidateConfig(
            seed=request.seed,
            temperature=request.temperature,
            candidate_count=request.candidate_count,
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

        candidates = self.candidate.generate_candidates(
            probabilities,
            candidate_config=candidate_config,
            risk_config=risk_config,
            previous_numbers=request.previous_numbers,
            long_gap_numbers=request.long_gap_numbers,
        )

        return PredictionGenerationResult(
            request=request,
            windows=bridge.windows,
            probabilities=probabilities,
            statistics_contract=contract_report,
            number_signals=number_signals,
            candidates=tuple(candidates),
            statistics_version=self.statistics.version,
            candidate_version=self.candidate.version,
        )

    def complete_from_snapshot(
        self,
        snapshot: object,
        request: PredictionRequest,
    ) -> PredictionResult:
        generation = self.generate_from_snapshot(
            snapshot,
            request,
        )

        module = self.candidate.module

        score_weights = module.ScoreWeights(
            recency=request.weights["recency"],
            frequency=request.weights["frequency"],
            gap_reversion=request.weights["gap_reversion"],
            pair_graph=request.weights["pair_graph"],
            terminal_dispersion=(
                request.weights["terminal_dispersion"]
            ),
            sum_band=request.weights["sum_band"],
            parity_balance=request.weights["parity_balance"],
        )

        scored = self.candidate.score_candidates(
            generation.candidates,
            signals=generation.number_signals,
            weights=score_weights,
            pair_affinities=_pair_affinities(snapshot),
            sum_band_center=145.0,
            sum_band_scale=35.0,
        )
        scored = tuple(scored)

        if not scored:
            raise ContractError(
                "Project D produced no scored candidates"
            )

        ranking_config = module.RankingConfig(
            score_weight=0.70,
            robustness_weight=0.15,
            diversity_weight=0.15,
            pareto_penalty=0.05,
        )

        ranking = self.candidate.rank_candidates(
            scored,
            config=ranking_config,
        )

        diversity_config = module.DiversityConfig(
            k=request.top_k,
            mmr_lambda=request.mmr_lambda,
            jaccard_max=request.jaccard_max,
            max_overlap_between_sets=(
                request.max_overlap_between_sets
            ),
        )

        diversity = self.candidate.select_diverse_candidates(
            scored,
            config=diversity_config,
        )

        practical_config = module.PracticalSelectionConfig(
            k=request.practical_k,
            preferred_sum_min=request.preferred_sum_min,
            preferred_sum_max=request.preferred_sum_max,
            max_overlap_previous_draw=1,
            require_no_risk_flags=True,
            jaccard_max=request.jaccard_max,
            max_overlap_between_sets=(
                request.max_overlap_between_sets
            ),
        )

        practical = self.candidate.select_practical_sets(
            ranking,
            config=practical_config,
            previous_numbers=request.previous_numbers,
        )

        return PredictionResult(
            generation=generation,
            scored_candidates=scored,
            ranking=ranking,
            diversity=diversity,
            practical=practical,
            generated_at_kst=datetime.now(_KST),
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
                f"lrp-prediction-round-{request.round_no}"
            ),
            parameters=request.to_dict(),
        )
