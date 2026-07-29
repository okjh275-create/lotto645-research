"""Project E Learning and Ensemble Intelligence API."""

from __future__ import annotations

from .adapters import (
    ranking_from_m6,
    rankings_from_m6,
    statistic_from_m6,
    statistics_from_m6,
    weight_from_m6,
    weights_from_m6,
    weights_from_rankings,
)
from .engine import EnsembleEngine, ScoreReader
from .explain import (
    explain_candidate,
    explain_result,
)
from .features import (
    StrategyFeatureVector,
    build_feature_catalog,
    build_strategy_feature,
    trend_value,
)
from .integration import (
    LearningSnapshotRepository,
    PipelineRescoringBridge,
    PipelineRescoringResult,
    SnapshotLoader,
    recursive_base_score_reader,
    recursive_strategy_resolver,
    replace_candidate_score,
)
from .models import (
    EnsembleCandidateScore,
    EnsembleConfig,
    EnsembleResult,
    StrategyWeight,
    normalize_strategy_weights,
)
from .repository import (
    EmptyStrategyWeightRepository,
    InMemoryStrategyWeightRepository,
    LearningSnapshotWeightRepository,
    M6LearningSnapshotRepository,
    SnapshotRepository,
    StrategyWeightRepository,
)
from .rescoring import (
    BaseScoreReader,
    CandidateRescorer,
    RescoredCandidate,
    RescoringConfig,
    RescoringResult,
    ScoreContribution,
    StrategyKey,
    StrategyResolver,
    default_base_score_reader,
    default_strategy_resolver,
)
from .snapshot import (
    LearningSnapshot,
    StrategyRankingSnapshot,
    StrategyStatisticSnapshot,
    ordered_rankings,
    ordered_statistics,
)
from .version import __version__


__all__ = [
    "BaseScoreReader",
    "CandidateRescorer",
    "EmptyStrategyWeightRepository",
    "EnsembleCandidateScore",
    "EnsembleConfig",
    "EnsembleEngine",
    "EnsembleResult",
    "InMemoryStrategyWeightRepository",
    "LearningSnapshot",
    "LearningSnapshotRepository",
    "LearningSnapshotWeightRepository",
    "M6LearningSnapshotRepository",
    "PipelineRescoringBridge",
    "PipelineRescoringResult",
    "RescoredCandidate",
    "RescoringConfig",
    "RescoringResult",
    "ScoreContribution",
    "ScoreReader",
    "SnapshotLoader",
    "SnapshotRepository",
    "StrategyFeatureVector",
    "StrategyKey",
    "StrategyRankingSnapshot",
    "StrategyResolver",
    "StrategyStatisticSnapshot",
    "StrategyWeight",
    "StrategyWeightRepository",
    "__version__",
    "build_feature_catalog",
    "build_strategy_feature",
    "default_base_score_reader",
    "default_strategy_resolver",
    "explain_candidate",
    "explain_result",
    "normalize_strategy_weights",
    "ordered_rankings",
    "ordered_statistics",
    "ranking_from_m6",
    "rankings_from_m6",
    "recursive_base_score_reader",
    "recursive_strategy_resolver",
    "replace_candidate_score",
    "statistic_from_m6",
    "statistics_from_m6",
    "trend_value",
    "weight_from_m6",
    "weights_from_m6",
    "weights_from_rankings",
]
