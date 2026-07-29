"""Public learning package API."""

from .adaptive_engine import (
    AdaptiveWeightConfig,
    AdaptiveWeightEngine,
    AdaptiveWeightFactors,
)
from .adaptive_models import (
    AdaptiveRevision,
    AdaptiveWeight,
    AdaptiveWeightDataset,
)
from .adaptive_repository import AdaptiveWeightRepository
from .aggregator import StrategyAggregationSummary, StrategyAggregator
from .evaluator import determine_prize_rank, evaluate_prediction
from .learning_facade import LearningFacade
from .models import PredictionRecord, ResultRecord, ReviewRecord
from .ranking import (
    RankingConfig,
    RankingWeights,
    StrategyPerformancePoint,
    StrategyRanking,
    StrategyRankingEngine,
)
from .ranking_repository import RankingDataset, RankingRepository
from .repository import LearningRepository
from .service import IncrementalReviewSummary, LearningService
from .strategy_stats import StrategyStatistics

__all__ = [
    "AdaptiveRevision",
    "AdaptiveWeight",
    "AdaptiveWeightConfig",
    "AdaptiveWeightDataset",
    "AdaptiveWeightEngine",
    "AdaptiveWeightFactors",
    "AdaptiveWeightRepository",
    "IncrementalReviewSummary",
    "LearningFacade",
    "LearningRepository",
    "LearningService",
    "PredictionRecord",
    "RankingConfig",
    "RankingDataset",
    "RankingRepository",
    "RankingWeights",
    "ResultRecord",
    "ReviewRecord",
    "StrategyAggregationSummary",
    "StrategyAggregator",
    "StrategyPerformancePoint",
    "StrategyRanking",
    "StrategyRankingEngine",
    "StrategyStatistics",
    "determine_prize_rank",
    "evaluate_prediction",
]
