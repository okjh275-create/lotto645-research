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
from .adaptive_report import (
    AdaptiveWeightExplanation,
    AdaptiveWeightReport,
    AdaptiveWeightReporter,
)
from .adaptive_repository import AdaptiveWeightRepository
from .aggregator import (
    StrategyAggregationSummary,
    StrategyAggregator,
)
from .evaluator import (
    determine_prize_rank,
    evaluate_prediction,
)
from .learning_facade import LearningFacade
from .models import (
    PredictionRecord,
    ResultRecord,
    ReviewRecord,
)
from .performance import (
    PerformanceAnalyzer,
    StrategyPerformanceReport,
    StrategyPerformanceSummary,
)
from .ranking import (
    RankingConfig,
    RankingWeights,
    StrategyPerformancePoint,
    StrategyRanking,
    StrategyRankingEngine,
)
from .ranking_repository import (
    RankingDataset,
    RankingRepository,
)
from .repository import LearningRepository
from .service import (
    IncrementalReviewSummary,
    LearningService,
)
from .snapshot import (
    LearningSnapshot,
    LearningSnapshotWriter,
)
from .strategy_stats import StrategyStatistics


__all__ = [
    "AdaptiveRevision",
    "AdaptiveWeight",
    "AdaptiveWeightConfig",
    "AdaptiveWeightDataset",
    "AdaptiveWeightEngine",
    "AdaptiveWeightExplanation",
    "AdaptiveWeightFactors",
    "AdaptiveWeightReport",
    "AdaptiveWeightReporter",
    "AdaptiveWeightRepository",
    "IncrementalReviewSummary",
    "LearningFacade",
    "LearningRepository",
    "LearningService",
    "LearningSnapshot",
    "LearningSnapshotWriter",
    "PerformanceAnalyzer",
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
    "StrategyPerformanceReport",
    "StrategyPerformanceSummary",
    "StrategyRanking",
    "StrategyRankingEngine",
    "StrategyStatistics",
    "determine_prize_rank",
    "evaluate_prediction",
]
